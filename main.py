# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""
ComputeField Machine WebSocket client.

Connects to gpu-manager, registers with hardware capabilities, and handles
training and inference jobs proxied from orchestrator:
  Phase 1 (Loading)  — receives bounded dataset_shards and stages the first one
  Phase 2 (Execution)— trains or infers while later shards prefetch in parallel
"""

import asyncio
import json
import logging
import os
import tempfile
import traceback
from collections.abc import Callable
from functools import partial
from typing import BinaryIO

import requests
import websockets
from capabilities import default_machine_name, get_capabilities
from config import settings
from delta import DeltaAborted, compute_delta_file
from executor import DownloadAborted, Executor
from monitor import start_monitor
from sandbox_runtime import self_test as sandbox_self_test
from speedtest import run_speed_test
from websockets.exceptions import ConnectionClosed
from workspace import WorkRoot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# A large model/dataset download during the Loading phase can saturate a slow
# or congested link enough that the low-level WS ping's pong doesn't arrive
# within the window, even though the connection is otherwise fine — the
# client then tears down the connection with "keepalive ping timeout" and
# gpu-manager reports the host lost mid-run. Generous on purpose (matches
# gpu-manager's own HEARTBEAT_TIMEOUT default) so an ordinary large transfer
# doesn't cause a false-positive disconnect.
WS_PING_INTERVAL = 30
WS_PING_TIMEOUT = 90

# websockets' default incoming-message limit is 1 MiB. A task/update_model
# message carries one presigned URL (~400-500 bytes) per assigned shard —
# a run with a few thousand shards exceeds the default, which closes the
# connection with "message too big" and wedges the host in a reconnect loop
# where the rejoin re-send hits the same limit forever. orchestrator's
# own gpu-manager client uses the same 64 MiB ceiling.
WS_MAX_SIZE = 64 * 1024 * 1024
SESSION_CLEANUP_TIMEOUT = 35


def _connect(uri: str):
    return websockets.connect(
        uri,
        proxy=True,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT,
        max_size=WS_MAX_SIZE,
    )


class _WsLogHandler(logging.Handler):
    """Forwards Python log records to the ComputeField Machine WebSocket send queue."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._emit_fn: Callable[[str], None] | None = None

    def set_emit(self, fn: Callable[[str], None] | None) -> None:
        self._emit_fn = fn

    def emit(self, record: logging.LogRecord) -> None:
        fn = self._emit_fn
        if fn is None:
            return
        if record.levelno >= logging.ERROR:
            prefix = "[ERROR]"
        elif record.levelno >= logging.WARNING:
            prefix = "[WARN]"
        else:
            prefix = "[INFO]"
        try:
            fn(f"{prefix} {record.getMessage()}")
            if record.exc_info:
                for line in traceback.format_exception(*record.exc_info):
                    for subline in line.rstrip().splitlines():
                        fn(f"[ERROR] {subline}")
        except Exception:
            logger.debug("Could not forward a log record", exc_info=True)


_ws_handler = _WsLogHandler()
logging.getLogger().addHandler(_ws_handler)


class _Outbound:
    """Lease-aware, bounded outbound channel for one broker session."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self.lease_id: str | None = None

    def leased(self, message: dict, lease_id: str | None = None) -> dict:
        if message.get("type") == "heartbeat":
            return message
        lease = lease_id if lease_id is not None else self.lease_id
        return {**message, "lease_id": lease} if lease else message

    def offer(self, message: dict) -> None:
        try:
            self.queue.put_nowait(self.leased(message))
        except asyncio.QueueFull:
            pass

    def emit_stats(self, data: dict) -> None:
        self.loop.call_soon_threadsafe(self.offer, {"type": "stats", "kind": "training", **data})
        if data.get("kind") == "inference":
            summary = ", ".join(f"{key}={value}" for key, value in data.items() if key != "kind")
            self.loop.call_soon_threadsafe(self.offer, {"type": "log", "text": f"[INFO] Inference stats: {summary}"})

    def emit_log(self, text: str) -> None:
        self.loop.call_soon_threadsafe(self.offer, {"type": "log", "text": text})

    def emit_hardware(self, stats: dict) -> None:
        self.loop.call_soon_threadsafe(self.offer, {"type": "stats", "kind": "hardware", **stats})

    async def sender(self, ws) -> None:
        while True:
            await ws.send(json.dumps(await self.queue.get()))

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self.queue.put({"type": "heartbeat"})


async def _register_machine(ws, caps: dict) -> None:
    registration = {
        "type": "register",
        "id": settings.host_id,
        "capabilities": caps,
        "client_version": settings.client_version,
    }
    if settings.saved_credential:
        registration["credential"] = settings.saved_credential
    await ws.send(json.dumps(registration))
    response = json.loads(await ws.recv())
    if response.get("type") != "registered":
        raise RuntimeError(f"Registration rejected: {response}")
    logger.info("Registered  id=%s", settings.host_id)


async def _run_initial_speed_test(ws) -> None:
    if settings.speedtest_payload_mb <= 0:
        return
    try:
        result = await run_speed_test(ws, settings.speedtest_payload_mb)
        logger.info(
            "Speed test: upload=%.1f Mbps  download=%.1f Mbps  (payload=%d MB)",
            result["upload_mbps"],
            result["download_mbps"],
            settings.speedtest_payload_mb,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "speedtest_result",
                    "upload_mbps": result["upload_mbps"],
                    "download_mbps": result["download_mbps"],
                }
            )
        )
    except Exception:
        logger.warning("Speed test failed — continuing without it", exc_info=True)


async def _cleanup_session(
    loop: asyncio.AbstractEventLoop,
    executor: Executor,
    background: set[asyncio.Task],
    service_tasks: list[asyncio.Task],
) -> None:
    for task in service_tasks:
        task.cancel()
    await loop.run_in_executor(None, executor.stop)
    if background:
        _, pending = await asyncio.wait(background, timeout=SESSION_CLEANUP_TIMEOUT)
        if pending:
            logger.critical("Background task cleanup timed out; restarting Machine")
            os._exit(75)
    try:
        await loop.run_in_executor(None, executor.shutdown)
    except RuntimeError as exc:
        logger.critical("Session cleanup was not safe; restarting Machine: %s", exc)
        os._exit(75)


async def main() -> None:
    uri = settings.manager_url
    if not settings.host_id or not uri or (settings.require_auth and not settings.saved_credential):
        raise RuntimeError(f"Machine is not paired. Run '{settings.cli_name} pair CODE'.")
    work_root = WorkRoot(settings.work_dir)
    try:
        sandboxed = settings.sharing_supported
        sandbox_backend = sandbox_self_test(str(work_root.path)) if sandboxed else "none"
        caps = get_capabilities(settings.compute_mode)
        caps["machine_name"] = settings.machine_name or default_machine_name(caps)
        caps["sharing_enabled"] = settings.sharing_enabled
        caps["workload_isolation"] = sandbox_backend
        caps["transfer_route"] = "internal" if settings.machine_transfer_route == "internal" else "external"
        logger.info(
            "Starting ComputeField Machine  id=%s  manager=%s  compute=%s  isolation=%s",
            settings.host_id,
            uri,
            caps.get("compute_mode"),
            sandbox_backend,
        )
        while True:
            try:
                async with _connect(uri) as ws:
                    await _session(ws, caps, str(work_root.path), sandboxed=sandboxed)
            except (ConnectionClosed, OSError) as exc:
                logger.warning("Connection lost: %s — reconnecting in 5 s", exc)
            except Exception as exc:
                logger.exception("Unexpected error: %s — reconnecting in 5 s", exc)
            await asyncio.sleep(5)
    finally:
        work_root.close()


async def _session(ws, caps: dict, work_dir: str | None = None, *, sandboxed: bool = False) -> None:
    loop = asyncio.get_running_loop()
    outbound = _Outbound(loop)
    await _register_machine(ws, caps)
    await _run_initial_speed_test(ws)
    executor = Executor(
        emit_stats=outbound.emit_stats,
        emit_log=outbound.emit_log,
        host_id=settings.host_id,
        shard_cache_max_mb=settings.shard_cache_max_mb,
        work_dir=work_dir,
        sandboxed=sandboxed,
    )
    _ws_handler.set_emit(outbound.emit_log)
    monitor_stop = start_monitor(settings.hardware_stats_interval, outbound.emit_hardware)
    service_tasks = [asyncio.create_task(outbound.sender(ws)), asyncio.create_task(outbound.heartbeat())]
    transfer_lock: asyncio.Lock = asyncio.Lock()
    bg_tasks: set[asyncio.Task] = set()
    try:
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Malformed message ignored: %.200r", raw)
                continue
            try:
                if msg.get("type") == "task" and msg.get("lease_id"):
                    outbound.lease_id = str(msg["lease_id"])
                await _handle(ws, msg, executor, outbound.queue, loop, transfer_lock, bg_tasks)
                if msg.get("type") == "reset":
                    outbound.lease_id = None
            except Exception:
                logger.exception("Handler error for %r — session continues", msg.get("type"))
    finally:
        _ws_handler.set_emit(None)
        monitor_stop.set()
        await _cleanup_session(loop, executor, bg_tasks, service_tasks)


def _spawn_background(bg_tasks: set[asyncio.Task], coro) -> None:
    """Run session work with a strong reference for bounded cleanup."""
    task = asyncio.create_task(coro)
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)


class _MessageContext:
    def __init__(
        self,
        msg: dict,
        executor: Executor,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        transfer_lock: asyncio.Lock,
        background: set[asyncio.Task],
    ) -> None:
        self.msg = msg
        self.executor = executor
        self.queue = queue
        self.loop = loop
        self.transfer_lock = transfer_lock
        self.background = background
        self.lease_id = str(msg["lease_id"]) if msg.get("lease_id") else None

    def leased(self, message: dict) -> dict:
        return {**message, "lease_id": self.lease_id} if self.lease_id else message

    async def put(self, message: dict) -> None:
        await self.queue.put(self.leased(message))

    def spawn(self, coroutine) -> None:
        _spawn_background(self.background, coroutine)


async def _load_task(context: _MessageContext) -> None:
    logger.info("Loading task …")

    async def load() -> None:
        async with context.transfer_lock:
            try:
                await context.loop.run_in_executor(
                    None,
                    context.executor.load,
                    context.msg["model_url"],
                    context.msg["dataset_shards"],
                    context.msg["code"],
                    int(context.msg.get("batch_size", 32)),
                    context.msg.get("params") or {},
                    int(context.msg.get("initial_shard_index", 0)),
                )
                await context.put({"type": "ready"})
                logger.info("Task loaded — sent ready")
            except DownloadAborted:
                logger.info("Task load aborted by stop")
            except Exception as exc:
                logger.exception("Task load failed")
                await context.put({"type": "error", "message": str(exc)})

    context.spawn(load())


async def _run_training(context: _MessageContext) -> None:
    steps = int(context.msg.get("steps", 100))
    round_num = int(context.msg.get("round_num", 0))
    total_rounds = int(context.msg.get("total_rounds", 1))
    step_offset = max(0, int(context.msg.get("step_offset", 0)))
    label = f"{round_num + 1}/{total_rounds}" if total_rounds > 0 else str(round_num + 1)
    logger.info("Starting fine-tuning  round=%s  steps=%d", label, steps)
    context.executor.run(
        steps=steps,
        round_num=round_num,
        total_rounds=total_rounds,
        step_offset=step_offset,
    )

    async def complete() -> None:
        await context.loop.run_in_executor(None, context.executor.wait)
        if context.executor.execution_error:
            await context.put({"type": "error", **context.executor.execution_error})
        else:
            await context.put({"type": "run_complete"})
            logger.info("Fine-tuning complete (round %s)", label)

    context.spawn(complete())


async def _run_inference(context: _MessageContext) -> None:
    logger.info("Starting inference")
    context.executor.run(steps=0, round_num=0, total_rounds=1, mode="inference")

    async def complete() -> None:
        await context.loop.run_in_executor(None, context.executor.wait)
        if context.executor.execution_error:
            await context.put({"type": "error", **context.executor.execution_error})
            return
        report = context.executor.report
        if report:
            await context.put({"type": "stats", "kind": "inference", "final": True, **report})
            summary = ", ".join(f"{key}={value}" for key, value in report.items())
            await context.put({"type": "log", "text": f"[INFO] Final inference report: {summary}"})
        await context.put({"type": "inference_complete"})
        logger.info("Inference complete")

    context.spawn(complete())


async def _update_model(context: _MessageContext) -> None:
    logger.info("Updating model for next round …")

    async def update() -> None:
        async with context.transfer_lock:
            try:
                await context.loop.run_in_executor(
                    None,
                    context.executor.update_model,
                    context.msg.get("model_url", ""),
                    context.msg.get("params") or {},
                    context.msg.get("dataset_shards"),
                    context.msg.get("delta_url"),
                    context.msg.get("base_hash"),
                    context.msg.get("result_hash"),
                )
                await context.put({"type": "model_updated"})
                logger.info("Model updated — ready for next round")
            except DownloadAborted:
                logger.info("Model update aborted by stop")
            except Exception as exc:
                logger.exception("update_model failed")
                await context.put({"type": "error", "message": str(exc)})

    context.spawn(update())


async def _create_delta(context: _MessageContext, upload_url: str, object_key: str) -> None:
    async with context.transfer_lock:
        delta_file = ""
        try:
            original = await context.loop.run_in_executor(None, context.executor.get_original_state)
            if context.executor.modified_state is None or original is None:
                await context.put({"type": "error", "message": "Model state not available"})
                return
            logger.info("Computing delta")
            descriptor, delta_file = tempfile.mkstemp(prefix="delta-", suffix=".zst", dir=context.executor.work_dir)
            os.close(descriptor)
            stats = await context.loop.run_in_executor(
                None,
                partial(
                    compute_delta_file,
                    original,
                    context.executor.modified_state,
                    delta_file,
                    abort_event=context.executor.abort_event,
                ),
            )
            await context.loop.run_in_executor(None, _upload_file, upload_url, delta_file, context.executor.abort_event)
            await context.put(
                {
                    "type": "stats",
                    "kind": "delta",
                    "training_time_seconds": round(context.executor.training_time, 1),
                    "report": context.executor.report or {},
                    **stats,
                }
            )
            await context.put(
                {
                    "type": "delta_ready",
                    "object_key": object_key,
                    "size_bytes": stats["size_bytes"],
                    "sha256": stats["sha256"],
                }
            )
            logger.info("Delta uploaded directly: %.2f MB", stats["size_compressed_mb"])
        except (DeltaAborted, DownloadAborted):
            logger.info("Delta creation/upload aborted by stop")
        except Exception as exc:
            logger.exception("Delta creation/upload failed")
            await context.put({"type": "error", "message": str(exc)})
        finally:
            if delta_file:
                try:
                    os.unlink(delta_file)
                except FileNotFoundError:
                    pass
            context.executor.original_state = None
            context.executor.modified_state = None


async def _get_model(context: _MessageContext) -> None:
    upload_url = context.msg.get("delta_upload_url")
    object_key = context.msg.get("delta_object_key")
    if not upload_url or not object_key:
        await context.put({"type": "error", "message": "Missing delta upload descriptor"})
        return
    context.spawn(_create_delta(context, upload_url, object_key))


async def _stop_task(context: _MessageContext) -> None:
    logger.info("Stop requested")
    await context.loop.run_in_executor(None, context.executor.stop)


async def _reset_task(context: _MessageContext) -> None:
    logger.info("Reset requested")
    try:
        await context.loop.run_in_executor(None, context.executor.stop)
        async with context.transfer_lock:
            await context.loop.run_in_executor(None, context.executor.reset)
    except RuntimeError as exc:
        logger.critical("Safe task cleanup requires a Machine restart: %s", exc)
        raise SystemExit(75) from exc
    await context.queue.put({"type": "reset_ack"})


async def _heartbeat_ack(_context: _MessageContext) -> None:
    return


_MESSAGE_HANDLERS = {
    "task": _load_task,
    "run": _run_training,
    "inference": _run_inference,
    "update_model": _update_model,
    "get_model": _get_model,
    "stop": _stop_task,
    "reset": _reset_task,
    "heartbeat_ack": _heartbeat_ack,
}


async def _handle(
    ws,
    msg: dict,
    executor: Executor,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    transfer_lock: asyncio.Lock,
    bg_tasks: set[asyncio.Task],
) -> None:
    context = _MessageContext(msg, executor, queue, loop, transfer_lock, bg_tasks)
    handler = _MESSAGE_HANDLERS.get(msg.get("type"))
    if handler is None:
        logger.warning("Unknown message: %s", msg.get("type"))
        return
    await handler(context)


class _AbortableFile:
    """Seekable file facade that stops requests between streamed reads."""

    def __init__(self, payload: BinaryIO, abort_event) -> None:
        self._payload = payload
        self._abort_event = abort_event

    def read(self, size: int = -1):
        if self._abort_event.is_set():
            raise DownloadAborted("delta upload stopped")
        block = self._payload.read(size)
        if self._abort_event.is_set():
            raise DownloadAborted("delta upload stopped")
        return block

    def __getattr__(self, name):
        return getattr(self._payload, name)


def _upload_file(url: str, path: str, abort_event) -> None:
    """Stream a presigned PUT with retries; the file is seekable and reusable."""
    last_error = None
    for attempt in range(3):
        if abort_event.is_set():
            raise DownloadAborted("delta upload stopped")
        try:
            with open(path, "rb") as raw_payload:
                payload = _AbortableFile(raw_payload, abort_event)
                with requests.put(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=(30, 3600),
                ) as response:
                    response.raise_for_status()
            return
        except DownloadAborted:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                import time

                time.sleep(2**attempt)
    raise RuntimeError(f"Delta upload failed: {last_error}")


if __name__ == "__main__":
    asyncio.run(main())
