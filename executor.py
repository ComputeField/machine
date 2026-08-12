# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Isolated GPU training executor with disk-backed models and shard cache.

User code receives model/shard access, run metadata, reporting callbacks,
cooperative cancellation, and PyTorch. Only one decoded shard is retained in
memory; raw shards use a bounded on-disk LRU cache.
"""

import ctypes
import gc
import json
import logging
import os
import signal
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

import requests
import torch
import zstandard as zstd
from model_artifact import materialize_hf_model, prepare_model_artifact
from script_guard import execute_script, validate_script
from sandbox_runtime import popen as sandbox_popen

logger = logging.getLogger(__name__)
_http_local = threading.local()

# Retry transient object-store and proxy failures.
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_BACKOFF = 2.0
# Large chunks reduce syscall overhead on multi-gigabyte transfers.
DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024

# Native CUDA calls can delay delivery of the asynchronous stop exception.
STOP_JOIN_TIMEOUT = 30
MAX_IPC_FRAME = 1024 * 1024
MIN_RESULT_BYTES = 1024 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024 * 1024
RESULT_OVERHEAD_BYTES = 256 * 1024 * 1024


class StopExecution(BaseException):
    """BaseException prevents user `except Exception` blocks swallowing Stop."""


class DownloadAborted(Exception):
    """A deliberate, non-retryable transfer cancellation."""


def _force_stop_thread(thread_id: int) -> None:
    """Raise StopExecution asynchronously in a running thread via CPython internals.
    The exception is delivered at the next Python opcode — even inside a training loop,
    without waiting for should_stop() to be called explicitly."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(StopExecution),
    )
    if res > 1:
        # More than one thread affected — undo
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)


class Executor:
    def __init__(
        self,
        emit_stats: Callable,
        emit_log: Callable,
        host_id: str = "",
        shard_cache_max_mb: int = 4096,
        stage_first_shard: bool = True,
        work_dir: str | None = None,
        sandboxed: bool = False,
    ) -> None:
        self._emit_stats = emit_stats
        self._emit_log = emit_log
        self._host_id = host_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._work_dir = work_dir
        self._sandboxed = sandboxed
        self._process: subprocess.Popen[str] | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="task-", dir=work_dir)
        # Raw shard LRU; <=0 disables caching and prefetching.
        self._shard_cache_max_bytes = shard_cache_max_mb * 1024 * 1024
        self._shard_disk_dir = os.path.join(self._tmpdir, "shards")
        self._disk_cache_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._stage_first_shard = stage_first_shard

        self.original_state: dict[str, torch.Tensor] | None = None
        self.modified_state: dict[str, torch.Tensor] | None = None
        self.training_time: float = 0.0
        self.report: dict | None = None
        self._model_path: str | None = None
        self._bundle_root: str | None = None
        self._image_processor_config: dict | None = None
        self._dataset_shards: list[dict] = []
        self._shard_cache_idx: int | None = None
        self._shard_cache_data: Any = None
        self._code: str | None = None
        self._params: dict = {}
        self._batch_size: int = 32
        self._steps: int = 100  # mini-batch steps per round
        self._round_num: int = 0
        self._total_rounds: int = 1
        self._step_offset: int = 0
        self._mode: str = "training"
        self.execution_error: dict | None = None

    @property
    def work_dir(self) -> str:
        return self._tmpdir

    @property
    def abort_event(self) -> threading.Event:
        """Task-scoped cancellation signal for transfers owned by the session."""
        return self._stop_event

    # Load phase
    def load(
        self,
        model_url: str,
        dataset_shards: list[dict | str],
        code: str,
        batch_size: int = 32,
        params: dict | None = None,
        initial_shard_index: int = 0,
    ) -> None:
        """Download the model and stage only the first bounded microshard.

        `ready` therefore means useful compute can start immediately. Later
        microshards are fetched one ahead while the GPU processes this one.
        """
        # A stale stop flag from the PREVIOUS run must not abort this run's
        # download — cleared before wiring the event in as the abort signal.
        self._stop_event.clear()
        validate_script(code)
        os.makedirs(self._tmpdir, mode=0o700, exist_ok=True)

        # preserve original extension so the temp file matches the format
        ext = _ext_from_url(model_url)
        model_path = os.path.join(self._tmpdir, f"model{ext}")

        logger.info("Downloading model …")
        _download(model_url, model_path, abort_event=self._stop_event)

        self._model_path, self._bundle_root = prepare_model_artifact(model_path, self._tmpdir)
        self._image_processor_config = self._load_image_processor_config()
        self._dataset_shards = [_normalize_shard(item) for item in dataset_shards]
        self._shard_cache_idx = None
        self._shard_cache_data = None
        # A new task = a new run (possibly a different dataset) — stale
        # shards from the previous run must never be served from disk.
        self._clear_shard_disk_cache()
        self._code = code
        self._batch_size = batch_size
        self._params = params or {}

        # Loaded here only to validate the file and log its key count — then
        # freed immediately rather than held in self.original_state for the
        # whole round. It's not needed again until get_model() computes the
        # delta, and keeping a spare full-size copy resident in RAM through
        # the entire (far more memory-hungry) training phase below was
        # observed to push large models (e.g. a 2.5GB ViT) past the host's
        # available memory and get the process OOM-killed with no traceback.
        original_state = _load_model_state(self._model_path, mmap=True)
        logger.info("Load complete — model keys: %d", len(original_state))
        del original_state

        if self._dataset_shards and self._stage_first_shard:
            staged_index = max(0, int(initial_shard_index)) % len(self._dataset_shards)
            self._download_shard(staged_index, cache=self._shard_cache_max_bytes > 0)
            logger.info(
                "Initial dataset microshard %d staged: %.1f MB (%s)",
                staged_index,
                self._dataset_shards[staged_index].get("size_bytes", 0) / 1e6,
                self._dataset_shards[staged_index]["compression"],
            )

        self.original_state = None
        self.modified_state = None
        self.training_time = 0.0

    # Execution phase
    def run(
        self,
        steps: int = 100,
        round_num: int = 0,
        total_rounds: int = 1,
        mode: str = "training",
        step_offset: int = 0,
    ) -> None:
        """Start user code in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Executor already running")
            return
        self._steps = steps
        self._round_num = round_num
        self._total_rounds = total_rounds
        self._step_offset = max(0, int(step_offset))
        self._mode = mode
        self.execution_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._execute, daemon=True, name="executor")
        self._thread.start()

    def update_model(
        self,
        model_url: str,
        params: dict | None = None,
        dataset_shards: list[dict | str] | None = None,
        delta_url: str | None = None,
        base_hash: str | None = None,
        result_hash: str | None = None,
    ) -> None:
        """Prepare the model for the next round: apply middleware's fp16
        sync delta to the model file already on disk when possible (see
        _try_delta_update), otherwise download the full model from
        model_url — which also serves as the automatic fallback whenever
        the delta path can't be used (hash mismatch, missing base, any
        error). A sync must never fail BECAUSE of the delta mechanism.

        params, when provided (e.g. via aggregation code's save_params()), replaces
        the params available to the next round's exec() namespace. Omitted/None
        leaves the existing params unchanged.

        dataset_shards, when provided, replaces the shard descriptors get_shard() uses —
        those are otherwise only presigned once at load() and would expire
        partway through a run longer than the presigned URL TTL. Omitted/None
        leaves the existing URLs (and the currently cached shard, if any)
        untouched — only future get_shard() calls need the refreshed URL."""
        # Free previous state tensors before the update
        self.original_state = None
        self.modified_state = None
        _cuda_empty_cache()

        # Deliberately does NOT clear _stop_event first (unlike load()) — a
        # stop that lands during the between-rounds sync should abort this
        # transfer too; only a fresh task (load) resets the flag.
        applied_delta = False
        if delta_url and base_hash and result_hash:
            applied_delta = self._try_delta_update(delta_url, base_hash, result_hash)

        if not applied_delta:
            # Reuse fixed path to avoid accumulating files on disk each round
            ext = ".safetensors" if ".safetensors" in model_url else ".pt"
            new_path = os.path.join(self._tmpdir, f"current_model{ext}")
            _download(model_url, new_path, abort_event=self._stop_event)
            self._model_path = new_path

        # original_state is intentionally left None here too — see load()'s
        # comment. get_original_state() reloads it from self._model_path on
        # demand, right before the next round's get_model() needs it.
        if params is not None:
            self._params = params
        if dataset_shards is not None:
            self._dataset_shards = [_normalize_shard(item) for item in dataset_shards]
        logger.info(
            "Model updated for round %d (via %s)", self._round_num + 2, "delta" if applied_delta else "full download"
        )

    def _try_delta_update(self, delta_url: str, base_hash: str, result_hash: str) -> bool:
        """Apply middleware's fp16 sync delta to the local model file.
        Returns False (caller falls back to the full download) on any
        divergence or error; only DownloadAborted (an explicit stop)
        propagates — a stop must abort the sync, not degrade it into an
        even bigger transfer.

        Bit-exactness contract with orchestrator/sync_delta.py: both
        sides compute base_fp32 + delta_fp16.float() elementwise, so the
        result_hash comparison holds exactly, not approximately."""
        try:
            if not self._model_path or not os.path.exists(self._model_path):
                logger.warning("Delta sync: no local base model — falling back to full download")
                return False

            base = _load_model_state(self._model_path)
            from delta import state_hash  # local import — avoids a cycle at module load

            if state_hash(base) != base_hash:
                logger.warning(
                    "Delta sync: local base hash mismatch (file overwritten "
                    "or fresh host) — falling back to full download"
                )
                return False

            compressed = os.path.join(self._tmpdir, "sync_delta.zst")
            raw_path = os.path.join(self._tmpdir, "sync_delta.pt")
            _download(delta_url, compressed, abort_event=self._stop_event)
            import zstandard as zstd

            with open(compressed, "rb") as source, open(raw_path, "wb") as target:
                zstd.ZstdDecompressor().copy_stream(source, target)
            os.unlink(compressed)
            delta = torch.load(raw_path, map_location="cpu", mmap=True, weights_only=True)

            # In-place: we own the freshly file-loaded `base` tensors, so
            # add_ avoids a second full-model copy. Keys absent from the
            # delta pass through unchanged (fp32-canonicalized).
            result: dict[str, torch.Tensor] = {}
            for key in list(base.keys()):
                t = base.pop(key).float()
                d = delta.pop(key, None)
                if d is not None:
                    t = t.contiguous().add_(d.float())
                result[key] = t

            if state_hash(result) != result_hash:
                logger.warning("Delta sync: result hash mismatch — falling back to full download")
                return False

            _save_state_file(result, self._model_path)
            logger.info("Delta sync applied to %s (%d keys)", self._model_path, len(result))
            return True
        except DownloadAborted:
            raise
        except Exception:
            logger.warning("Delta sync failed — falling back to full download", exc_info=True)
            return False

    def get_original_state(self) -> dict[str, torch.Tensor] | None:
        """Returns the pre-training state_dict, reloading it from
        self._model_path on demand — it's deliberately not kept resident in
        RAM between load()/update_model() and this call (see their
        comments). mmap-backed: this reload sits on every round boundary's
        critical path right before the delta computation, and mmap turns a
        full multi-GB read+copy into lazy page-in as compute_delta walks
        the keys. Safe here because nothing rewrites _model_path while the
        delta is being computed (update_model comes later)."""
        if self.original_state is None and self._model_path:
            self.original_state = _load_model_state(self._model_path, mmap=True)
        return self.original_state

    def get_shard(self, i: int) -> Any:
        """Load one disk-backed shard; retain only its decoded value in memory."""
        if self._shard_cache_idx == i:
            return self._shard_cache_data
        if not (0 <= i < len(self._dataset_shards)):
            raise IndexError(f"Shard index {i} out of range (num_shards={len(self._dataset_shards)})")

        # Evict the old shard BEFORE downloading the new one — holding both
        # through the download would double the peak on every shard switch.
        # (On a failed fetch the cache stays empty; shards are re-fetchable.)
        self._shard_cache_idx = None
        self._shard_cache_data = None

        disk_path = self.get_shard_path(i)
        try:
            data = torch.load(disk_path, map_location="cpu", mmap=True, weights_only=True)
        except (TypeError, RuntimeError, ValueError):
            data = torch.load(disk_path, map_location="cpu", weights_only=True)

        self._shard_cache_idx = i
        self._shard_cache_data = data
        # Opportunistically warm the disk cache with the shard the default
        # training code will ask for next round ((round+1) → next index).
        return data

    def get_shard_path(self, i: int) -> str:
        """Return a task-local shard file for the isolated runner."""
        if not (0 <= i < len(self._dataset_shards)):
            raise IndexError(f"Shard index {i} out of range (num_shards={len(self._dataset_shards)})")
        disk_path = self._shard_disk_get(i)
        if disk_path is None:
            disk_path = self._download_shard(i, cache=self._shard_cache_max_bytes > 0)
        self._start_prefetch((i + 1) % len(self._dataset_shards))
        return disk_path

    # Disk shard cache + prefetch
    def _shard_disk_path(self, i: int) -> str:
        return os.path.join(self._shard_disk_dir, f"shard_{i}.bin")

    def _clear_shard_disk_cache(self) -> None:
        with self._disk_cache_lock:
            shutil.rmtree(self._shard_disk_dir, ignore_errors=True)

    def _shard_disk_get(self, i: int) -> str | None:
        """Path to the cached raw bytes of shard i, or None. Touches mtime
        so LRU eviction treats reads as recency."""
        if self._shard_cache_max_bytes <= 0:
            return None
        path = self._shard_disk_path(i)
        with self._disk_cache_lock:
            if not os.path.exists(path):
                return None
            try:
                os.utime(path)
            except OSError:
                pass
            return path

    def _download_shard(self, i: int, cache: bool) -> str:
        directory = self._shard_disk_dir if cache else self._tmpdir
        os.makedirs(directory, exist_ok=True)
        path = self._shard_disk_path(i) if cache else os.path.join(directory, f"active_shard_{i}.pt")
        tmp = f"{path}.tmp.{threading.get_ident()}"
        try:
            shard = self._dataset_shards[i]
            downloaded = f"{tmp}.download"
            _download(shard["url"], downloaded, abort_event=self._stop_event)
            if shard["compression"] == "zstd":
                with open(downloaded, "rb") as source, open(tmp, "wb") as target:
                    zstd.ZstdDecompressor().copy_stream(source, target)
                os.unlink(downloaded)
            else:
                os.replace(downloaded, tmp)
            with self._disk_cache_lock:
                if not os.path.exists(path):
                    os.replace(tmp, path)
                if cache:
                    self._shard_disk_evict_lru_locked(keep=path)
            return path
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
            downloaded = f"{tmp}.download"
            if os.path.exists(downloaded):
                os.unlink(downloaded)

    def _shard_disk_evict_lru_locked(self, keep: str) -> None:
        entries = []
        total = 0
        for name in os.listdir(self._shard_disk_dir):
            p = os.path.join(self._shard_disk_dir, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        entries.sort()  # oldest first
        for _, size, p in entries:
            if total <= self._shard_cache_max_bytes:
                break
            if p == keep:
                continue  # never evict the entry just written
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass

    def _start_prefetch(self, i: int) -> None:
        """Kick off a background download of shard i into the disk cache.
        At most one prefetch in flight; skipped entirely for single-shard
        runs or when the disk cache is disabled."""
        if self._shard_cache_max_bytes <= 0 or len(self._dataset_shards) < 2:
            return
        t = self._prefetch_thread
        if t is not None and t.is_alive():
            return
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_shard, args=(i,), daemon=True, name="shard-prefetch"
        )
        self._prefetch_thread.start()

    def _prefetch_shard(self, i: int) -> None:
        """Best-effort: any failure is logged and swallowed — prefetching
        is an optimization, the on-demand path in get_shard still works."""
        try:
            if not (0 <= i < len(self._dataset_shards)):
                return
            if self._shard_disk_get(i) is not None:
                return
            self._download_shard(i, cache=True)
        except DownloadAborted:
            pass
        except Exception:
            logger.debug("Shard prefetch failed for %d", i, exc_info=True)

    def wait(self) -> None:
        """Block until user code finishes (used for testing / sequential flow)."""
        if self._thread:
            self._thread.join()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        execution = self._thread
        if execution and execution.is_alive() and execution.ident is not None:
            _force_stop_thread(execution.ident)
        deadline = time.monotonic() + STOP_JOIN_TIMEOUT
        for thread in (execution, self._prefetch_thread):
            if thread and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def reset(self) -> None:
        self.stop()
        still_running = [
            thread.name for thread in (self._thread, self._prefetch_thread) if thread is not None and thread.is_alive()
        ]
        if still_running:
            raise RuntimeError("Cannot prove task cleanup; restart required for: " + ", ".join(still_running))
        self._stop_event.clear()
        # A broker release is a tenant boundary. Remove models, deltas,
        # offload files and shard cache before acknowledging reset; the broker
        # will not reassign this machine until reset_ack arrives.
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self.original_state = None
        self.modified_state = None
        self._model_path = None
        self._bundle_root = None
        self._image_processor_config = None
        self._dataset_shards = []
        self._shard_cache_idx = None
        self._shard_cache_data = None
        self._shard_disk_dir = os.path.join(self._tmpdir, "shards")
        self._code = None
        self._params = {}
        self.report = None
        self._thread = None
        self._process = None
        self._prefetch_thread = None

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def shutdown(self) -> None:
        """stop() + remove the temp directory (downloaded model files).
        Called when the WS session ends — every reconnect builds a fresh
        Executor with a fresh tmpdir, so without this each reconnect leaked
        a directory holding a potentially multi-GB model file until the
        disk filled. Idempotent."""
        self.stop()
        still_running = [
            thread.name for thread in (self._thread, self._prefetch_thread) if thread is not None and thread.is_alive()
        ]
        if still_running:
            raise RuntimeError("Cannot safely close Machine while workload threads remain: " + ", ".join(still_running))
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @staticmethod
    def _device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _execution_namespace(self, device: str, saved: list) -> dict:
        self.report = None

        def save_model(model: torch.nn.Module) -> None:
            saved[0] = {key: value.cpu() for key, value in model.state_dict().items()}

        def save_report(data: dict) -> None:
            self.report = dict(data)

        def should_stop() -> bool:
            if self._stop_event.is_set():
                raise StopExecution("Stopped by controller")
            return False

        def load_model(path: str) -> dict:
            """Load a safe state_dict or safetensors artifact."""
            return _load_model_file(path)

        return {
            "model_path": self._model_path,
            "get_shard": self.get_shard,
            "num_shards": len(self._dataset_shards),
            "host_id": self._host_id,
            "device": device,
            "batch_size": self._batch_size,
            "params": self._params,
            "steps": self._steps,
            "round_num": self._round_num,  # 0-indexed current round
            "total_rounds": self._total_rounds,
            "step_offset": self._step_offset,
            "emit_stats": self._emit_stats,
            "emit_log": self._emit_log,
            "save_model": save_model,
            "save_report": save_report,
            "should_stop": should_stop,
            "load_model": load_model,
            "materialize_model": self._materialize_model,
            "place_model": self._place_model,
            "prepare_inputs": self._prepare_inputs,
            "torch": torch,
        }

    def _run_user_code(self, namespace: dict) -> None:
        t0 = time.monotonic()
        try:
            execute_script(self._code, namespace)
        except StopExecution:
            logger.info("Execution stopped by controller")
        except torch.cuda.OutOfMemoryError as exc:
            self.execution_error = self._oom_error(exc)
            logger.exception("CUDA out of memory in user code")
            self._emit_log(f"[ERROR] {self.execution_error['message']}")
        except Exception as exc:
            self.execution_error = {"code": "user_code_error", "message": str(exc)}
            logger.exception("User code raised an exception")
            self._emit_log(f"[ERROR] {exc}")
        finally:
            self.training_time = time.monotonic() - t0

    def _resolve_modified_state(self, namespace: dict, saved: list) -> None:
        if self._mode == "inference" or self.execution_error is not None:
            self.modified_state = None
        elif saved[0] is not None:
            self.modified_state = saved[0]
        elif "model" in namespace and hasattr(namespace["model"], "state_dict"):
            self.modified_state = {key: value.cpu() for key, value in namespace["model"].state_dict().items()}
        else:
            try:
                self.modified_state = torch.load(self._model_path, weights_only=True, map_location="cpu")
            except Exception:
                logger.error("Could not resolve modified model state_dict")

    def _execute(self) -> None:
        if self._sandboxed:
            self._execute_isolated()
            return
        device = self._device()
        logger.info("Starting user code  device=%s", device)
        saved: list = [None]
        namespace = self._execution_namespace(device, saved)
        self._run_user_code(namespace)
        self._resolve_modified_state(namespace, saved)
        namespace.clear()
        saved[0] = None
        gc.collect()
        _cuda_empty_cache()

        logger.info(
            "Execution finished  time=%.1fs  state_dict=%s",
            self.training_time,
            "ok" if self.modified_state else "MISSING",
        )

    def _execute_isolated(self) -> None:
        """Run user code in a fresh OS-sandboxed interpreter.

        Only framed messages and task-local paths cross stdio. The child has
        neither the daemon environment nor a network namespace containing an
        external interface.
        """
        spec_path = os.path.join(self._tmpdir, "runner-spec.json")
        try:
            model_size = os.path.getsize(self._model_path or "")
        except OSError:
            model_size = 0
        result_max_bytes = min(
            MAX_RESULT_BYTES,
            max(MIN_RESULT_BYTES, model_size * 2 + RESULT_OVERHEAD_BYTES),
        )
        spec = {
            "model_path": self._model_path,
            "bundle_root": self._bundle_root,
            "image_processor_config": self._image_processor_config,
            "num_shards": len(self._dataset_shards),
            "host_id": self._host_id,
            "params": self._params,
            "batch_size": self._batch_size,
            "steps": self._steps,
            "round_num": self._round_num,
            "total_rounds": self._total_rounds,
            "step_offset": self._step_offset,
            "mode": self._mode,
            "code": self._code,
            "result_max_bytes": result_max_bytes,
        }
        with open(spec_path, "w", encoding="utf-8") as file:
            json.dump(spec, file)
        os.chmod(spec_path, 0o600)
        runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workload_runner.py")
        process = sandbox_popen([sys.executable, runner_path, spec_path], self._tmpdir)
        self._process = process
        completed: dict | None = None
        stderr_lines: list[str] = []

        def drain_stderr() -> None:
            if process.stderr is None:
                return
            for line in process.stderr:
                line = line.rstrip()
                if line:
                    if len(stderr_lines) >= 100:
                        del stderr_lines[0]
                    stderr_lines.append(line)
                    self._emit_log(f"[ERROR] {line[:16_384]}")

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True, name="runner-stderr")
        stderr_thread.start()
        try:
            if process.stdout is None or process.stdin is None:
                raise RuntimeError("workload runner pipes were not created")
            while True:
                line = process.stdout.readline(MAX_IPC_FRAME + 1)
                if not line:
                    break
                if len(line) > MAX_IPC_FRAME:
                    raise RuntimeError("workload IPC frame exceeds 1 MiB")
                try:
                    message = json.loads(line)
                except ValueError:
                    self._emit_log(f"[INFO] {line.rstrip()}")
                    continue
                kind = message.get("type")
                if kind == "log":
                    self._emit_log(str(message.get("text") or "")[:16_384])
                elif kind == "stats" and isinstance(message.get("data"), dict):
                    self._emit_stats(message["data"])
                elif kind == "shard":
                    try:
                        path = self.get_shard_path(int(message["index"]))
                        response = {"ok": True, "path": path}
                    except Exception as exc:
                        response = {"ok": False, "error": str(exc)}
                    process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                elif kind == "complete":
                    completed = message
            returncode = process.wait()
            # The protocol process is complete; no descendant is allowed to
            # race result validation or survive into the next tenant lease.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stderr_thread.join(timeout=2)
            if completed is None:
                if self._stop_event.is_set():
                    logger.info("Isolated execution stopped by controller")
                    return
                detail = stderr_lines[-1] if stderr_lines else f"exit status {returncode}"
                self.execution_error = {"code": "sandbox_runner_error", "message": detail}
                return
            self.training_time = float(completed.get("training_time") or 0)
            self.report = completed.get("report") if isinstance(completed.get("report"), dict) else None
            self.execution_error = completed.get("error") if isinstance(completed.get("error"), dict) else None
            expected_result = os.path.join(self._tmpdir, "runner-result.pt")
            result_path = str(completed.get("result_path") or "")
            if result_path and self.execution_error is None and self._mode != "inference":
                if os.path.realpath(result_path) != os.path.realpath(expected_result):
                    self.execution_error = {
                        "code": "sandbox_protocol_error",
                        "message": "workload returned an invalid result path",
                    }
                    self.modified_state = None
                else:
                    metadata = os.lstat(result_path)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise RuntimeError("workload result must be one regular, unlinked file")
                    if metadata.st_size > result_max_bytes:
                        raise RuntimeError("workload result exceeds its model-derived size limit")
                    self.modified_state = torch.load(result_path, weights_only=True, map_location="cpu")
            else:
                self.modified_state = None
        except Exception as exc:
            logger.exception("Isolated workload runner failed")
            self.execution_error = {"code": "sandbox_runner_error", "message": str(exc)}
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        finally:
            self._process = None
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def _place_model(self, model: torch.nn.Module, training: bool = True):
        """Place standard module trees without blindly exhausting VRAM."""
        if not torch.cuda.is_available():
            target = "mps" if torch.backends.mps.is_available() else "cpu"
            return model.to(target)

        parameter_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in list(model.parameters()) + list(model.buffers())
        )
        free_bytes, _ = torch.cuda.mem_get_info()
        default_multiplier = 1.15
        if training:
            default_multiplier = 3.5 if self._params.get("train_mode", "head") == "all" else 1.25
        multiplier = float(
            self._params.get(
                "training_memory_multiplier",
                default_multiplier,
            )
        )
        if parameter_bytes * multiplier <= free_bytes * 0.85:
            try:
                return model.to("cuda")
            except torch.cuda.OutOfMemoryError:
                _cuda_empty_cache()

        try:
            from accelerate import dispatch_model, infer_auto_device_map

            offload_dir = os.path.join(self._tmpdir, "model-offload")
            # One round owns this directory. Reusing stale offload files can
            # retain gigabytes after a changed model layout.
            shutil.rmtree(offload_dir, ignore_errors=True)
            os.makedirs(offload_dir, exist_ok=True)
            device_map = infer_auto_device_map(
                model,
                max_memory={
                    0: f"{max(256, int(free_bytes * 0.82 / 1024 / 1024))}MiB",
                    "cpu": f"{max(1024, _available_ram_mb() * 3 // 4)}MiB",
                },
                no_split_module_classes=list(
                    self._params.get("no_split_module_classes", []),
                ),
            )
            placed = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                offload_buffers=True,
            )
            self._emit_log("[INFO] Automatic GPU/CPU/disk model placement enabled; single-GPU VRAM was insufficient")
            return placed
        except Exception as exc:
            raise RuntimeError(
                "Model does not fit GPU and automatic module sharding failed. "
                "Use a standard torch/torchvision/Hugging Face module tree or "
                f"configure no_split_module_classes. Cause: {exc}"
            ) from exc

    def _materialize_model(self, loaded) -> torch.nn.Module:
        """Build a standard architecture locally, then apply an uploaded state_dict."""
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else None
        if not isinstance(state, dict):
            raise TypeError("Model artifact must contain a tensor state_dict")
        if self._bundle_root:
            model, missing, unexpected = materialize_hf_model(self._bundle_root, state)
            self._emit_log(
                f"[INFO] Loaded Hugging Face bundle automatically; missing={len(missing)}, unexpected={len(unexpected)}"
            )
            return model
        model_name = str(self._params.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("A state_dict requires params.model_name")
        backend = str(self._params.get("model_backend", "timm")).lower()
        classes = int(self._params.get("num_classes", 1000))
        if backend == "torchvision":
            from torchvision import models

            model = models.get_model(model_name, weights=None, num_classes=classes)
        elif backend == "huggingface":
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageClassification,
                AutoModelForSequenceClassification,
            )

            config_data = dict(self._params.get("hf_config") or {})
            model_type = config_data.pop("model_type", None)
            if not model_type:
                raise ValueError("Hugging Face models require params.hf_config.model_type")
            config = AutoConfig.for_model(model_type, **config_data)
            factories = {
                "causal-lm": AutoModelForCausalLM,
                "image-classification": AutoModelForImageClassification,
                "sequence-classification": AutoModelForSequenceClassification,
            }
            task = str(self._params.get("hf_task", "image-classification"))
            if task not in factories:
                raise ValueError(f"Unsupported params.hf_task: {task}")
            model = factories[task].from_config(config)
        elif backend == "timm":
            import timm

            model = timm.create_model(model_name, pretrained=False, num_classes=classes)
        else:
            raise ValueError(f"Unsupported params.model_backend: {backend}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._emit_log(f"[INFO] Loaded {model_name}; missing={len(missing)}, unexpected={len(unexpected)}")
        return model

    def _load_image_processor_config(self) -> dict:
        if not self._bundle_root:
            return {}
        path = os.path.join(self._bundle_root, "preprocessor_config.json")
        try:
            with open(path, encoding="utf-8") as source:
                value = json.load(source)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _image_target(self, model: torch.nn.Module, config: dict) -> tuple[int, int] | None:
        size = (
            self._params.get("image_size")
            or config.get("crop_size")
            or config.get("size")
            or getattr(getattr(model, "config", None), "image_size", None)
        )
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                return int(size["height"]), int(size["width"])
            if "shortest_edge" in size:
                edge = int(size["shortest_edge"])
                return edge, edge
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return int(size[-2]), int(size[-1])
        return (size, size) if isinstance(size, int) else None

    def _resize_inputs(self, value: torch.Tensor, target: tuple[int, int] | None, config: dict) -> torch.Tensor:
        requested = self._params.get("image_size")
        processor_size = config.get("crop_size") or config.get("size")
        should_resize = requested is not None or bool(config.get("do_resize", processor_size is not None))
        if not (should_resize and target and value.ndim == 4 and tuple(value.shape[-2:]) != target):
            return value
        import torch.nn.functional as functional

        return functional.interpolate(value, size=target, mode="bilinear", align_corners=False)

    def _rescale_inputs(self, value: torch.Tensor, original_dtype: torch.dtype, config: dict) -> torch.Tensor:
        factor = self._params.get("rescale_factor", config.get("rescale_factor"))
        if not bool(config.get("do_rescale", factor is not None)) or factor is None:
            return value
        already_scaled = original_dtype.is_floating_point and (not value.numel() or float(value.detach().max()) <= 1.5)
        return value if already_scaled else value * float(factor)

    def _normalize_inputs(self, value: torch.Tensor, config: dict) -> torch.Tensor:
        mean = self._params.get("image_mean", config.get("image_mean"))
        std = self._params.get("image_std", config.get("image_std"))
        requested = self._params.get("image_mean") is not None or self._params.get("image_std") is not None
        enabled = requested or bool(config.get("do_normalize", mean is not None and std is not None))
        if not enabled or mean is None or std is None or value.ndim != 4:
            return value
        mean_tensor = torch.as_tensor(mean, dtype=value.dtype).flatten()
        std_tensor = torch.as_tensor(std, dtype=value.dtype).flatten()
        if value.shape[1] == 1 and mean_tensor.numel() == 3:
            value = value.repeat(1, 3, 1, 1)
        if mean_tensor.numel() != value.shape[1] or std_tensor.numel() != value.shape[1]:
            raise ValueError("Image processor channel count does not match the dataset")
        if bool((std_tensor <= 0).any()):
            raise ValueError("Image processor standard deviation must be positive")
        return (value - mean_tensor.view(1, -1, 1, 1)) / std_tensor.view(1, -1, 1, 1)

    def _prepare_inputs(self, value: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Apply bundled image resize/normalization without remote model code."""
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        original_dtype = value.dtype
        value = value.float()
        config = dict(self._image_processor_config or {})
        squeezed = value.ndim == 3
        if squeezed:
            value = value.unsqueeze(0)
        value = self._resize_inputs(value, self._image_target(model, config), config)
        value = self._rescale_inputs(value, original_dtype, config)
        value = self._normalize_inputs(value, config)
        return value.squeeze(0) if squeezed else value

    def _oom_error(self, exc: BaseException) -> dict:
        details = {}
        if torch.cuda.is_available():
            details = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
            }
            _cuda_empty_cache()
        return {
            "code": "gpu_oom",
            "message": (
                "GPU ran out of memory. Reduce batch_size or call "
                "place_model(model), which enables automatic module sharding."
            ),
            "details": details,
            "cause": str(exc),
        }


# Helpers
def _normalize_shard(item: dict | str) -> dict:
    if isinstance(item, str):
        return {"url": item, "compression": "none", "size_bytes": 0}
    compression = item.get("compression", "none")
    if compression not in {"none", "zstd"}:
        raise ValueError(f"Unsupported shard compression: {compression}")
    url = item.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Dataset shard descriptor requires a URL")
    return {**item, "url": url, "compression": compression}


def _available_ram_mb() -> int:
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024)
    except (ValueError, OSError, AttributeError):
        return 4096


def _cuda_empty_cache() -> None:
    """Release device memory cache (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _http_session() -> requests.Session:
    session = getattr(_http_local, "session", None)
    if session is None:
        session = requests.Session()
        _http_local.session = session
    return session


def _prepare_resumed_sink(response, sink, offset: int) -> None:
    if not offset:
        return
    content_range = response.headers.get("Content-Range", "")
    if response.status_code == 206 and content_range.startswith(f"bytes {offset}-"):
        sink.seek(offset)
        return
    sink.seek(0)
    sink.truncate()


def _write_download(response, sink, url: str, abort_event: threading.Event | None) -> None:
    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
        if abort_event is not None and abort_event.is_set():
            raise DownloadAborted(url)
        sink.write(chunk)


def _fetch_once(url: str, sink, timeout: int, abort_event: threading.Event | None) -> None:
    if abort_event is not None and abort_event.is_set():
        raise DownloadAborted(url)
    offset = sink.seek(0, os.SEEK_END)
    options = {"headers": {"Range": f"bytes={offset}-"}} if offset else {}
    response = _http_session().get(url, stream=True, timeout=(30, timeout), **options)
    try:
        response.raise_for_status()
        _prepare_resumed_sink(response, sink, offset)
        _write_download(response, sink, url, abort_event)
    finally:
        response.close()


def _fetch_to(url: str, sink, timeout: int = 300, abort_event: threading.Event | None = None) -> None:
    """Stream with retry, resumable ranges, and cooperative cancellation."""
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            _fetch_once(url, sink, timeout, abort_event)
            return
        except DownloadAborted:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < DOWNLOAD_RETRIES:
                logger.warning("Download failed (attempt %d/%d): %s — retrying", attempt, DOWNLOAD_RETRIES, exc)
                time.sleep(DOWNLOAD_RETRY_BACKOFF * attempt)
    if last_exc is None:
        raise RuntimeError("download retry loop ended without a result")
    raise last_exc


def _download(url: str, dest: str, timeout: int = 300, abort_event: threading.Event | None = None) -> None:
    with open(dest, "wb") as fh:
        _fetch_to(url, fh, timeout=timeout, abort_event=abort_event)


def _ext_from_url(url: str) -> str:
    """Extract file extension from a URL (ignoring query string)."""
    from urllib.parse import urlparse

    path = urlparse(url).path
    for suffix in (".safetensors", ".zip", ".pth", ".bin", ".pt"):
        if path.lower().endswith(suffix):
            return suffix
    return ".pt"


def _save_state_file(state: dict, path: str) -> None:
    """Save a state_dict to path in the format its extension implies —
    counterpart of _load_model_state, used by the delta sync to rewrite
    the local model file after applying middleware's delta. Writes via
    tmp+rename so a crash/error mid-save never leaves a truncated model
    file behind (and mmap-backed readers of the old inode stay valid)."""
    tmp_path = path + ".tmp"
    try:
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file

            save_file({k: v.contiguous() for k, v in state.items()}, tmp_path)
        else:
            torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _as_state_dict(loaded) -> dict:
    if isinstance(loaded, dict):
        return {k: v.cpu() for k, v in loaded.items()}
    # Full nn.Module — extract state_dict
    raise ValueError(
        "Models must be safetensors or a tensor state_dict; serialized nn.Module "
        "objects are rejected because pickle can execute code"
    )


def _load_model_state(path: str, mmap: bool = False) -> dict:
    """Load model state_dict from .pt/.pth or .safetensors file.

    mmap=True maps the file instead of reading it whole (lazy page-in) —
    use ONLY where nothing will rewrite `path` while the returned tensors
    are alive (see get_original_state). torch.load's mmap is
    copy-on-write, but rewriting the backing file mid-save while COW pages
    still reference it is exactly the hazard _try_delta_update avoids by
    loading its base without mmap. (.safetensors is mmap-backed either
    way — that's the format's native behavior.)"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        logger.info("Loading safetensors model: %s", path)
        return load_file(path)  # always returns {key: tensor}

    if mmap:
        try:
            return _as_state_dict(
                torch.load(
                    path,
                    weights_only=True,
                    map_location="cpu",
                    mmap=True,
                )
            )
        except (RuntimeError, ValueError, TypeError):
            logger.info("mmap load failed for %s — using a regular safe load", path)
    return _as_state_dict(torch.load(path, weights_only=True, map_location="cpu"))


def _load_model_file(path: str):
    """Load a non-executable safetensors file or tensor state_dict."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)
