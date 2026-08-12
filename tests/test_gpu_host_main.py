# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/main.py — the WS session loop.

Machine/main.py and orchestrator/main.py share a module name (same for
config.py), so this file loads Machine's copies explicitly by path under
unique module names instead of a plain `import main`, and restores
sys.modules afterwards so orchestrator tests importing their own config
are unaffected.
"""

import asyncio
import importlib.util
import json
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

_GPU_HOST_DIR = pathlib.Path(__file__).parent.parent


def _load_by_path(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _GPU_HOST_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load Machine's config under the canonical "config" name just long enough
# for main.py's own `from config import settings` to resolve to it, then
# restore whatever was there before.
_saved_config = sys.modules.get("config")
sys.modules["config"] = _load_by_path("gpu_host_config", "config.py")
try:
    gh_main = _load_by_path("gpu_host_main", "main.py")
finally:
    if _saved_config is not None:
        sys.modules["config"] = _saved_config
    else:
        sys.modules.pop("config", None)


# Test harness
class FakeWS:
    """Minimal websocket stand-in: recv() yields a scripted sequence of
    incoming frames, then raises ConnectionClosedOK (normal disconnect)."""

    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list = []

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        if not self._incoming:
            raise ConnectionClosedOK(None, None)
        item = self._incoming.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _registered() -> str:
    return json.dumps({"type": "registered"})


@pytest.fixture(autouse=True)
def _no_speedtest(monkeypatch):
    # The real run_speed_test does its own ws.send()/ws.recv() directly on
    # the session's WS object — against FakeWS that would silently consume
    # whatever scripted message a given test queued up next, derailing
    # tests that have nothing to do with the speed test. TestSpeedTestWiring
    # overrides this per-test with its own explicit monkeypatch.
    return monkeypatch.setattr(
        gh_main,
        "run_speed_test",
        AsyncMock(
            return_value={
                "upload_mbps": 0.0,
                "download_mbps": 0.0,
                "payload_bytes": 0,
            }
        ),
    )


# S1: connect kwargs
class TestConnectKwargs:
    def test_connect_sets_a_generous_max_size(self, monkeypatch):
        """websockets' default incoming-message limit is 1 MiB — a task
        message carrying presigned URLs for a few thousand shards exceeds
        that, closing the connection with 'message too big' and wedging the
        host in a reconnect loop where the rejoin re-send hits the same
        limit forever. orchestrator's own client sets 1 GB explicitly;
        the Machine side needs an explicit limit too."""
        connect = MagicMock()
        monkeypatch.setattr(gh_main.websockets, "connect", connect)

        gh_main._connect("ws://fake:9100/ws/machine")

        kwargs = connect.call_args.kwargs
        assert kwargs.get("max_size", 1024 * 1024) >= 16 * 1024 * 1024
        assert kwargs.get("ping_interval") == gh_main.WS_PING_INTERVAL
        assert kwargs.get("ping_timeout") == gh_main.WS_PING_TIMEOUT


# S6: session resilience
class TestSessionResilience:
    @pytest.mark.asyncio
    async def test_malformed_json_does_not_kill_the_session(self):
        """One garbage frame must be ignored, not escalate into a full
        disconnect/reconnect cycle."""
        ws = FakeWS([_registered(), "{{{not json", json.dumps({"type": "heartbeat_ack"})])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})
        # register + nothing else queued synchronously — the point is we
        # reached the normal ConnectionClosed exit, not a JSONDecodeError.

    @pytest.mark.asyncio
    async def test_handler_error_does_not_kill_the_session(self):
        """A message with unexpected field types (here: non-numeric steps)
        must be logged and skipped, not tear the session down."""
        ws = FakeWS(
            [
                _registered(),
                json.dumps({"type": "run", "steps": "not-a-number"}),
                json.dumps({"type": "heartbeat_ack"}),
            ]
        )
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})


# S2/S5: executor shutdown on disconnect
class TestSessionCleanup:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_the_executor_tmpdir(self, monkeypatch):
        """Every session builds a fresh Executor (fresh tmpdir with the
        downloaded model inside) — session end must remove it, or every
        reconnect leaks a multi-GB directory until the disk fills."""
        created = []
        real_executor = gh_main.Executor

        class TrackingExecutor(real_executor):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                created.append(self)

        monkeypatch.setattr(gh_main, "Executor", TrackingExecutor)

        ws = FakeWS([_registered()])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})

        assert len(created) == 1
        assert not os.path.exists(created[0]._tmpdir)


# S4: stop processed while a transfer is in flight
class TestNonBlockingTransfers:
    @pytest.mark.asyncio
    async def test_stop_aborts_an_in_flight_load(self, monkeypatch):
        """Previously the receive loop awaited the whole load inline — a
        stop arriving mid-download sat unread until the transfer finished.
        Now the load runs as a background task and stop gets through
        immediately, aborting the download via the executor's stop event."""
        # executor is already importable under its own name via conftest path
        import executor as executor_module

        outcomes = []
        real_executor = gh_main.Executor

        class BlockingLoadExecutor(real_executor):
            def load(self, *a, **kw):
                # Simulates an abortable download: returns only once stop()
                # sets the stop event; times out (test failure mode) if the
                # stop never gets processed while we're "downloading".
                aborted = self._stop_event.wait(timeout=5.0)
                if aborted:
                    outcomes.append("aborted")
                    raise executor_module.DownloadAborted("test")
                outcomes.append("timed_out")

        monkeypatch.setattr(gh_main, "Executor", BlockingLoadExecutor)

        ws = FakeWS(
            [
                _registered(),
                json.dumps({"type": "task", "model_url": "http://fake/m.pt", "dataset_shards": [], "code": ""}),
                json.dumps({"type": "stop"}),
            ]
        )

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})
        # allow the background load task to observe the abort
        for _ in range(50):
            if outcomes:
                break
            await asyncio.sleep(0.05)
        elapsed = loop.time() - t0

        assert outcomes == ["aborted"]
        assert elapsed < 3.0, f"session took {elapsed:.1f}s — stop was queued behind the load"

    @pytest.mark.asyncio
    async def test_completion_keeps_the_lease_that_started_the_run(self, monkeypatch):
        """A new task lease must not relabel an older queued completion."""
        import threading

        wait_started = threading.Event()
        wait_release = threading.Event()
        real_executor = gh_main.Executor

        class ControlledExecutor(real_executor):
            def load(self, *_args, **_kwargs):
                return None

            def run(self, **_kwargs):
                return None

            def wait(self):
                wait_started.set()
                wait_release.wait(timeout=2)

        monkeypatch.setattr(gh_main, "Executor", ControlledExecutor)

        class LeaseWS(FakeWS):
            async def recv(self):
                if self._incoming:
                    return await super().recv()
                while not wait_started.is_set():
                    await asyncio.sleep(0.01)
                wait_release.set()
                await asyncio.sleep(0.1)
                raise ConnectionClosedOK(None, None)

        ws = LeaseWS(
            [
                _registered(),
                json.dumps(
                    {
                        "type": "task",
                        "lease_id": "lease-a",
                        "model_url": "http://fake/a.pt",
                        "dataset_shards": [],
                        "code": "",
                    }
                ),
                json.dumps({"type": "run", "lease_id": "lease-a", "steps": 1}),
                json.dumps(
                    {
                        "type": "task",
                        "lease_id": "lease-b",
                        "model_url": "http://fake/b.pt",
                        "dataset_shards": [],
                        "code": "",
                    }
                ),
            ]
        )

        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})

        messages = [json.loads(item) for item in ws.sent if isinstance(item, str)]
        completion = next(item for item in messages if item.get("type") == "run_complete")
        assert completion["lease_id"] == "lease-a"


# L3: delta sync fields pass through to the executor
class TestUpdateModelDeltaPassthrough:
    @pytest.mark.asyncio
    async def test_delta_fields_reach_executor_update_model(self, monkeypatch):
        captured = {}
        real_executor = gh_main.Executor

        class CapturingExecutor(real_executor):
            def update_model(
                self, model_url, params=None, dataset_shards=None, delta_url=None, base_hash=None, result_hash=None
            ):
                captured.update(model_url=model_url, delta_url=delta_url, base_hash=base_hash, result_hash=result_hash)

        monkeypatch.setattr(gh_main, "Executor", CapturingExecutor)

        ws = FakeWS(
            [
                _registered(),
                json.dumps(
                    {
                        "type": "update_model",
                        "model_url": "http://fake/full.pt",
                        "delta_url": "http://fake/delta.zst",
                        "base_hash": "b" * 64,
                        "result_hash": "r" * 64,
                    }
                ),
            ]
        )
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})
        for _ in range(50):  # update runs as a background task
            if captured:
                break
            await asyncio.sleep(0.02)

        assert captured == {
            "model_url": "http://fake/full.pt",
            "delta_url": "http://fake/delta.zst",
            "base_hash": "b" * 64,
            "result_hash": "r" * 64,
        }


# R2: shard cache size is operator-configurable
class TestShardCacheConfig:
    @pytest.mark.asyncio
    async def test_executor_receives_the_configured_cap(self, monkeypatch):
        """SHARD_CACHE_MAX_MB in the host's .env controls the disk cache
        cap; the session must construct its Executor with that value."""
        assert hasattr(gh_main.settings, "shard_cache_max_mb")
        monkeypatch.setattr(gh_main.settings, "shard_cache_max_mb", 123, raising=False)

        captured = {}
        real_executor = gh_main.Executor

        class CapturingExecutor(real_executor):
            def __init__(self, *a, **kw):
                captured.update(kw)
                super().__init__(*a, **kw)

        monkeypatch.setattr(gh_main, "Executor", CapturingExecutor)

        ws = FakeWS([_registered()])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})

        assert captured.get("shard_cache_max_mb") == 123


# Speed test wiring
class TestSpeedTestWiring:
    @pytest.mark.asyncio
    async def test_runs_speed_test_after_registration_and_sends_result(self, monkeypatch):
        monkeypatch.setattr(gh_main.settings, "speedtest_payload_mb", 8, raising=False)
        run_speed_test = AsyncMock(
            return_value={
                "upload_mbps": 12.5,
                "download_mbps": 34.0,
                "payload_bytes": 8 * 1024 * 1024,
            }
        )
        monkeypatch.setattr(gh_main, "run_speed_test", run_speed_test)

        ws = FakeWS([_registered()])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})

        run_speed_test.assert_awaited_once()
        assert run_speed_test.await_args.args[0] is ws
        assert run_speed_test.await_args.args[1] == 8

        result_msgs = [json.loads(s) for s in ws.sent if isinstance(s, str)]
        speedtest_results = [m for m in result_msgs if m.get("type") == "speedtest_result"]
        assert len(speedtest_results) == 1
        assert speedtest_results[0]["upload_mbps"] == 12.5
        assert speedtest_results[0]["download_mbps"] == 34.0

    @pytest.mark.asyncio
    async def test_speed_test_failure_does_not_crash_the_session(self, monkeypatch):
        """A network hiccup during the speed test must not prevent the host
        from proceeding into normal operation."""
        monkeypatch.setattr(gh_main.settings, "speedtest_payload_mb", 8, raising=False)
        monkeypatch.setattr(gh_main, "run_speed_test", AsyncMock(side_effect=RuntimeError("boom")))

        ws = FakeWS([_registered(), json.dumps({"type": "heartbeat_ack"})])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})
        # reaching ConnectionClosed (not RuntimeError) proves the session
        # survived the speed-test failure and kept processing messages

    @pytest.mark.asyncio
    async def test_zero_payload_mb_skips_the_speed_test_entirely(self, monkeypatch):
        monkeypatch.setattr(gh_main.settings, "speedtest_payload_mb", 0, raising=False)
        run_speed_test = AsyncMock()
        monkeypatch.setattr(gh_main, "run_speed_test", run_speed_test)

        ws = FakeWS([_registered()])
        with pytest.raises(ConnectionClosed):
            await gh_main._session(ws, {"compute_mode": "cpu"})

        run_speed_test.assert_not_awaited()
