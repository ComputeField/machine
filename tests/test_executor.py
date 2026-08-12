# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/executor.py"""

import io
import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import zstandard
from delta import state_hash
from executor import Executor
from torch import nn


# Helpers
def _make_model_file(path: str) -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    torch.save(model.state_dict(), path)
    return model


def _make_executor(host_id=""):
    stats, logs = [], []
    ex = Executor(
        emit_stats=lambda d: stats.append(d),
        emit_log=lambda t: logs.append(t),
        host_id=host_id,
        stage_first_shard=False,
    )
    return ex, stats, logs


def _make_sync_payload(old, new):
    """Build the documented server-to-Machine delta wire format locally."""
    delta = {}
    canonical = {}
    for key in old:
        base = old[key].detach().cpu().float()
        diff = (new[key].detach().cpu().float() - base).half()
        if diff.count_nonzero().item():
            delta[key] = diff
            canonical[key] = base + diff.float()
        else:
            canonical[key] = base
    raw = io.BytesIO()
    torch.save(delta, raw)
    blob = zstandard.ZstdCompressor(level=3).compress(raw.getvalue())
    return blob, canonical, state_hash(old), state_hash(canonical)


# Tests
class TestExecutorLoad:
    def test_default_load_stages_first_bounded_microshard_before_ready(self, tmp_path):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex = Executor(emit_stats=lambda _: None, emit_log=lambda _: None)

        with patch("executor._download") as download:

            def side_effect(url, destination, **_kwargs):
                if "model" in url:
                    import shutil

                    shutil.copy(model_path, destination)
                else:
                    torch.save([(torch.zeros(1), torch.tensor(0))], destination)

            download.side_effect = side_effect
            ex.load(
                "http://fake/model.pt",
                [
                    {
                        "url": "http://fake/shard.pt",
                        "compression": "none",
                        "size_bytes": 1024,
                    }
                ],
                "",
            )

        assert download.call_count == 2
        assert os.path.exists(ex._shard_disk_path(0))

    def test_bundled_image_processor_resizes_rescales_and_normalizes(self, tmp_path):
        ex, _, _ = _make_executor()
        ex._bundle_root = str(tmp_path)
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "do_rescale": True,
                    "rescale_factor": 1 / 255,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "size": {"height": 8, "width": 8},
                }
            )
        )
        ex._image_processor_config = ex._load_image_processor_config()
        model = MagicMock()
        model.config.image_size = 8

        prepared = ex._prepare_inputs(torch.full((2, 1, 4, 4), 255, dtype=torch.uint8), model)

        assert prepared.shape == (2, 3, 8, 8)
        assert torch.allclose(prepared, torch.ones_like(prepared))

    def test_float_images_are_not_rescaled_twice(self):
        ex, _, _ = _make_executor()
        ex._image_processor_config = {
            "do_rescale": True,
            "rescale_factor": 1 / 255,
        }
        model = MagicMock()
        model.config.image_size = 4
        images = torch.full((2, 3, 4, 4), 0.5)

        assert torch.equal(ex._prepare_inputs(images, model), images)

    def test_disabled_processor_steps_are_respected(self):
        ex, _, _ = _make_executor()
        ex._image_processor_config = {
            "do_resize": False,
            "size": {"height": 8, "width": 8},
            "do_normalize": False,
            "image_mean": [0.5],
            "image_std": [0.5],
        }
        model = MagicMock()
        model.config.image_size = 8
        images = torch.full((2, 1, 4, 4), 0.5)

        prepared = ex._prepare_inputs(images, model)

        assert prepared.shape == images.shape
        assert torch.equal(prepared, images)

    def test_inference_emits_stats_without_creating_training_delta(self, tmp_path):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, stats, _ = _make_executor()
        with patch("executor._download") as download:
            download.side_effect = lambda _url, destination, **_kw: __import__(
                "shutil",
            ).copy(model_path, destination)
            ex.load("http://fake/model.pt", [], "emit_stats({'kind': 'inference', 'step': 1, 'loss': 0.25})")
        ex.run(mode="inference")
        ex.wait()

        assert ex.execution_error is None
        assert ex.modified_state is None
        assert stats == [{"kind": "inference", "step": 1, "loss": 0.25}]

    def test_cuda_oom_becomes_a_structured_failure(self, tmp_path):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, _, logs = _make_executor()
        with patch("executor._download") as download:
            download.side_effect = lambda _url, destination, **_kw: __import__(
                "shutil",
            ).copy(model_path, destination)
            ex.load("http://fake/model.pt", [], "raise torch.cuda.OutOfMemoryError('synthetic OOM')")
        ex.run()
        ex.wait()

        assert ex.execution_error["code"] == "gpu_oom"
        assert any("automatic module sharding" in line for line in logs)

    def test_load_does_not_keep_original_state_resident(self, tmp_path):
        """original_state must NOT be populated right after load() — it's
        validated once (for the key-count log) and then freed, so a large
        model isn't held in RAM for the entire (much more memory-hungry)
        training phase. See executor.py's load()/get_original_state()."""
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        ex, _, _ = _make_executor()

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect

            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "emit_log('loaded')")

        assert ex.original_state is None
        assert ex.modified_state is None

    def test_load_does_not_download_dataset_shards_eagerly(self, tmp_path):
        """load() must not fetch any shard content — shards are only ever
        fetched lazily via get_shard() from user code (see get_shard())."""
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        ex, _, _ = _make_executor()

        with patch("executor._download") as mock_dl, patch("executor.requests.Session.get") as mock_get:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect

            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt", "http://fake/shard_1.pt"], "emit_log('loaded')")

        mock_get.assert_not_called()

    def test_get_original_state_lazily_loads_from_model_path(self, tmp_path):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        ex, _, _ = _make_executor()

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect

            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "emit_log('loaded')")

        state = ex.get_original_state()
        assert state is not None
        assert isinstance(state, dict)
        # cached — second call returns the same object, doesn't reload from disk
        assert ex.get_original_state() is state


class TestGetShard:
    @pytest.fixture(autouse=True)
    def _no_prefetch(self, monkeypatch):
        # Background prefetch (see TestShardPrefetch) would consume the
        # mocked responses in these tests non-deterministically.
        monkeypatch.setattr(Executor, "_start_prefetch", lambda self, i: None, raising=False)

    def _loaded_executor(self, tmp_path, n_shards=2):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, self._stats, self._logs = _make_executor()
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", [f"http://fake/shard_{i}.pt" for i in range(n_shards)], "")
        return ex

    def _fake_response(self, payload) -> MagicMock:
        buf = io.BytesIO()
        torch.save(payload, buf)
        content = buf.getvalue()
        resp = MagicMock()
        resp.content = content
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter([content])
        return resp

    def test_downloads_and_deserializes_shard_content(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        shard_data = [(torch.randn(4), torch.randn(2)) for _ in range(5)]

        with patch("executor.requests.Session.get", return_value=self._fake_response(shard_data)) as mock_get:
            result = ex.get_shard(0)

        mock_get.assert_called_once_with("http://fake/shard_0.pt", stream=True, timeout=(30, 300))
        assert len(result) == 5

    def test_caches_the_same_shard_without_refetching(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        shard_data = [(torch.randn(4), torch.randn(2)) for _ in range(3)]

        with patch("executor.requests.Session.get", return_value=self._fake_response(shard_data)) as mock_get:
            first = ex.get_shard(0)
            second = ex.get_shard(0)

        assert mock_get.call_count == 1
        assert first is second

    def test_fetching_a_different_shard_evicts_the_previous_one(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        shard_a = [(torch.randn(4), torch.randn(2)) for _ in range(3)]
        shard_b = [(torch.randn(4), torch.randn(2)) for _ in range(3)]

        with patch(
            "executor.requests.Session.get", side_effect=[self._fake_response(shard_a), self._fake_response(shard_b)]
        ) as mock_get:
            ex.get_shard(0)
            ex.get_shard(1)

        assert mock_get.call_count == 2
        assert ex._shard_cache_idx == 1

    def test_out_of_range_index_raises(self, tmp_path):
        ex = self._loaded_executor(tmp_path, n_shards=2)
        with pytest.raises(IndexError):
            ex.get_shard(5)

    def test_evicts_cache_before_fetching_a_replacement(self, tmp_path):
        """The old cached shard must be freed BEFORE the new one downloads,
        not after — otherwise the peak during every shard switch is
        old shard + new shard simultaneously resident. Observable via the
        failure path: if fetching shard 1 fails, shard 0 is already gone."""
        ex = self._loaded_executor(tmp_path)
        shard_a = [(torch.randn(4), torch.randn(2)) for _ in range(3)]

        import requests as requests_module

        with (
            patch(
                "executor.requests.Session.get",
                side_effect=[
                    self._fake_response(shard_a),
                    requests_module.ConnectionError("boom"),
                    requests_module.ConnectionError("boom"),
                    requests_module.ConnectionError("boom"),
                ],
            ),
            patch("executor.time.sleep"),
        ):
            ex.get_shard(0)
            with pytest.raises(requests_module.ConnectionError):
                ex.get_shard(1)

        assert ex._shard_cache_idx is None
        assert ex._shard_cache_data is None

    def test_get_shard_retries_transient_failures(self, tmp_path):
        """A single transient network error must not kill the round — the
        fetch retries (with backoff) before giving up."""
        ex = self._loaded_executor(tmp_path)
        shard_data = [(torch.randn(4), torch.randn(2)) for _ in range(3)]

        import requests as requests_module

        with (
            patch(
                "executor.requests.Session.get",
                side_effect=[requests_module.ConnectionError("blip"), self._fake_response(shard_data)],
            ) as mock_get,
            patch("executor.time.sleep") as mock_sleep,
        ):
            result = ex.get_shard(0)

        assert mock_get.call_count == 2
        assert len(result) == 3
        mock_sleep.assert_called()  # backoff between attempts, not a hot loop

    def test_num_shards_injected_into_namespace(self, tmp_path):
        ex = self._loaded_executor(tmp_path, n_shards=3)
        ex._code = "emit_log(f'num_shards={num_shards}')"
        ex.run()
        ex.wait()
        assert any("num_shards=3" in log for log in self._logs)

    def test_get_shard_reachable_from_user_code(self, tmp_path):
        ex = self._loaded_executor(tmp_path, n_shards=1)
        shard_data = [(torch.randn(4), torch.randn(2)) for _ in range(7)]
        logs = []
        ex._emit_log = lambda t: logs.append(t)

        with patch("executor.requests.Session.get", return_value=self._fake_response(shard_data)):
            ex._code = "emit_log(f'len={len(get_shard(0))}')"
            ex.run()
            ex.wait()

        assert any("len=7" in log for log in logs)


class TestHostId:
    def test_host_id_injected_into_namespace(self, tmp_path):
        ex, _, logs = _make_executor(host_id="Machine-alpha")
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")

        ex._code = "emit_log(f'host_id={host_id}')"
        ex.run()
        ex.wait()

        assert any("host_id=Machine-alpha" in log for log in logs)

    def test_host_id_defaults_to_empty_string(self, tmp_path):
        ex, _, logs = _make_executor()
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")

        ex._code = "emit_log(f'host_id=[{host_id}]')"
        ex.run()
        ex.wait()

        assert any("host_id=[]" in log for log in logs)


class TestExecutorRun:
    def _load(self, ex, tmp_path):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")

        return model_path

    def test_emit_log_injection(self, tmp_path):
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_log('hello from test')"
        ex.run()
        ex.wait()

        assert "hello from test" in logs

    def test_emit_stats_injection(self, tmp_path):
        ex, stats, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_stats({'epoch': 1, 'loss': 0.5})"
        ex.run()
        ex.wait()

        assert any(s.get("epoch") == 1 for s in stats)
        assert any(s.get("loss") == 0.5 for s in stats)

    def test_save_model_sets_modified_state(self, tmp_path):
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = """
import torch
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
model.load_state_dict(load_model(model_path))
for p in model.parameters():
    p.data.add_(torch.ones_like(p) * 0.1)
save_model(model)
"""
        ex.run()
        ex.wait()

        assert ex.modified_state is not None
        # verify weights actually changed
        original_state = ex.get_original_state()
        for key in original_state:
            diff = (ex.modified_state[key] - original_state[key]).abs().max().item()
            assert diff > 0, f"Key '{key}' was not modified"

    def test_device_variable_injected(self, tmp_path):
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_log(f'device={device}')"
        ex.run()
        ex.wait()

        assert any("device=" in log for log in logs)

    def test_stop_uses_the_configurable_join_timeout(self, tmp_path, monkeypatch):
        """stop() must respect executor.STOP_JOIN_TIMEOUT (not a hardcoded
        value) — a thread blocked in a call that doesn't check for the async
        StopExecution exception (e.g. time.sleep, a slow network call) can
        easily outlast a few seconds; the timeout needs to be tunable."""
        import executor as executor_module

        monkeypatch.setattr(executor_module, "STOP_JOIN_TIMEOUT", 0.2)

        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)
        ex._code = "import time as _t; _t.sleep(5)"  # outlives the shortened timeout
        ex.run()
        time.sleep(0.05)  # let the thread actually start

        t0 = time.monotonic()
        ex.stop()
        elapsed = time.monotonic() - t0

        assert 0.15 < elapsed < 1.0, f"stop() took {elapsed:.2f}s, expected ~0.2s"

    def test_user_except_exception_cannot_swallow_stop(self, tmp_path):
        """Regression: user training loops commonly wrap a step in a broad
        `except Exception:` (to survive occasional NaN loss, OOM, etc.) —
        StopExecution must not be an Exception subclass, or such code would
        silently swallow a forced stop and keep training regardless."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = """
import time
time.sleep(0.15)   # allow main thread to set stop_event after run() clears it
try:
    should_stop()
except Exception:
    pass
emit_log('SHOULD NOT REACH HERE')
"""
        ex.run()
        ex._stop_event.set()  # thread is sleeping — event arrives before should_stop()
        ex.wait()

        assert "SHOULD NOT REACH HERE" not in logs

    def test_should_stop_raises_stop_execution(self, tmp_path):
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        # run() clears stop_event internally before starting the thread.
        # Set it after run() returns while the thread is still in its sleep.
        ex._code = """
import time
time.sleep(0.15)   # allow main thread to set stop_event after run() clears it
should_stop()
emit_log('SHOULD NOT REACH HERE')
"""
        ex.run()
        ex._stop_event.set()  # thread is sleeping — event arrives before should_stop()
        ex.wait()

        assert "SHOULD NOT REACH HERE" not in logs

    def test_user_exception_logs_error(self, tmp_path):
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "raise ValueError('test error')"
        ex.run()
        ex.wait()

        assert any("[ERROR]" in log for log in logs)

    def test_training_time_measured(self, tmp_path):
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "import time; time.sleep(0.05)"
        ex.run()
        ex.wait()

        assert ex.training_time >= 0.05

    def test_reset_clears_state(self, tmp_path):
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)
        task_dir = ex.work_dir

        ex._code = "save_model(None)"  # will fail → state stays None
        ex.run()
        ex.wait()

        ex.reset()
        assert ex.original_state is None
        assert ex.modified_state is None
        assert not os.path.exists(task_dir)

    def test_load_recreates_task_directory_after_reset(self, tmp_path):
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)
        task_dir = ex.work_dir
        ex.reset()

        self._load(ex, tmp_path)

        assert os.path.isdir(task_dir)

    def test_reset_never_acknowledges_a_still_running_workload(self, monkeypatch):
        ex, _, _ = _make_executor()
        stuck = MagicMock()
        stuck.name = "stuck-native-workload"
        stuck.is_alive.return_value = True
        ex._thread = stuck
        monkeypatch.setattr(ex, "stop", MagicMock())

        with pytest.raises(RuntimeError, match="restart required"):
            ex.reset()

    def test_model_in_namespace_fallback(self, tmp_path):
        """If user doesn't call save_model but leaves 'model' in namespace, it should be used."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = """
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
model.load_state_dict(load_model(model_path))
for p in model.parameters():
    p.data.fill_(99.0)
# deliberately NOT calling save_model — rely on namespace fallback
"""
        ex.run()
        ex.wait()

        # modified_state should be extracted from namespace['model'].state_dict()
        assert ex.modified_state is not None
        for tensor in ex.modified_state.values():
            if tensor.numel() > 0:
                assert tensor.abs().max().item() == pytest.approx(99.0, abs=1e-3)
                break

    def test_load_model_helper_returns_safe_state_dict(self, tmp_path):
        """load_model() exposes tensors, never an executable pickle object."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_log(f'is_dict={isinstance(load_model(model_path), dict)}')"
        ex.run()
        ex.wait()

        assert any("is_dict=True" in log for log in logs)

    def test_device_is_valid_torch_device(self, tmp_path):
        """device should be one of 'cuda', 'mps', or 'cpu'."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_log(f'dev={device}')"
        ex.run()
        ex.wait()

        valid = {"dev=cuda", "dev=mps", "dev=cpu"}
        assert any(any(v in log for v in valid) for log in logs)

    def test_params_defaults_to_empty_dict(self, tmp_path):
        """params should be injected as {} when load() is called without params."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "emit_log(f'params={params}')"
        ex.run()
        ex.wait()

        assert any("params={}" in log for log in logs)

    def test_params_values_are_accessible_in_user_code(self, tmp_path):
        """User-supplied params dict should be injected and readable by key."""
        ex, _, logs = _make_executor()
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load(
                "http://fake/model.pt",
                ["http://fake/shard_0.pt"],
                "",
                params={"learning_rate": 0.01, "label": "trial-1"},
            )

        ex._code = 'emit_log(f\'lr={params["learning_rate"]} label={params["label"]}\')'
        ex.run()
        ex.wait()

        assert any("lr=0.01 label=trial-1" in log for log in logs)

    def test_reset_clears_params(self, tmp_path):
        ex, _, _ = _make_executor()
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "", params={"learning_rate": 0.01})

        ex.reset()
        assert ex._params == {}

    def test_update_model_replaces_params_when_provided(self, tmp_path):
        """save_params() in aggregation code should reach the next round's namespace."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)

        with patch("executor._download"), patch("executor._load_model_state", return_value={"w": torch.zeros(2)}):
            ex.update_model("http://fake/updated_model.pt", params={"learning_rate": 0.005, "round": 1})

        ex._code = 'emit_log(f\'lr={params["learning_rate"]} round={params["round"]}\')'
        ex.run()
        ex.wait()

        assert any("lr=0.005 round=1" in log for log in logs)

    def test_update_model_keeps_params_when_omitted(self, tmp_path):
        """update_model(params=None) — e.g. a caller that never sends params — must not wipe existing ones."""
        ex, _, logs = _make_executor()
        self._load(ex, tmp_path)
        ex._params = {"learning_rate": 0.02}

        with patch("executor._download"), patch("executor._load_model_state", return_value={"w": torch.zeros(2)}):
            ex.update_model("http://fake/updated_model.pt")

        ex._code = "emit_log(f'lr={params[\"learning_rate\"]}')"
        ex.run()
        ex.wait()

        assert any("lr=0.02" in log for log in logs)

    def test_update_model_refreshes_dataset_shards_when_provided(self, tmp_path):
        """Regression: get_shard() reuses whatever URLs it was last given for
        the rest of that round — dataset_shards piggybacked on update_model
        (sent between every round) is how those URLs stay fresh across a
        long-running, multi-round training run instead of expiring partway
        through with 403 Forbidden."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)
        assert [item["url"] for item in ex._dataset_shards] == ["http://fake/shard_0.pt"]

        with patch("executor._download"), patch("executor._load_model_state", return_value={"w": torch.zeros(2)}):
            ex.update_model(
                "http://fake/updated_model.pt", dataset_shards=["http://fresh/shard_0.pt", "http://fresh/shard_1.pt"]
            )

        assert [item["url"] for item in ex._dataset_shards] == [
            "http://fresh/shard_0.pt",
            "http://fresh/shard_1.pt",
        ]

    def test_update_model_keeps_dataset_shards_when_omitted(self, tmp_path):
        """update_model(dataset_shards=None) — e.g. a caller that doesn't refresh
        them this round — must not wipe the ones from load()/a previous round."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        with patch("executor._download"), patch("executor._load_model_state", return_value={"w": torch.zeros(2)}):
            ex.update_model("http://fake/updated_model.pt")

        assert [item["url"] for item in ex._dataset_shards] == ["http://fake/shard_0.pt"]

    def test_update_model_does_not_evict_the_cached_shard(self, tmp_path):
        """Refreshing the URL for a shard that's already cached must not
        force a re-download — only future get_shard() calls need the new URL."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)
        ex._shard_cache_idx = 0
        ex._shard_cache_data = "cached-shard-0-data"

        with patch("executor._download"), patch("executor._load_model_state", return_value={"w": torch.zeros(2)}):
            ex.update_model("http://fake/updated_model.pt", dataset_shards=["http://fresh/shard_0.pt"])

        assert ex._shard_cache_idx == 0
        assert ex._shard_cache_data == "cached-shard-0-data"

    def test_save_report_stores_report_for_this_round(self, tmp_path):
        """save_report() should let training code hand data back to the aggregator."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "save_report({'local_loss': 0.42, 'converged': True})"
        ex.run()
        ex.wait()

        assert ex.report == {"local_loss": 0.42, "converged": True}

    def test_report_defaults_to_none_when_not_called(self, tmp_path):
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "pass"
        ex.run()
        ex.wait()

        assert ex.report is None

    def test_report_does_not_leak_into_next_round_that_skips_it(self, tmp_path):
        """A stale report from round N must not reappear after a round that doesn't call save_report."""
        ex, _, _ = _make_executor()
        self._load(ex, tmp_path)

        ex._code = "save_report({'local_loss': 0.42})"
        ex.run()
        ex.wait()
        assert ex.report == {"local_loss": 0.42}

        ex._code = "pass"
        ex.run()
        ex.wait()
        assert ex.report is None


class TestDownloadRetries:
    def _fake_stream_response(self, payload: bytes) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter([payload])
        return resp

    def test_download_retries_transient_failure_then_succeeds(self, tmp_path):
        import requests as requests_module
        from executor import _download

        dest = str(tmp_path / "model.pt")
        with (
            patch(
                "executor.requests.Session.get",
                side_effect=[requests_module.ConnectionError("blip"), self._fake_stream_response(b"model-bytes")],
            ) as mock_get,
            patch("executor.time.sleep") as mock_sleep,
        ):
            _download("http://fake/model.pt", dest)

        assert mock_get.call_count == 2
        with open(dest, "rb") as fh:
            assert fh.read() == b"model-bytes"
        mock_sleep.assert_called()

    def test_download_gives_up_after_max_attempts(self, tmp_path):
        import executor as executor_module
        import requests as requests_module
        from executor import _download

        dest = str(tmp_path / "model.pt")
        with (
            patch("executor.requests.Session.get", side_effect=requests_module.ConnectionError("down")) as mock_get,
            patch("executor.time.sleep"),
        ):
            with pytest.raises(requests_module.ConnectionError):
                _download("http://fake/model.pt", dest)

        assert mock_get.call_count == executor_module.DOWNLOAD_RETRIES

    def test_partial_download_is_discarded_on_retry(self, tmp_path):
        """A failure mid-stream must not leave the partial bytes prepended
        to the retry's content — the sink restarts from scratch."""
        import requests as requests_module
        from executor import _download

        def _failing_iter(chunk_size):
            yield b"partial-"
            raise requests_module.ChunkedEncodingError("truncated")

        bad = MagicMock()
        bad.status_code = 200
        bad.headers = {}
        bad.raise_for_status = lambda: None
        bad.iter_content = _failing_iter

        dest = str(tmp_path / "model.pt")
        with (
            patch("executor.requests.Session.get", side_effect=[bad, self._fake_stream_response(b"complete")]),
            patch("executor.time.sleep"),
        ):
            _download("http://fake/model.pt", dest)

        with open(dest, "rb") as fh:
            assert fh.read() == b"complete"

    def test_partial_download_resumes_when_server_honours_range(self, tmp_path):
        import requests as requests_module
        from executor import _download

        def _failing_iter(chunk_size):
            yield b"partial-"
            raise requests_module.ChunkedEncodingError("truncated")

        first = self._fake_stream_response(b"")
        first.iter_content = _failing_iter
        resumed = self._fake_stream_response(b"complete")
        resumed.status_code = 206
        resumed.headers = {"Content-Range": "bytes 8-15/16"}

        dest = str(tmp_path / "model.pt")
        with (
            patch("executor.requests.Session.get", side_effect=[first, resumed]) as get,
            patch("executor.time.sleep"),
        ):
            _download("http://fake/model.pt", dest)

        assert get.call_args_list[1].kwargs["headers"] == {"Range": "bytes=8-"}
        with open(dest, "rb") as fh:
            assert fh.read() == b"partial-complete"


class TestShutdown:
    def test_shutdown_removes_the_tmpdir(self):
        """Every reconnect builds a fresh Executor (fresh tmpdir) — without
        cleanup at session end, each one leaked a directory holding a
        potentially multi-GB downloaded model until the disk filled."""
        ex, _, _ = _make_executor()
        tmpdir = ex._tmpdir
        with open(os.path.join(tmpdir, "model.pt"), "wb") as fh:
            fh.write(b"x" * 1024)
        assert os.path.isdir(tmpdir)

        ex.shutdown()

        assert not os.path.exists(tmpdir)

    def test_shutdown_is_idempotent(self):
        ex, _, _ = _make_executor()
        ex.shutdown()
        ex.shutdown()  # second call must not raise

    def test_shutdown_stops_a_running_thread_first(self, tmp_path):
        import executor as executor_module

        ex, _, _ = _make_executor()
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")

        # Interruptible loop — the async StopExecution can only fire between
        # bytecode instructions, so one long C-level sleep(30) would survive
        # it (documented limitation); many short sleeps get interrupted fast.
        ex._code = "import time as _t\nwhile True:\n    _t.sleep(0.01)"
        ex.run()
        time.sleep(0.05)

        with patch.object(executor_module, "STOP_JOIN_TIMEOUT", 5):
            ex.shutdown()

        assert not ex.is_running()
        assert not os.path.exists(ex._tmpdir)


class TestAbortableDownloads:
    """Stop must be able to interrupt an in-flight download — previously a
    stop during a multi-GB model/shard transfer had no effect until the
    transfer finished (or its socket timeout expired)."""

    def _fake_stream_response(self, chunks) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter(chunks)
        return resp

    def test_fetch_aborts_between_chunks_when_event_is_set(self):
        from executor import DownloadAborted, _fetch_to

        ev = threading.Event()

        def chunk_gen():
            yield b"chunk-1"
            ev.set()  # stop arrives mid-download
            yield b"chunk-2"
            raise AssertionError("must not be consumed after abort")

        sink = io.BytesIO()
        with (
            patch("executor.requests.Session.get", return_value=self._fake_stream_response(chunk_gen())) as mock_get,
            patch("executor.time.sleep"),
            pytest.raises(DownloadAborted),
        ):
            _fetch_to("http://fake/model.pt", sink, abort_event=ev)

        assert mock_get.call_count == 1  # an abort is NOT retried

    def test_load_clears_a_stale_stop_flag_before_downloading(self, tmp_path):
        """A stop from the PREVIOUS run must not abort the next run's
        load — load() clears the flag before starting its download."""
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, _, _ = _make_executor()
        ex._stop_event.set()  # stale, from a previous run's stop

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")

        assert not ex._stop_event.is_set()

    def test_download_receives_the_stop_event_as_abort_signal(self, tmp_path):
        """load() and update_model() must wire self._stop_event into the
        download so executor.stop() aborts an in-flight transfer."""
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, _, _ = _make_executor()

        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/shard_0.pt"], "")
            ex.update_model("http://fake/model2.pt")

        for call in mock_dl.call_args_list:
            assert call.kwargs.get("abort_event") is ex._stop_event


class TestUpdateModelViaDelta:
    """L3: update_model applies an fp16 delta to the model file the host
    already holds instead of re-downloading the full model — guarded by
    base/result hashes with automatic fallback to the full download."""

    def _write_state_file(self, path, state):
        torch.save(state, path)

    def _make_state(self, seed=0):
        torch.manual_seed(seed)
        return {"weight": torch.randn(8, 8), "bias": torch.randn(8)}

    def _loaded(self, tmp_path, state):
        model_path = str(tmp_path / "model.pt")
        self._write_state_file(model_path, state)
        ex, _, _ = _make_executor()
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", [], "")
        return ex

    def _sync_payload(self, old, new):
        return _make_sync_payload(old, new)

    def _delta_response(self, blob):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter([blob])
        return resp

    @staticmethod
    def _write_delta(blob):
        def download(url, destination, **_kwargs):
            if url.endswith(".zst"):
                with open(destination, "wb") as payload:
                    payload.write(blob)

        return download

    def test_applies_delta_bit_exact_without_full_download(self, tmp_path):
        old = self._make_state(seed=0)
        new = self._make_state(seed=1)
        ex = self._loaded(tmp_path, old)
        blob, canonical, base_hash, result_hash = self._sync_payload(old, new)

        with patch("executor._download", side_effect=self._write_delta(blob)) as download:
            ex.update_model(
                "http://fake/full.pt", delta_url="http://fake/delta.zst", base_hash=base_hash, result_hash=result_hash
            )

        assert [call.args[0] for call in download.call_args_list] == ["http://fake/delta.zst"]
        saved = torch.load(ex._model_path, weights_only=True, map_location="cpu")
        for key in canonical:
            assert torch.equal(saved[key], canonical[key]), f"saved[{key}] must be bit-exactly the canonical state"

    def test_keys_absent_from_delta_stay_unchanged(self, tmp_path):
        old = self._make_state(seed=0)
        new = {k: v.clone() for k, v in old.items()}
        new["bias"] = new["bias"] + 0.5  # weight untouched → omitted from delta
        ex = self._loaded(tmp_path, old)
        blob, canonical, base_hash, result_hash = self._sync_payload(old, new)

        with patch("executor._download", side_effect=self._write_delta(blob)):
            ex.update_model(
                "http://fake/full.pt", delta_url="http://fake/delta.zst", base_hash=base_hash, result_hash=result_hash
            )

        saved = torch.load(ex._model_path, weights_only=True, map_location="cpu")
        assert torch.equal(saved["weight"], old["weight"].float())

    def test_base_hash_mismatch_falls_back_to_full_download(self, tmp_path):
        """e.g. user code overwrote the model file, or a rejoined host with
        a fresh tmpdir — the local base is not what middleware thinks."""
        old = self._make_state(seed=0)
        new = self._make_state(seed=1)
        ex = self._loaded(tmp_path, old)
        blob, _, _, result_hash = self._sync_payload(old, new)

        with (
            patch("executor.requests.Session.get", return_value=self._delta_response(blob)) as mock_get,
            patch("executor._download") as mock_full,
        ):
            ex.update_model(
                "http://fake/full.pt", delta_url="http://fake/delta.zst", base_hash="0" * 64, result_hash=result_hash
            )

        mock_full.assert_called_once()
        assert mock_full.call_args.args[0] == "http://fake/full.pt"
        mock_get.assert_not_called()  # delta not even downloaded — base check is first

    def test_result_hash_mismatch_falls_back_to_full_download(self, tmp_path):
        old = self._make_state(seed=0)
        new = self._make_state(seed=1)
        ex = self._loaded(tmp_path, old)
        blob, _, base_hash, _ = self._sync_payload(old, new)

        with patch("executor._download", side_effect=self._write_delta(blob)) as download:
            ex.update_model(
                "http://fake/full.pt", delta_url="http://fake/delta.zst", base_hash=base_hash, result_hash="0" * 64
            )

        assert [call.args[0] for call in download.call_args_list] == ["http://fake/delta.zst", "http://fake/full.pt"]

    def test_no_delta_url_keeps_todays_full_download_path(self, tmp_path):
        old = self._make_state(seed=0)
        ex = self._loaded(tmp_path, old)

        with patch("executor._download") as mock_full:
            ex.update_model("http://fake/full.pt")

        mock_full.assert_called_once()

    def test_stop_aborts_the_delta_download_instead_of_falling_back(self, tmp_path):
        """A stop mid-sync must abort the update entirely (DownloadAborted
        propagates), NOT silently degrade into a full download."""
        from executor import DownloadAborted

        old = self._make_state(seed=0)
        new = self._make_state(seed=1)
        ex = self._loaded(tmp_path, old)
        blob, _, base_hash, result_hash = self._sync_payload(old, new)

        ex._stop_event.set()  # stop lands before/during the sync
        with patch("executor._download", side_effect=DownloadAborted("stopped")) as download:
            with pytest.raises(DownloadAborted):
                ex.update_model(
                    "http://fake/full.pt",
                    delta_url="http://fake/delta.zst",
                    base_hash=base_hash,
                    result_hash=result_hash,
                )

        download.assert_called_once()


class TestMmapModelLoading:
    """R3: get_original_state() reloads the (potentially multi-GB) model at
    every round boundary for the delta computation — mmap-backed loading
    turns that from a full read+copy into lazy page-in during the diff."""

    def _state_file(self, tmp_path):
        path = str(tmp_path / "model.pt")
        torch.save({"w": torch.randn(8, 8), "b": torch.randn(8)}, path)
        return path

    def _loaded(self, tmp_path):
        path = self._state_file(tmp_path)
        ex, _, _ = _make_executor()
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", [], "")
        return ex, path

    def test_get_original_state_requests_mmap(self, tmp_path):
        import executor as executor_module

        ex, _ = self._loaded(tmp_path)

        real_load = torch.load
        calls = []

        def capturing_load(*args, **kwargs):
            calls.append(kwargs)
            return real_load(*args, **kwargs)

        with patch.object(executor_module.torch, "load", capturing_load):
            state = ex.get_original_state()

        assert state is not None and "w" in state
        assert any(kw.get("mmap") is True for kw in calls), "round-boundary reload must use mmap, not a full read+copy"

    def test_mmap_failure_falls_back_to_regular_load(self, tmp_path):
        import executor as executor_module

        ex, _ = self._loaded(tmp_path)

        real_load = torch.load

        def flaky_load(*args, **kwargs):
            if kwargs.get("mmap"):
                raise RuntimeError("mmap unsupported for this file")
            return real_load(*args, **kwargs)

        with patch.object(executor_module.torch, "load", flaky_load):
            state = ex.get_original_state()

        assert state is not None and "w" in state

    def test_delta_update_base_load_does_not_use_mmap(self, tmp_path):
        """_try_delta_update REWRITES the same file it loaded the base
        from — mmap-backed tensors whose backing file gets overwritten
        mid-save is a correctness hazard, so that path must stay a full
        read."""
        import executor as executor_module

        old = {"w": torch.randn(8, 8)}
        new = {"w": torch.randn(8, 8)}
        path = str(tmp_path / "model.pt")
        torch.save(old, path)
        ex, _, _ = _make_executor()
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", [], "")

        blob, _canonical, base_hash, result_hash = _make_sync_payload(old, new)

        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter([blob])

        real_load = torch.load
        mmap_calls = []

        def capturing_load(*args, **kwargs):
            if kwargs.get("mmap"):
                mmap_calls.append(args)
            return real_load(*args, **kwargs)

        with (
            patch.object(executor_module.torch, "load", capturing_load),
            patch("executor._download", side_effect=TestUpdateModelViaDelta._write_delta(blob)),
        ):
            ex.update_model(
                "http://fake/full.pt", delta_url="http://fake/d.zst", base_hash=base_hash, result_hash=result_hash
            )

        assert all(call[0] != ex._model_path for call in mmap_calls), "delta-update base load must NOT be mmap-backed"


class TestAtomicStateSave:
    def test_failed_save_leaves_the_original_file_intact(self, tmp_path):
        """_save_state_file writes via tmp+rename — a crash/error mid-save
        must not leave a truncated model file behind (the next round's
        base-hash check would fail it into a full download, but a corrupt
        file on disk is still worth never producing)."""
        import executor as executor_module
        from executor import _save_state_file

        path = str(tmp_path / "model.pt")
        original = {"w": torch.ones(4)}
        torch.save(original, path)

        def partial_write_then_fail(state, dest):
            # simulates dying mid-write: whatever file torch.save was
            # given gets truncated/garbled before the error surfaces
            with open(dest, "wb") as fh:
                fh.write(b"garbage")
            raise RuntimeError("disk full")

        with patch.object(executor_module.torch, "save", side_effect=partial_write_then_fail):
            with pytest.raises(RuntimeError):
                _save_state_file({"w": torch.zeros(4)}, path)

        recovered = torch.load(path, weights_only=True)
        assert torch.equal(recovered["w"], original["w"])


class _ShardTestBase:
    """Shared helpers for disk-cache/prefetch tests."""

    def _loaded_executor(self, tmp_path, n_shards=3):
        model_path = str(tmp_path / "model.pt")
        _make_model_file(model_path)
        ex, _, _ = _make_executor()
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", [f"http://fake/shard_{i}.pt" for i in range(n_shards)], "")
        return ex

    def _fake_response(self, payload) -> MagicMock:
        buf = io.BytesIO()
        torch.save(payload, buf)
        content = buf.getvalue()
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.iter_content = lambda chunk_size: iter([content])
        return resp

    def _shard(self, n=3):
        return [(torch.randn(4), torch.randn(2)) for _ in range(n)]


class TestShardDiskCache(_ShardTestBase):
    """R2: raw shard bytes are cached on disk (inside the executor tmpdir,
    LRU with a size cap) — the default training code visits a different
    shard every round, so without this every shard gets re-downloaded once
    per epoch for the whole run."""

    @pytest.fixture(autouse=True)
    def _no_prefetch(self, monkeypatch):
        monkeypatch.setattr(Executor, "_start_prefetch", lambda self, i: None, raising=False)

    def test_shard_served_from_disk_after_memory_eviction(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        s0, s1 = self._shard(), self._shard()

        with patch(
            "executor.requests.Session.get", side_effect=[self._fake_response(s0), self._fake_response(s1)]
        ) as mock_get:
            first = ex.get_shard(0)
            ex.get_shard(1)  # evicts shard 0 from memory
            again = ex.get_shard(0)  # must come from disk, not the network

        assert mock_get.call_count == 2
        assert len(again) == len(first)

    def test_lru_evicts_oldest_when_over_the_cap(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        s0, s1 = self._shard(), self._shard()

        with patch("executor.requests.Session.get", return_value=self._fake_response(s0)) as mock_get:
            ex.get_shard(0)
        # cap below two shards' worth: adding shard 1 must evict shard 0
        blob_size = os.path.getsize(ex._shard_disk_path(0))
        ex._shard_cache_max_bytes = int(blob_size * 1.5)

        with patch(
            "executor.requests.Session.get", side_effect=[self._fake_response(s1), self._fake_response(s0)]
        ) as mock_get:
            ex.get_shard(1)  # over cap → shard 0 evicted from disk
            ex.get_shard(0)  # gone from disk → network again

        assert mock_get.call_count == 2

    def test_disk_cache_cleared_on_a_new_load(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        s0 = self._shard()
        with patch("executor.requests.Session.get", return_value=self._fake_response(s0)) as mock_get:
            ex.get_shard(0)
        assert mock_get.call_count == 1

        # a new task (new run, possibly a different dataset) must not serve
        # stale shards from the previous run's cache
        model_path = str(tmp_path / "model2.pt")
        _make_model_file(model_path)
        with patch("executor._download") as mock_dl:

            def side_effect(url, dest, **kw):
                import shutil

                shutil.copy(model_path, dest)

            mock_dl.side_effect = side_effect
            ex.load("http://fake/model.pt", ["http://fake/other_shard_0.pt"], "")

        with patch("executor.requests.Session.get", return_value=self._fake_response(s0)) as mock_get2:
            ex.get_shard(0)
        assert mock_get2.call_count == 1  # network again — cache was cleared

    def test_zero_cap_disables_the_disk_cache(self, tmp_path):
        ex = self._loaded_executor(tmp_path)
        ex._shard_cache_max_bytes = 0
        s0, s1 = self._shard(), self._shard()

        with patch(
            "executor.requests.Session.get",
            side_effect=[self._fake_response(s0), self._fake_response(s1), self._fake_response(s0)],
        ) as mock_get:
            ex.get_shard(0)
            ex.get_shard(1)
            ex.get_shard(0)  # no disk cache → network every switch

        assert mock_get.call_count == 3


class TestShardPrefetch(_ShardTestBase):
    """R1: serving shard i kicks off a background download of shard
    (i+1) % num_shards into the disk cache — the default training code's
    access pattern is exactly sequential-per-round, so next round's shard
    is already local by the time the round starts, instead of the GPU
    idling through a synchronous download."""

    def test_get_shard_triggers_prefetch_of_the_next_index(self, tmp_path, monkeypatch):
        ex = self._loaded_executor(tmp_path, n_shards=3)
        started = []
        monkeypatch.setattr(Executor, "_start_prefetch", lambda self, i: started.append(i), raising=False)
        with patch("executor.requests.Session.get", return_value=self._fake_response(self._shard())):
            ex.get_shard(0)
        assert started == [1]

    def test_prefetch_wraps_around_the_last_shard(self, tmp_path, monkeypatch):
        ex = self._loaded_executor(tmp_path, n_shards=3)
        started = []
        monkeypatch.setattr(Executor, "_start_prefetch", lambda self, i: started.append(i), raising=False)
        with patch("executor.requests.Session.get", return_value=self._fake_response(self._shard())):
            ex.get_shard(2)
        assert started == [0]

    def test_single_shard_run_never_prefetches(self, tmp_path, monkeypatch):
        ex = self._loaded_executor(tmp_path, n_shards=1)
        started = []
        monkeypatch.setattr(Executor, "_prefetch_shard", lambda self, i: started.append(i), raising=False)
        with patch("executor.requests.Session.get", return_value=self._fake_response(self._shard())):
            ex.get_shard(0)
        time.sleep(0.05)
        assert started == []

    def test_prefetched_shard_is_served_without_a_network_call(self, tmp_path):
        ex = self._loaded_executor(tmp_path, n_shards=3)
        s1 = self._shard()

        with patch("executor.requests.Session.get", return_value=self._fake_response(s1)) as mock_get:
            ex._prefetch_shard(1)  # direct call — thread orchestration tested separately
        assert mock_get.call_count == 1

        with patch("executor.requests.Session.get") as mock_get2, patch.object(Executor, "_start_prefetch"):
            result = ex.get_shard(1)  # served from the prefetched disk copy
        mock_get2.assert_not_called()
        assert len(result) == len(s1)

    def test_prefetch_failure_is_silent(self, tmp_path):
        import requests as requests_module

        ex = self._loaded_executor(tmp_path, n_shards=3)
        with (
            patch("executor.requests.Session.get", side_effect=requests_module.ConnectionError("down")),
            patch("executor.time.sleep"),
        ):
            ex._prefetch_shard(1)  # must not raise

    def test_prefetch_skips_an_already_cached_shard(self, tmp_path):
        ex = self._loaded_executor(tmp_path, n_shards=3)
        with patch("executor.requests.Session.get", return_value=self._fake_response(self._shard())) as mock_get:
            ex._prefetch_shard(1)
            ex._prefetch_shard(1)
        assert mock_get.call_count == 1
