# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/delta.py"""

import io
import os
import tempfile
import threading

import pytest
import torch
import zstandard
import delta
from delta import DeltaAborted, compute_delta_file, state_hash


def _make_state(shape=(10, 10), seed=0):
    torch.manual_seed(seed)
    return {"weight": torch.randn(shape), "bias": torch.randn(shape[0])}


def _write_delta(original: dict, modified: dict) -> tuple[bytes, dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "delta.zst")
        stats = compute_delta_file(original, modified, path)
        with open(path, "rb") as payload:
            return payload.read(), stats


class TestComputeDelta:
    def test_raw_intermediate_uses_the_crash_cleaned_task_directory(self, tmp_path, monkeypatch):
        destination = tmp_path / "task" / "delta.zst"
        destination.parent.mkdir()
        observed = {}
        create = tempfile.NamedTemporaryFile

        def named_temporary_file(*args, **kwargs):
            observed.update(kwargs)
            return create(*args, **kwargs)

        monkeypatch.setattr(delta.tempfile, "NamedTemporaryFile", named_temporary_file)
        compute_delta_file(_make_state(), _make_state(seed=1), str(destination))

        assert observed["dir"] == str(destination.parent)

    def test_returns_bytes_and_stats(self):
        original = _make_state()
        modified = _make_state(seed=1)
        result, stats = _write_delta(original, modified)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert isinstance(stats, dict)
        for key in ("size_raw_mb", "size_compressed_mb", "compression_ratio", "sparsity"):
            assert key in stats, f"Missing stats key: {key}"

    def test_round_trip(self):
        """Apply delta to original → should recover modified. Inputs are
        passed as copies — delta generation consumes its arguments (see
        test_inputs_are_consumed_progressively)."""
        original = _make_state(seed=0)
        modified = _make_state(seed=1)

        compressed, _ = _write_delta(dict(original), dict(modified))

        # decompress and apply
        raw = zstandard.ZstdDecompressor().decompress(compressed)
        delta = torch.load(io.BytesIO(raw), weights_only=True)

        for key in original:
            recovered = original[key].float() + delta[key].to_dense().float()
            # delta is stored as float16 (eps ≈ 2⁻¹⁰), so absolute error ≈ |v| × 1e-3
            assert torch.allclose(recovered, modified[key].float(), atol=2e-3), f"Round-trip failed for key '{key}'"

    def test_identical_models_produce_zero_delta(self):
        """If original == modified, delta should be all zeros (fully sparse)."""
        original = _make_state(seed=42)
        modified = {k: v.clone() for k, v in original.items()}

        compressed, stats = _write_delta(original, modified)

        raw = zstandard.ZstdDecompressor().decompress(compressed)
        delta = torch.load(io.BytesIO(raw), weights_only=True)

        for key, tensor in delta.items():
            assert tensor.to_dense().abs().max().item() < 1e-6, f"Expected zero delta for key '{key}'"

        assert stats["sparsity"] == pytest.approx(1.0, abs=1e-4)

    def test_compression_reduces_size(self):
        """zstd output should be smaller than raw serialised bytes for typical models."""
        original = _make_state(shape=(100, 100))
        modified = {k: v + 0.001 for k, v in original.items()}  # tiny change → high sparsity

        compressed, stats = _write_delta(original, modified)

        assert stats["size_compressed_mb"] <= stats["size_raw_mb"], "Compressed size should not exceed raw size"
        assert stats["compression_ratio"] <= 1.0

    def test_skips_keys_not_in_original(self):
        """Extra keys in modified that are absent in original must be ignored."""
        original = {"weight": torch.randn(5, 5)}
        modified = {"weight": torch.randn(5, 5), "extra_key": torch.randn(3)}

        compressed, _ = _write_delta(original, modified)
        raw = zstandard.ZstdDecompressor().decompress(compressed)
        delta = torch.load(io.BytesIO(raw), weights_only=True)

        assert "extra_key" not in delta
        assert "weight" in delta

    def test_stats_sparsity_range(self):
        """Sparsity is always in [0, 1]."""
        original = _make_state(seed=0)
        modified = _make_state(seed=99)

        _, stats = _write_delta(original, modified)
        assert 0.0 <= stats["sparsity"] <= 1.0

    def test_inputs_are_consumed_progressively(self):
        """Delta generation pops entries out of both input dicts as it goes —
        deliberately, so each key's original/modified tensors become
        collectable as soon as its diff is computed, instead of both full
        fp32 state_dicts staying resident until the very end. On a multi-GB
        model that difference is roughly a whole extra model's worth of peak
        RAM. Callers (main.py's get_model) discard both dicts right after
        anyway; tests that need the inputs afterwards pass copies."""
        original = _make_state(seed=0)
        original["orphan"] = torch.randn(3)  # key absent from modified
        modified = _make_state(seed=1)
        modified["extra_key"] = torch.randn(3)  # key absent from original

        _write_delta(original, modified)

        assert original == {}
        assert modified == {}

    def test_pre_cancelled_delta_fails_before_serialization(self, tmp_path):
        stop = threading.Event()
        stop.set()

        with pytest.raises(DeltaAborted):
            compute_delta_file(
                {"weight": torch.zeros(4)},
                {"weight": torch.ones(4)},
                str(tmp_path / "cancelled.zst"),
                abort_event=stop,
            )


def test_state_hash_matches_protocol_golden_value():
    state = {
        "b": torch.arange(4, dtype=torch.float32),
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    assert state_hash(state) == "df30bad89fe42827bdb24a0ebb23082032e19f7fde693dd7e3b5a83c7eeccfd7"
