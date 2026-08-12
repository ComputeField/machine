# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Build sparse-or-dense fp16 model deltas and compress them with Zstandard."""

import hashlib
import logging
import os
import tempfile
import threading

import torch
import zstandard as zstd

logger = logging.getLogger(__name__)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """SHA-256 over the fp32-canonicalized state_dict — verifies the
    base/result of a delta-based model sync (see executor.update_model).
    MUST stay byte-identical to the platform protocol. Both repositories
    lock this contract to the same golden digest in their own tests."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous().float()
        h.update(key.encode())
        h.update(str(tuple(t.shape)).encode())
        # .data (a memoryview) — same bytes as .tobytes() without copying
        # the whole tensor first; on a multi-GB model that copy alone cost
        # seconds and doubled the transient allocation per key.
        h.update(t.numpy().data)
    return h.hexdigest()


SPARSE_THRESHOLD = 0.5  # Use sparse COO only when more than half is zero.
ZSTD_LEVEL = 3

# threads=-1 → one worker per CPU core. Output is a standard zstd frame,
# byte-compatible with orchestrator's one-shot ZstdDecompressor (verified),
# but compression of a multi-GB delta drops from seconds to a fraction.
_compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)


class DeltaAborted(Exception):
    """Delta construction was cancelled by the active task."""


def _check_abort(abort_event: threading.Event | None) -> None:
    if abort_event is not None and abort_event.is_set():
        raise DeltaAborted("delta construction stopped")


def _build_parts(
    original: dict[str, torch.Tensor],
    modified: dict[str, torch.Tensor],
    threshold: float,
    abort_event: threading.Event | None = None,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Consume two state dicts and return encoded tensors plus counters."""
    parts: dict[str, torch.Tensor] = {}
    total_params = nonzero_params = sparse_keys = 0
    for key in list(modified):
        _check_abort(abort_event)
        changed = modified.pop(key)
        baseline = original.pop(key, None)
        if baseline is None:
            continue
        # Keep caller tensors intact; only dictionary ownership is consumed.
        diff = changed.detach().to(dtype=torch.float32, copy=True)
        diff.sub_(baseline.float())
        if threshold > 0:
            diff[diff.abs() < threshold] = 0
        encoded = diff.half()
        nonzero = int(encoded.count_nonzero().item())
        count = encoded.numel()
        total_params += count
        nonzero_params += nonzero
        if count and 1 - nonzero / count > SPARSE_THRESHOLD:
            encoded = encoded.to_sparse()
            sparse_keys += 1
        parts[key] = encoded
    original.clear()
    return parts, total_params, nonzero_params, sparse_keys


def compute_delta_file(
    original: dict[str, torch.Tensor],
    modified: dict[str, torch.Tensor],
    destination: str,
    threshold: float = 0.0,
    abort_event: threading.Event | None = None,
) -> dict:
    """Write a compressed delta to disk for direct multipart upload.

    Inputs are consumed progressively so tensors can be released while the
    delta is built. The compressed payload is never materialized as `bytes`.
    """
    parts, total_params, nonzero_params, sparse_keys = _build_parts(
        original,
        modified,
        threshold,
        abort_event,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        dir=os.path.dirname(os.path.abspath(destination)),
        delete=False,
    ) as raw_file:
        raw_path = raw_file.name
    try:
        _check_abort(abort_event)
        torch.save(parts, raw_path)
        parts.clear()
        _check_abort(abort_event)
        raw_size = os.path.getsize(raw_path)
        with open(raw_path, "rb") as source, open(destination, "wb") as target:
            # Supplying the pledged size keeps the frame compatible with both
            # streaming readers and one-shot `decompress()` consumers.
            with _compressor.stream_writer(target, size=raw_size, closefd=False) as compressor:
                while block := source.read(8 * 1024 * 1024):
                    _check_abort(abort_event)
                    compressor.write(block)
    finally:
        try:
            os.unlink(raw_path)
        except FileNotFoundError:
            pass

    compressed_size = os.path.getsize(destination)
    digest = hashlib.sha256()
    with open(destination, "rb") as payload:
        for block in iter(lambda: payload.read(8 * 1024 * 1024), b""):
            _check_abort(abort_event)
            digest.update(block)
    sparsity = 1 - nonzero_params / total_params if total_params else 0.0
    return {
        "size_raw_mb": round(raw_size / 1e6, 3),
        "size_compressed_mb": round(compressed_size / 1e6, 3),
        "compression_ratio": round(compressed_size / raw_size, 4) if raw_size else 0.0,
        "sparsity": round(sparsity, 4),
        "sparse_keys": sparse_keys,
        "size_bytes": compressed_size,
        "sha256": digest.hexdigest(),
    }
