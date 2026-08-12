# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/speedtest.py — the post-registration upload/
download speed test against gpu-manager (not MinIO, see the design doc for
why: Machine has no MinIO credentials of its own, and gpu-manager already
carries real bulk traffic — the compressed model delta)."""

import json
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("minio", MagicMock())
sys.modules.setdefault("minio_client", MagicMock())

import speedtest


class FakeWS:
    """Records what was sent; recv() returns a scripted queue of frames."""

    def __init__(self, recv_queue):
        self.sent = []
        self._recv_queue = list(recv_queue)

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return self._recv_queue.pop(0)


def _make_clock(times):
    """A fake clock passed explicitly via run_speed_test's `clock` param —
    NOT a monkeypatch of the real time.monotonic, which would also affect
    pytest-asyncio's own event-loop internals running the same test."""
    it = iter(times)
    return lambda: next(it)


@pytest.mark.asyncio
async def test_sends_a_size_header_then_a_binary_payload_of_that_size():
    ws = FakeWS(recv_queue=[b"x" * (2 * 1024 * 1024)])

    await speedtest.run_speed_test(ws, payload_mb=2, clock=_make_clock([0.0, 1.0, 2.0]))

    header = json.loads(ws.sent[0])
    assert header["type"] == "speedtest_upload"
    assert header["size_bytes"] == 2 * 1024 * 1024
    binary_sent = ws.sent[1]
    assert isinstance(binary_sent, (bytes, bytearray))
    assert len(binary_sent) == 2 * 1024 * 1024


@pytest.mark.asyncio
async def test_computes_upload_and_download_mbps_from_timings():
    payload_mb = 1
    payload_bytes = payload_mb * 1024 * 1024
    ws = FakeWS(recv_queue=[b"x" * payload_bytes])
    # t0=0 (before header+payload), t1=1 (upload done, i.e. 1s to send 1 MiB),
    # t2=3 (download done, i.e. 2s to receive 1 MiB back)
    clock = _make_clock([0.0, 1.0, 3.0])

    result = await speedtest.run_speed_test(ws, payload_mb=payload_mb, clock=clock)

    expected_upload_mbps = (payload_bytes * 8) / 1.0 / 1e6
    expected_download_mbps = (payload_bytes * 8) / 2.0 / 1e6
    assert result["upload_mbps"] == pytest.approx(expected_upload_mbps)
    assert result["download_mbps"] == pytest.approx(expected_download_mbps)
    assert result["payload_bytes"] == payload_bytes


@pytest.mark.asyncio
async def test_payload_is_random_not_all_zero_or_repeating():
    """A payload of all-zeros (or otherwise trivially compressible) risks
    an intermediary silently compressing it and skewing the measurement —
    real transfers (model weights, deltas) aren't compressible like that."""
    ws = FakeWS(recv_queue=[b"x" * (1024 * 1024)])

    await speedtest.run_speed_test(ws, payload_mb=1, clock=_make_clock([0.0, 1.0, 2.0]))

    binary_sent = ws.sent[1]
    assert len(set(binary_sent[:256])) > 10  # not constant/trivially patterned


@pytest.mark.asyncio
async def test_raises_on_echo_size_mismatch():
    """A malformed/truncated echo must surface as an error, not silently
    report a bogus download speed."""
    ws = FakeWS(recv_queue=[b"too short"])

    with pytest.raises(speedtest.SpeedTestError):
        await speedtest.run_speed_test(ws, payload_mb=1, clock=_make_clock([0.0, 1.0, 2.0]))
