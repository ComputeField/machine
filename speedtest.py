# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Estimate broker link throughput using one bounded echoed binary frame.

Bulk model traffic uses object storage; this diagnostic compares host control
links without distributing MinIO credentials.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class SpeedTestError(Exception):
    pass


async def run_speed_test(ws, payload_mb: int, clock=time.monotonic) -> dict:
    """Returns {"upload_mbps", "download_mbps", "payload_bytes"}. Raises
    SpeedTestError if the echoed payload doesn't match what was sent.

    clock is injectable (defaults to time.monotonic) rather than called
    directly — monkeypatching the real time.monotonic globally in tests
    would also affect pytest-asyncio's own event-loop internals."""
    payload = os.urandom(payload_mb * 1024 * 1024)
    size = len(payload)

    t0 = clock()
    await ws.send(json.dumps({"type": "speedtest_upload", "size_bytes": size}))
    await ws.send(payload)
    t1 = clock()

    echoed = await ws.recv()
    t2 = clock()

    if len(echoed) != size:
        raise SpeedTestError(f"speed test echo size mismatch: sent {size} bytes, got back {len(echoed)}")

    upload_s = max(t1 - t0, 1e-6)  # guard against a near-zero elapsed time on a very fast loopback
    download_s = max(t2 - t1, 1e-6)
    return {
        "upload_mbps": (size * 8) / upload_s / 1e6,
        "download_mbps": (size * 8) / download_s / 1e6,
        "payload_bytes": size,
    }
