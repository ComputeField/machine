#!/usr/bin/env python3
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""End-to-end controller, OS sandbox, PyTorch and IPC smoke test."""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import Executor  # noqa: E402
from sandbox_runtime import self_test  # noqa: E402


def main() -> None:
    root = tempfile.mkdtemp(prefix="computefield-sandbox-smoke-")
    stats: list[dict] = []
    logs: list[str] = []
    executor = None
    try:
        backend = self_test(root)
        executor = Executor(
            stats.append,
            logs.append,
            work_dir=root,
            stage_first_shard=False,
            sandboxed=True,
        )
        executor._code = (
            "model = torch.nn.Linear(2, 1).to(device)\n"
            "value = model(torch.ones(1, 2, device=device)).sum().item()\n"
            "save_model(model)\n"
            "emit_stats({'kind': 'training', 'device': device, 'finite': bool(torch.isfinite(torch.tensor(value)))})"
        )
        executor.run(mode="training")
        executor.wait()
        if executor.execution_error is not None:
            raise RuntimeError(str(executor.execution_error))
        expected = [{"kind": "training", "device": executor._device(), "finite": True}]
        if stats != expected:
            raise RuntimeError(f"unexpected sandbox stats: {stats!r}; logs={logs!r}")
        if not executor.modified_state or set(executor.modified_state) != {"weight", "bias"}:
            raise RuntimeError("sandbox did not return the expected state_dict")
        print(f"sandbox smoke passed: {backend}, {stats[0]['device']}")
    finally:
        if executor is not None:
            executor.shutdown()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
