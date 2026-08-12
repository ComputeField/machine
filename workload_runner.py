# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Credential-free child process used exclusively for workload execution."""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
from pathlib import Path

import torch
from executor import Executor

_protocol_out = sys.stdout
_protocol_in = sys.stdin


def _apply_resource_limits(result_max_bytes: int) -> None:
    """Set hard limits that workload code cannot raise again."""

    def lower(kind: int, requested: int) -> None:
        _, hard = resource.getrlimit(kind)
        target = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (target, target))

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lower(resource.RLIMIT_NOFILE, 1024)
    if hasattr(resource, "RLIMIT_NPROC"):
        lower(resource.RLIMIT_NPROC, 512)
    lower(resource.RLIMIT_FSIZE, result_max_bytes)


def _send(message: dict) -> None:
    _protocol_out.write(json.dumps(message, separators=(",", ":"), default=str) + "\n")
    _protocol_out.flush()


class _LogStream:
    def __init__(self, level: str) -> None:
        self.level = level
        self.buffer = ""

    def write(self, value: str) -> int:
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line:
                _send({"type": "log", "text": f"[{self.level}] {line}"})
        return len(value)

    def flush(self) -> None:
        if self.buffer:
            _send({"type": "log", "text": f"[{self.level}] {self.buffer}"})
            self.buffer = ""


def _request_shard(index: int):
    _send({"type": "shard", "index": index})
    response = json.loads(_protocol_in.readline())
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "controller rejected shard request"))
    path = str(response["path"])
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_dir = str(spec_path.parent)
    # The runner must not inherit a platform credential even if a future
    # launcher accidentally adds one to the environment.
    for key in tuple(os.environ):
        if "CREDENTIAL" in key or "TOKEN" in key or "SECRET" in key or key.startswith("MACHINE_"):
            os.environ.pop(key, None)
    _apply_resource_limits(int(spec["result_max_bytes"]))

    executor = Executor(
        emit_stats=lambda data: _send({"type": "stats", "data": data}),
        emit_log=lambda text: _send({"type": "log", "text": text}),
        host_id=str(spec.get("host_id") or ""),
        stage_first_shard=False,
        work_dir=task_dir,
    )
    nested_dir = executor._tmpdir
    executor._tmpdir = task_dir
    executor._model_path = spec["model_path"]
    executor._bundle_root = spec.get("bundle_root")
    executor._image_processor_config = spec.get("image_processor_config")
    executor._dataset_shards = [{} for _ in range(int(spec.get("num_shards", 0)))]
    executor._params = spec.get("params") or {}
    executor._batch_size = int(spec.get("batch_size", 32))
    executor._steps = int(spec.get("steps", 100))
    executor._round_num = int(spec.get("round_num", 0))
    executor._total_rounds = int(spec.get("total_rounds", 1))
    executor._step_offset = int(spec.get("step_offset", 0))
    executor._mode = str(spec.get("mode") or "training")
    executor._code = str(spec.get("code") or "")
    executor._stop_event = threading.Event()
    executor.get_shard = _request_shard  # type: ignore[method-assign]

    sys.stdout = _LogStream("INFO")
    sys.stderr = _LogStream("ERROR")
    try:
        executor._execute()
        result_path = str(Path(task_dir, "runner-result.pt"))
        if executor.modified_state is not None:
            torch.save(executor.modified_state, result_path)
        _send(
            {
                "type": "complete",
                "training_time": executor.training_time,
                "report": executor.report,
                "error": executor.execution_error,
                "result_path": result_path if executor.modified_state is not None else "",
            }
        )
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            Path(nested_dir).rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
