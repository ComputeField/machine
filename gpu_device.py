# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Which physical GPU NVML calls should target on this process.

capabilities.py and monitor.py both call pynvml.nvmlDeviceGetHandleByIndex —
historically always with a hardcoded 0. That's wrong once multiple Machine
processes run on one multi-GPU machine, one per GPU, each pinned via
CUDA_VISIBLE_DEVICES=N (the standard way to run several instances on a
single box — see CUDA_VISIBLE_DEVICES in .env.example). NVML
does not reliably remap its own device indices based on CUDA_VISIBLE_DEVICES
the way the CUDA runtime does for torch.cuda calls — driver/version
dependent — so a hardcoded 0 can report the wrong physical GPU's
temperature/utilization/VRAM even though training itself correctly targets
the right device via the CUDA runtime.
"""

import os


def nvml_device_index(device_count: int) -> int:
    """Physical GPU index for an NVML call, given how many devices NVML
    itself currently enumerates (pynvml.nvmlDeviceGetCount()).

    Handles both possible driver behaviors defensively:
    - device_count <= 1: NVML already sees only this process's GPU (some
      driver/NVML versions DO filter by CUDA_VISIBLE_DEVICES) — index 0 is
      already correct; treating CUDA_VISIBLE_DEVICES as a physical index
      here would be wrong (out of range or double-remapped).
    - device_count > 1: NVML enumerates all physical GPUs regardless of
      CUDA_VISIBLE_DEVICES (the more common case) — resolve the physical
      index this process is actually bound to from CUDA_VISIBLE_DEVICES.
      Only ever a single bare digit here: ComputeField Machine launcher's
      --cuda-device sets it that way itself, never a comma-list or UUID.
    - Unset, unparsable, or out of range for device_count: 0 (today's
      single-GPU-machine default — never crashes on a stale/misconfigured
      value).
    """
    if device_count <= 1:
        return 0
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = raw.split(",")[0].strip()
    if first.isdigit() and int(first) < device_count:
        return int(first)
    return 0
