# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Background hardware monitor — polls GPU (pynvml/MPS) and CPU/RAM (psutil) on a fixed interval."""

import logging
import threading
from collections.abc import Callable

import psutil
from gpu_device import nvml_device_index

logger = logging.getLogger(__name__)


def start_monitor(interval: int, callback: Callable[[dict], None]) -> threading.Event:
    """
    Starts a daemon thread that calls callback(stats_dict) every `interval` seconds.
    Returns a threading.Event — set it to stop the monitor.
    stats_dict always contains cpu_util, ram_used_mb, and ram_total_mb.
    GPU fields added when available:
      NVIDIA (pynvml): gpu_util, gpu_mem_used_mb, gpu_mem_total_mb, gpu_temp_c
      Apple MPS:       gpu_mem_used_mb  (unified memory allocated by MPS)
    """
    stop_event = threading.Event()

    # Try NVIDIA pynvml first, then Apple MPS
    _nvml_handle = None
    _use_mps = False

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _nvml_count = pynvml.nvmlDeviceGetCount()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index(_nvml_count))
        logger.info("Hardware monitor: NVIDIA GPU detected via pynvml")
    except Exception:
        try:
            import torch

            if torch.backends.mps.is_available():
                _use_mps = True
                logger.info("Hardware monitor: Apple MPS detected — reporting MPS memory")
            else:
                logger.info("Hardware monitor: no GPU / pynvml unavailable — CPU-only metrics")
        except Exception:
            logger.info("Hardware monitor: no GPU / pynvml unavailable — CPU-only metrics")

    def _run() -> None:
        while not stop_event.wait(interval):
            vm = psutil.virtual_memory()
            stats: dict = {
                "cpu_util": psutil.cpu_percent(interval=None),
                "ram_used_mb": int(vm.used / 1_048_576),
                "ram_total_mb": int(vm.total / 1_048_576),
            }

            if _nvml_handle is not None:
                try:
                    import pynvml  # type: ignore

                    util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    stats["gpu_util"] = util.gpu
                    stats["gpu_mem_used_mb"] = int(mem.used / 1_048_576)
                    stats["gpu_mem_total_mb"] = int(mem.total / 1_048_576)
                    stats["gpu_temp_c"] = temp
                except Exception as exc:
                    logger.warning("pynvml read error: %s", exc)

            elif _use_mps:
                try:
                    import torch

                    stats["gpu_mem_used_mb"] = int(torch.mps.current_allocated_memory() / 1_048_576)
                except Exception as exc:
                    logger.warning("MPS memory read error: %s", exc)

            callback(stats)

    thread = threading.Thread(target=_run, daemon=True, name="hw-monitor")
    thread.start()
    return stop_event
