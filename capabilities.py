# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Detect and report hardware capabilities at registration."""

import logging
import re
import socket

import torch
from gpu_device import nvml_device_index

logger = logging.getLogger(__name__)


def default_machine_name(caps: dict | None = None) -> str:
    """Produce a useful hardware-first name without exposing an internal ID."""
    capabilities = caps or get_capabilities()
    mode = str(capabilities.get("compute_mode") or "cpu").lower()
    hardware = str(capabilities.get("gpu_model") or "").strip()
    if not hardware:
        hardware = {"mps": "Apple GPU", "cuda": "CUDA GPU"}.get(mode, "CPU")
    hostname = socket.gethostname().split(".", 1)[0].strip()
    internal = re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27}", hostname, re.IGNORECASE)
    if hostname and hostname.lower() != "localhost" and not internal:
        return f"{hardware} · {hostname}"
    return f"{hardware} Machine"


def get_capabilities() -> dict:
    """Return a capabilities dict for the registration message.

    Always present:
        compute_mode  "cuda" | "mps" | "cpu"

    Present when CUDA is available:
        gpu_count     int
        gpu_model     str   (first GPU name)
        vram_mb       int   (total VRAM of first GPU)
        cuda_version  str   (e.g. "12.1")
    """
    caps: dict = {"compute_mode": "cpu"}

    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            caps["compute_mode"] = "mps"
            logger.info("Capabilities: MPS (Apple GPU)")
        else:
            logger.info("Capabilities: CPU-only (no CUDA/MPS)")
        return caps

    caps["compute_mode"] = "cuda"
    caps["gpu_count"] = torch.cuda.device_count()
    caps["cuda_version"] = torch.version.cuda or ""

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index(count))
        name = pynvml.nvmlDeviceGetName(handle)
        caps["gpu_model"] = name if isinstance(name, str) else name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        caps["vram_mb"] = int(mem.total / 1_048_576)
    except Exception as exc:
        logger.debug("pynvml unavailable for capability reporting: %s", exc)

    logger.info(
        "Capabilities: %s  gpus=%s  vram=%s MB  cuda=%s",
        caps["compute_mode"],
        caps.get("gpu_count", "?"),
        caps.get("vram_mb", "?"),
        caps.get("cuda_version", "?"),
    )
    return caps
