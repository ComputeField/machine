# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/capabilities.py"""

from unittest.mock import MagicMock, patch

import pytest

from capabilities import default_machine_name, get_capabilities


class TestGetCapabilities:
    def test_forced_cpu_does_not_probe_accelerators(self):
        with patch("capabilities.torch") as mock_torch:
            caps = get_capabilities("cpu")

        assert caps == {"compute_mode": "cpu"}
        mock_torch.cuda.is_available.assert_not_called()

    def test_rejects_unknown_compute_mode(self):
        with patch("capabilities.torch"):
            with pytest.raises(ValueError, match="compute_mode"):
                get_capabilities("quantum")

    def test_cpu_only_when_no_accelerator(self):
        with patch("capabilities.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            caps = get_capabilities()
        assert caps["compute_mode"] == "cpu"

    def test_reports_mps_on_apple_silicon(self):
        """Previously fell through to "cpu" — MPS wasn't checked at all, so an
        Apple Silicon Machine was indistinguishable from a plain CPU host."""
        with patch("capabilities.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            caps = get_capabilities()
        assert caps["compute_mode"] == "mps"

    def test_cuda_takes_priority_over_mps(self):
        with patch("capabilities.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 2
            mock_torch.version.cuda = "12.4"
            mock_torch.backends.mps.is_available.return_value = True
            caps = get_capabilities()
        assert caps["compute_mode"] == "cuda"
        assert caps["gpu_count"] == 2
        assert caps["cuda_version"] == "12.4"

    def test_pynvml_queried_at_the_cuda_visible_devices_physical_index(self, monkeypatch):
        """Regression: this process may be one of several Machine instances
        on a multi-GPU machine, each pinned via CUDA_VISIBLE_DEVICES — NVML
        (unlike torch.cuda) doesn't reliably remap its own indices for that,
        so a hardcoded nvmlDeviceGetHandleByIndex(0) can report the wrong
        physical GPU's model/VRAM. See gpu_device.nvml_device_index."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(total=24 * 1_048_576)

        with patch("capabilities.torch") as mock_torch, patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1  # this process's own view (CUDA-runtime-remapped)
            mock_torch.version.cuda = "12.4"
            caps = get_capabilities()

        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)
        assert caps["gpu_model"] == "NVIDIA RTX 4090"
        assert caps["vram_mb"] == 24


def test_default_name_includes_hardware_and_local_host(monkeypatch):
    monkeypatch.setattr("capabilities.socket.gethostname", lambda: "Studio-Mac.local")

    assert default_machine_name({"compute_mode": "mps"}) == "Apple GPU · Studio-Mac"
