# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Unit tests for Machine/gpu_device.py"""

from gpu_device import nvml_device_index


class TestNvmlDeviceIndex:
    def test_single_device_ignores_cuda_visible_devices(self, monkeypatch):
        """Some driver/NVML versions already filter enumeration down to just
        the visible device — treating CUDA_VISIBLE_DEVICES as a physical
        index in that case would be wrong (out of range or double-remapped),
        so a count of 1 always means "use index 0, it's already correct"."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert nvml_device_index(device_count=1) == 0

    def test_multi_device_resolves_the_bound_physical_index(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert nvml_device_index(device_count=2) == 1

    def test_unset_defaults_to_zero(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert nvml_device_index(device_count=4) == 0

    def test_garbage_value_defaults_to_zero(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-number")
        assert nvml_device_index(device_count=4) == 0

    def test_out_of_range_index_defaults_to_zero_instead_of_crashing(self, monkeypatch):
        """A stale/misconfigured CUDA_VISIBLE_DEVICES must not make an NVML
        call with an index NVML doesn't actually have."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
        assert nvml_device_index(device_count=2) == 0

    def test_zero_device_count_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert nvml_device_index(device_count=0) == 0
