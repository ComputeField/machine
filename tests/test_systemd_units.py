# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    ("computefield-machine.service", "computefield-machine-cpu.service"),
)
def test_service_keeps_kernel_proc_metadata_visible_to_bubblewrap(name):
    unit = (Path(__file__).parents[1] / "packaging" / "systemd" / name).read_text(encoding="utf-8")

    assert "ProtectProc=invisible" in unit
    assert "ProtectKernelTunables=true" in unit
    assert "ProcSubset=pid" not in unit
    assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK" in unit
    assert "ExecStartPre=" in unit
    assert "computefield-machine doctor" in unit
