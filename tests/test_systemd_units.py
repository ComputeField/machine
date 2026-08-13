# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    (
        "computefield-machine.service",
        "computefield-machine-cpu.service",
        "computefield-machine-verify.service",
        "computefield-machine-cpu-verify.service",
    ),
)
def test_service_allows_bubblewrap_to_construct_the_workload_namespaces(name):
    unit = (Path(__file__).parents[1] / "packaging" / "systemd" / name).read_text(encoding="utf-8")

    assert "ProtectProc=" not in unit
    assert "ProcSubset=pid" not in unit
    assert "ProtectKernelLogs=" not in unit
    assert "ProtectKernelTunables=" not in unit
    assert "ProtectHostname=" not in unit
    assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK" in unit
    assert "computefield-machine doctor" in unit


@pytest.mark.parametrize(
    "name",
    ("computefield-machine.service", "computefield-machine-cpu.service"),
)
def test_long_running_service_has_bounded_failure_restarts(name):
    unit = (Path(__file__).parents[1] / "packaging" / "systemd" / name).read_text(encoding="utf-8")

    assert "ExecStartPre=" in unit
    assert "Restart=on-failure" in unit
    assert "StartLimitIntervalSec=300" in unit
    assert "StartLimitBurst=5" in unit


@pytest.mark.parametrize(
    ("runtime_name", "verification_name"),
    (
        ("computefield-machine.service", "computefield-machine-verify.service"),
        ("computefield-machine-cpu.service", "computefield-machine-cpu-verify.service"),
    ),
)
def test_installation_verification_uses_the_runtime_security_boundary(runtime_name, verification_name):
    root = Path(__file__).parents[1] / "packaging" / "systemd"
    runtime = root.joinpath(runtime_name).read_text(encoding="utf-8")
    verification = root.joinpath(verification_name).read_text(encoding="utf-8")
    security_keys = {
        "NoNewPrivileges",
        "PrivateTmp",
        "ProtectSystem",
        "ProtectHome",
        "ReadWritePaths",
        "RestrictSUIDSGID",
        "LockPersonality",
        "PrivateDevices",
        "PrivateMounts",
        "ProtectClock",
        "ProtectControlGroups",
        "ProtectKernelModules",
        "RestrictRealtime",
        "RestrictAddressFamilies",
        "SystemCallFilter",
        "SystemCallErrorNumber",
        "SystemCallArchitectures",
        "UMask",
    }

    def boundary(unit: str) -> set[str]:
        return {
            line
            for line in unit.splitlines()
            if line.partition("=")[0] in security_keys
        }

    assert boundary(verification) == boundary(runtime)


def test_package_verifies_candidate_before_activation():
    script = (Path(__file__).parents[1] / "packaging" / "build-deb.sh").read_text(encoding="utf-8")

    assert script.index('systemctl start "$verify_service_name"') < script.index('mv -Tf "$link" "$current"')
