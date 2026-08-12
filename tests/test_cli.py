# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import sys
from types import SimpleNamespace

import pytest

import cli
from config import Settings


def test_sharing_enable_requires_os_isolation(monkeypatch):
    identity = SimpleNamespace(
        cli_name="computefield-machine",
        sharing_enabled=False,
        sharing_supported=False,
        machine_isolation_mode="none",
        update_identity=lambda **_values: None,
    )
    monkeypatch.setattr(cli, "settings", identity)
    monkeypatch.setattr(sys, "argv", ["computefield-machine", "sharing", "enable"])

    with pytest.raises(SystemExit, match="requires the packaged per-workload sandbox"):
        cli.main()


def test_sharing_enable_persists_in_isolated_service(monkeypatch, capsys, tmp_path):
    saved = {}
    identity = SimpleNamespace(
        cli_name="computefield-machine",
        sharing_enabled=False,
        sharing_supported=True,
        machine_isolation_mode="sandbox",
        work_dir=str(tmp_path / "work"),
        update_identity=lambda **values: saved.update(values),
    )
    monkeypatch.setattr(cli, "settings", identity)
    monkeypatch.setattr(cli, "sandbox_self_test", lambda _path: "test-sandbox")
    monkeypatch.setattr(
        sys,
        "argv",
        ["computefield-machine", "sharing", "enable"],
    )

    cli.main()

    assert saved == {"sharing_enabled": True}
    assert "Restart computefield-machine" in capsys.readouterr().out


def test_saved_shared_flag_is_inactive_without_isolation(tmp_path):
    identity = tmp_path / "identity.json"
    identity.write_text('{"sharing_enabled":true}', encoding="utf-8")

    unisolated = Settings(machine_identity_file=str(identity), machine_isolation_mode="none")
    isolated = Settings(machine_identity_file=str(identity), machine_isolation_mode="sandbox")

    assert unisolated.sharing_enabled is False
    assert isolated.sharing_enabled is True


def test_cpu_profile_uses_the_cpu_command_name():
    assert Settings(machine_compute_mode="cpu").cli_name == "computefield-machine-cpu"
    assert Settings(machine_compute_mode="auto").cli_name == "computefield-machine"


def test_pair_prompt_consent(monkeypatch):
    args = SimpleNamespace(share=False, private=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    assert cli._pair_sharing_consent(args) is True


def test_noninteractive_pair_stays_private_without_explicit_flag(monkeypatch):
    args = SimpleNamespace(share=False, private=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    assert cli._pair_sharing_consent(args) is False
