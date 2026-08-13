# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import platform
import stat
import sys
from types import SimpleNamespace

import pytest

import sandbox_runtime


def test_child_environment_does_not_inherit_platform_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MACHINE_CREDENTIAL", "secret")
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    environment = sandbox_runtime._clean_environment(str(tmp_path))

    assert "MACHINE_CREDENTIAL" not in environment
    assert "API_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["HOME"].startswith(str(tmp_path))
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_linux_policy_has_separate_namespaces_and_task_only_write(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("", encoding="utf-8")
    bwrap.chmod(0o755)
    monkeypatch.setattr(sandbox_runtime.shutil, "which", lambda name: str(bwrap) if name == "bwrap" else None)

    command = sandbox_runtime.sandbox_command([sys.executable, "-c", "pass"], str(tmp_path))

    assert "--unshare-pid" in command
    assert "--unshare-net" in command
    assert "--unshare-user" in command
    assert "--disable-userns" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    bind_index = command.index("--bind")
    assert command[bind_index + 1 : bind_index + 3] == [str(tmp_path), str(tmp_path)]


def test_linux_policy_rejects_setuid_bubblewrap(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("", encoding="utf-8")
    real_stat = sandbox_runtime.os.stat
    monkeypatch.setattr(
        sandbox_runtime.os,
        "stat",
        lambda path, *args, **kwargs: (
            SimpleNamespace(st_mode=0o755 | stat.S_ISUID)
            if str(path) == str(bwrap)
            else real_stat(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(sandbox_runtime.shutil, "which", lambda name: str(bwrap) if name == "bwrap" else None)

    with pytest.raises(sandbox_runtime.SandboxUnavailable, match="setuid"):
        sandbox_runtime.sandbox_command([sys.executable, "-c", "pass"], str(tmp_path))


def test_configured_bubblewrap_must_be_root_owned_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("", encoding="utf-8")
    bwrap.chmod(0o755)
    monkeypatch.setenv("COMPUTEFIELD_BWRAP", str(bwrap))

    with pytest.raises(sandbox_runtime.SandboxUnavailable, match="root-owned"):
        sandbox_runtime.sandbox_command([sys.executable, "-c", "pass"], str(tmp_path))


def test_configured_root_owned_bubblewrap_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    bwrap = tmp_path / "private-bwrap"
    bwrap.write_text("", encoding="utf-8")
    real_stat = sandbox_runtime.os.stat
    monkeypatch.setattr(
        sandbox_runtime.os,
        "stat",
        lambda path, *args, **kwargs: (
            SimpleNamespace(st_mode=0o755, st_uid=0)
            if str(path) == str(bwrap)
            else real_stat(path, *args, **kwargs)
        ),
    )
    monkeypatch.setenv("COMPUTEFIELD_BWRAP", str(bwrap))

    command = sandbox_runtime.sandbox_command([sys.executable, "-c", "pass"], str(tmp_path))

    assert command[0] == str(bwrap)


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only policy")
def test_macos_policy_denies_network_identity_tree_and_external_writes(tmp_path):
    task = tmp_path / "work" / "task"
    task.mkdir(parents=True)

    profile = sandbox_runtime._macos_profile(str(task), [sys.executable, "-c", "pass"])

    assert "(deny network*)" in profile
    assert "(deny file-write*" in profile
    assert "(deny process-fork)" in profile
    assert "(deny process-exec" in profile
    assert str(tmp_path) in profile
    assert str(task) in profile
