# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "machine_workspace",
    ROOT / "workspace.py",
)
workspace = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(workspace)


def test_work_root_removes_every_crash_artifact_from_its_dedicated_directory(tmp_path):
    root = tmp_path / "machine-work"
    stale = root / "task-stale"
    stale.mkdir(parents=True)
    (stale / "model.pt").write_bytes(b"old")
    unexpected = root / "unexpected-artifact"
    unexpected.write_text("stale")

    owner = workspace.WorkRoot(str(root))
    try:
        assert not stale.exists()
        assert not unexpected.exists()
        assert owner.path == root
    finally:
        owner.close()


def test_work_root_is_exclusive(tmp_path):
    owner = workspace.WorkRoot(str(tmp_path / "machine-work"))
    try:
        with pytest.raises(RuntimeError, match="another Machine process"):
            workspace.WorkRoot(str(tmp_path / "machine-work"))
    finally:
        owner.close()


def test_work_root_unlinks_symlinks_without_touching_their_targets(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("safe")
    root = tmp_path / "machine-work"
    root.mkdir()
    (root / "unexpected-link").symlink_to(outside, target_is_directory=True)

    owner = workspace.WorkRoot(str(root))
    try:
        assert not (root / "unexpected-link").exists()
        assert marker.read_text() == "safe"
    finally:
        owner.close()
