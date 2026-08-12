# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Exclusive, crash-cleaned storage for tenant-bound Machine artifacts."""

import fcntl
import os
import shutil
from pathlib import Path
from typing import BinaryIO


class WorkRoot:
    """Own one Machine work root for the daemon's complete lifetime."""

    def __init__(self, path: str) -> None:
        root = Path(path).expanduser().resolve()
        if root == Path(root.anchor) or root == Path.home():
            raise ValueError("MACHINE_WORK_DIR must be a dedicated subdirectory")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        self.path = root
        self._lock: BinaryIO = (root / ".machine.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise RuntimeError(f"another Machine process owns {root}") from exc

        self._clean_children()

    def _clean_children(self) -> None:
        for child in self.path.iterdir():
            if child.name == ".machine.lock":
                continue
            if child.is_symlink() or not child.is_dir():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child)

    def close(self) -> None:
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
        self._lock.close()
