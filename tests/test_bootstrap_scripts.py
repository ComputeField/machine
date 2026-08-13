# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ("bootstrap-ubuntu.sh", "bootstrap-ubuntu-cpu.sh"))
def test_apt_can_read_only_the_verified_public_package(name):
    script = (Path(__file__).parents[1] / "packaging" / name).read_text(encoding="utf-8")

    checksum = script.index("sha256sum --check")
    readable = script.index('chmod 0644 "$package"')
    install = script.index('apt-get install -y "$package"')
    assert 'download_dir="$(mktemp -d)"' in script
    assert 'chmod 0755 "$download_dir"' in script
    assert checksum < readable < install
    assert "trap 'rm -rf \"$download_dir\"' EXIT" in script
