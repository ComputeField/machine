# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "packaging" / "macos" / "computefield-machine"


def _fixture(tmp_path: Path, *, cli_exit: int = 0, loaded: bool = False) -> tuple[dict[str, str], Path, Path]:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    machine_bin = home / "Library" / "Application Support" / "ComputeField Machine" / "venv" / "bin"
    machine_bin.mkdir(parents=True)
    bin_dir.mkdir()

    cli_log = tmp_path / "cli.log"
    launchctl_log = tmp_path / "launchctl.log"
    cli = machine_bin / "computefield-machine"
    cli.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "$CLI_LOG"\nexit {cli_exit}\n',
        encoding="utf-8",
    )
    cli.chmod(0o755)
    launchctl = bin_dir / "launchctl"
    launchctl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$LAUNCHCTL_LOG"\n'
        f'if [ "$1" = print ]; then exit {0 if loaded else 1}; fi\n',
        encoding="utf-8",
    )
    launchctl.chmod(0o755)

    env = os.environ.copy()
    env.update(
        HOME=str(home),
        PATH=f"{bin_dir}:{env['PATH']}",
        CLI_LOG=str(cli_log),
        LAUNCHCTL_LOG=str(launchctl_log),
    )
    return env, cli_log, launchctl_log


def test_pair_starts_launchd_agent_automatically(tmp_path):
    env, cli_log, launchctl_log = _fixture(tmp_path)

    result = subprocess.run(  # noqa: S603
        [str(WRAPPER), "pair", "ABCD-EF12-3456", "--private"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert cli_log.read_text(encoding="utf-8").strip() == "pair ABCD-EF12-3456 --private"
    calls = launchctl_log.read_text(encoding="utf-8")
    assert "print gui/" in calls
    assert "bootstrap gui/" in calls
    assert "ComputeField Machine is running." in result.stdout


def test_failed_pair_does_not_start_launchd_agent(tmp_path):
    env, _, launchctl_log = _fixture(tmp_path, cli_exit=23)

    result = subprocess.run(  # noqa: S603
        [str(WRAPPER), "pair", "INVALID"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not launchctl_log.exists()


def test_unpair_stops_loaded_launchd_agent(tmp_path):
    env, _, launchctl_log = _fixture(tmp_path, loaded=True)

    result = subprocess.run(  # noqa: S603
        [str(WRAPPER), "unpair", "--yes"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "bootout gui/" in launchctl_log.read_text(encoding="utf-8")
