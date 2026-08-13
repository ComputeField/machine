# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import os
from pathlib import Path

from config import Settings


def test_settings_do_not_resolve_dotenv_from_the_callers_directory(tmp_path):
    inaccessible = tmp_path / "inaccessible"
    inaccessible.mkdir()
    previous = Path.cwd()
    os.chdir(inaccessible)
    inaccessible.chmod(0)
    try:
        settings = Settings()
    finally:
        inaccessible.chmod(0o700)
        os.chdir(previous)

    assert settings.computefield_api_url == "https://computefield.net"
    assert Path(Settings.model_config["env_file"]).is_absolute()
