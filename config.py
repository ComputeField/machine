# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_FILE = Path(__file__).resolve().with_name(".env")


class Settings(BaseSettings):
    # Resolve development settings beside this module. A relative `.env`
    # would depend on the caller's working directory and can be unreadable
    # when the packaged CLI drops privileges during installation.
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")
    machine_broker_url: str = "wss://computefield.net/ws/machine"
    computefield_api_url: str = "https://computefield.net"
    machine_name: str = ""
    # Packaged CPU services force CPU even when the same server also exposes
    # an NVIDIA device to a separate Machine instance.
    machine_compute_mode: str = "auto"
    # Selects one server-configured object-storage route. Ordinary installed
    # Machines stay on "external"; the Docker dev worker uses "internal".
    machine_transfer_route: str = "external"
    client_version: str = "0.1.9"
    allow_foreign_workloads: bool = False
    machine_isolation_mode: str = "none"
    hardware_stats_interval: int = 5
    machine_id: str = ""
    machine_identity_file: str = "/var/lib/computefield-machine/identity.json"
    machine_work_dir: str = ""
    # Disk cap (MB) for the raw-shard LRU cache + next-shard prefetch —
    # avoids re-downloading each shard once per epoch. 0 disables both.
    shard_cache_max_mb: int = 4096
    # Payload size (MB) for the post-registration upload/download speed
    # test against gpu-manager (see speedtest.py). 0 disables the test.
    speedtest_payload_mb: int = 8
    require_auth: bool = True

    @property
    def host_id(self) -> str:
        return str(self.saved_identity.get("host_id") or self.machine_id)

    @property
    def manager_url(self) -> str:
        return str(self.saved_identity.get("broker_url") or self.machine_broker_url)

    @property
    def sharing_enabled(self) -> bool:
        saved = self.saved_identity.get("sharing_enabled")
        requested = saved if isinstance(saved, bool) else self.allow_foreign_workloads
        return bool(requested and self.sharing_supported)

    @property
    def sharing_supported(self) -> bool:
        """Whether this installation includes the per-workload OS sandbox."""
        return self.machine_isolation_mode == "sandbox"

    @property
    def saved_identity(self) -> dict:
        try:
            data = json.loads(Path(self.machine_identity_file).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @property
    def saved_credential(self) -> str:
        return str(self.saved_identity.get("credential") or "")

    @property
    def work_dir(self) -> str:
        if self.machine_work_dir:
            return self.machine_work_dir
        return str(Path(self.machine_identity_file).parent / "work")

    @property
    def compute_mode(self) -> str:
        mode = self.machine_compute_mode.strip().lower()
        if mode not in {"auto", "cpu"}:
            raise ValueError("MACHINE_COMPUTE_MODE must be 'auto' or 'cpu'")
        return mode

    @property
    def cli_name(self) -> str:
        return "computefield-machine-cpu" if self.compute_mode == "cpu" else "computefield-machine"

    def update_identity(self, **values: object) -> None:
        data = self.saved_identity
        data.update(values)
        self._write_identity(data)

    def _write_identity(self, data: dict) -> None:
        path = Path(self.machine_identity_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def save_identity(self, host_id: str, credential: str, broker_url: str = "") -> None:
        # Replace rather than merge: short-lived pairing claims must not
        # survive once the durable host credential has been issued.
        values = {"host_id": host_id, "credential": credential}
        if broker_url:
            values["broker_url"] = broker_url
        sharing = self.saved_identity.get("sharing_enabled")
        if isinstance(sharing, bool):
            values["sharing_enabled"] = sharing
        self._write_identity(values)

    def clear_identity(self) -> None:
        try:
            Path(self.machine_identity_file).unlink()
        except FileNotFoundError:
            pass


settings = Settings()
