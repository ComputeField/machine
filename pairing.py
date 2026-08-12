# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Two-sided account pairing for ComputeField Machine."""

import time
import uuid
from urllib.parse import urlsplit

import requests
from capabilities import default_machine_name, get_capabilities
from config import settings


def _api_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        parsed.scheme not in ({"http", "https"} if local else {"https"})
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("API URL must be HTTPS (HTTP is allowed only for localhost)")
    return value.strip().rstrip("/")


def _broker_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        parsed.scheme not in ({"ws", "wss"} if local else {"wss"})
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Broker URL must be WSS (WS is allowed only for localhost)")
    return value.strip()


def _post(api_url: str, path: str, body: dict) -> dict:
    response = requests.post(
        f"{api_url.rstrip('/')}{path}",
        json=body,
        timeout=20,
        allow_redirects=False,
        headers={"User-Agent": f"computefield-machine/{settings.client_version}"},
    )
    if not response.ok:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"Pairing failed ({response.status_code}): {message}")
    return response.json()


def pair(api_url: str, code: str, *, name: str = "", wait: bool = True) -> dict:
    """Claim a browser-created code and wait for owner confirmation."""
    api_url = _api_origin(api_url)
    identity = settings.saved_identity
    host_id = str(identity.get("host_id") or uuid.uuid4())
    settings.update_identity(host_id=host_id, api_url=api_url.rstrip("/"))
    claim = _post(
        api_url,
        "/api/v1/machinery/pairings/claim",
        {
            "code": code,
            "host_id": host_id,
            "name": name or settings.machine_name or default_machine_name(get_capabilities(settings.compute_mode)),
            "client_version": settings.client_version,
        },
    )
    settings.update_identity(
        host_id=host_id,
        api_url=api_url.rstrip("/"),
        pairing_id=str(claim["pairing_id"]),
        claim_secret=str(claim["claim_secret"]),
    )
    print(f"Machine fingerprint: {claim['fingerprint']}", flush=True)
    print("Confirm the same fingerprint on the Machines page.", flush=True)
    if not wait:
        return claim
    return finish_pairing()


def finish_pairing() -> dict:
    identity = settings.saved_identity
    required = ("api_url", "pairing_id", "claim_secret", "host_id")
    if any(not identity.get(key) for key in required):
        raise RuntimeError(f"Run '{settings.cli_name} pair CODE' first")
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        status = _post(
            str(identity["api_url"]),
            "/api/v1/machinery/pairings/status",
            {
                "pairing_id": identity["pairing_id"],
                "claim_secret": identity["claim_secret"],
            },
        )
        if status.get("status") == "completed":
            settings.save_identity(
                str(status["host_id"]),
                str(status["credential"]),
                _broker_url(str(status["broker_url"])),
            )
            print("Machine connected to the account.", flush=True)
            return status
        time.sleep(2)
    raise TimeoutError("Pairing expired before it was confirmed")
