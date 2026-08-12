# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import pytest

import pairing
from config import Settings
from pairing import _api_origin, _broker_url


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        ("https://compute.example.com/", "https://compute.example.com"),
        ("http://localhost:8080", "http://localhost:8080"),
    ],
)
def test_api_origin_accepts_secure_production_and_local_development(value, normalized):
    assert _api_origin(value) == normalized


@pytest.mark.parametrize(
    "value",
    [
        "http://compute.example.com",
        "https://compute.example.com/unexpected-prefix",
        "https://user:secret@compute.example.com",
    ],
)
def test_api_origin_rejects_downgrades_and_ambiguous_bases(value):
    with pytest.raises(ValueError):
        _api_origin(value)


def test_broker_requires_wss_outside_local_development():
    assert _broker_url("wss://compute.example.com/ws/machine") == "wss://compute.example.com/ws/machine"
    assert _broker_url("ws://localhost:8080/ws/machine") == "ws://localhost:8080/ws/machine"
    with pytest.raises(ValueError):
        _broker_url("ws://compute.example.com/ws/machine")


def test_cpu_pairing_uses_cpu_capabilities_for_the_default_name(monkeypatch, tmp_path):
    identity = tmp_path / "identity.json"
    settings = Settings(
        machine_compute_mode="cpu",
        machine_identity_file=str(identity),
    )
    captured = {}
    monkeypatch.setattr(pairing, "settings", settings)
    monkeypatch.setattr(pairing, "get_capabilities", lambda mode: {"compute_mode": mode})
    monkeypatch.setattr(pairing, "default_machine_name", lambda caps: f"{caps['compute_mode'].upper()} Machine")
    monkeypatch.setattr(
        pairing,
        "_post",
        lambda _api, _path, body: captured.update(body)
        or {"pairing_id": "pair", "claim_secret": "secret", "fingerprint": "ABCD"},
    )

    pairing.pair("https://computefield.net", "CODE", wait=False)

    assert captured["name"] == "CPU Machine"
