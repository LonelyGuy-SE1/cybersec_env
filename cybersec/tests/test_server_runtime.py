from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def test_server_app_imports_in_top_level_mode() -> None:
    module = importlib.import_module("server.app")
    assert hasattr(module, "app")


def test_server_health_schema_and_reset_endpoints() -> None:
    module = importlib.import_module("server.app")
    client = TestClient(module.app)

    health = client.get("/health")
    assert health.status_code == 200

    schema = client.get("/schema")
    assert schema.status_code == 200

    reset = client.post(
        "/reset",
        json={"seed": 123, "scenario_id": "supply_chain_token_drift"},
    )
    assert reset.status_code == 200
    body = reset.json()
    assert "observation" in body
    assert "done" in body


def test_env_var_bounds_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("server.app")

    monkeypatch.setenv("CYBERSEC_MAX_CONCURRENT_ENVS", "0")
    with pytest.raises(RuntimeError, match="CYBERSEC_MAX_CONCURRENT_ENVS"):
        module._max_concurrent_envs()

    monkeypatch.setenv("CYBERSEC_FALSE_POSITIVE_RATE", "1.2")
    with pytest.raises(RuntimeError, match="CYBERSEC_FALSE_POSITIVE_RATE"):
        module._env_factory()

    monkeypatch.setenv("CYBERSEC_FALSE_POSITIVE_RATE", "0.2")
    monkeypatch.setenv("CYBERSEC_HORIZON", "0")
    with pytest.raises(RuntimeError, match="CYBERSEC_HORIZON"):
        module._env_factory()
