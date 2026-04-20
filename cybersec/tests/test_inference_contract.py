from __future__ import annotations

import json
from pathlib import Path

import pytest

import inference
from inference import _extract_last_action_error, _safe_json_parse, _severity_rank
from models import CybersecAction, CybersecObservation


def _obs(info: dict) -> CybersecObservation:
    return CybersecObservation(
        scenario_id="supply_chain_token_drift",
        scenario_title="Supply Chain",
        scenario_objective="Contain attacker",
        tick=1,
        horizon=32,
        enterprise_risk_score=0.2,
        alerts=[],
        open_tickets=[],
        ticket_updates=[],
        forensics_updates=[],
        known_compromised_assets=[],
        known_compromised_identities=[],
        recent_activity=[],
        available_actions=["MONITOR"],
        valid_targets={},
        info=info,
        done=False,
        reward=0.0,
    )


def test_extract_last_action_error_prefers_last_action_error() -> None:
    observation = _obs({"last_action_error": "bad_action", "invalid_action": "other"})
    assert _extract_last_action_error(observation) == "bad_action"


def test_extract_last_action_error_falls_back_to_invalid_action() -> None:
    observation = _obs({"invalid_action": "bad_action"})
    assert _extract_last_action_error(observation) == "bad_action"


def test_safe_json_parse_handles_wrapped_json() -> None:
    value = _safe_json_parse('prefix {"action_type": "MONITOR"} suffix')
    assert isinstance(value, dict)
    assert value["action_type"] == "MONITOR"


def test_severity_rank_orders_expected_values() -> None:
    assert _severity_rank("critical") > _severity_rank("high")
    assert _severity_rank("high") > _severity_rank("medium")
    assert _severity_rank("medium") > _severity_rank("low")


def test_enforce_action_contract_rejects_unknown_action() -> None:
    observation = _obs(
        {
            "available_actions": ["MONITOR"],
            "step_contract": "observation_reward_done_info",
        }
    )
    action = CybersecAction(action_type="QUERY_LOGS", parameter="cloud")
    assert inference._enforce_action_contract(observation, action) is None


def test_enforce_action_contract_accepts_valid_query_logs() -> None:
    observation = _obs(
        {
            "available_actions": ["QUERY_LOGS"],
            "step_contract": "observation_reward_done_info",
        }
    )
    observation.valid_targets = {"query_log_sources": ["cloud", "network"]}
    action = CybersecAction(action_type="QUERY_LOGS", parameter="cloud")
    constrained = inference._enforce_action_contract(observation, action)
    assert constrained is not None
    assert constrained.parameter == "cloud"


def test_build_llm_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN or API_KEY"):
        inference._build_llm_client()


def test_inference_main_local_random_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "inference_results.json"
    monkeypatch.setenv("INFERENCE_ENV", "local")
    monkeypatch.setenv("INFERENCE_MODE", "random")
    monkeypatch.setenv("CYBERSEC_SCENARIOS", "supply_chain_token_drift")
    monkeypatch.setenv("CYBERSEC_SEEDS", "101")
    monkeypatch.setenv("MAX_STEPS", "8")
    monkeypatch.setenv("INFERENCE_OUTPUT", str(output_path))

    inference.main()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == inference.BENCHMARK
    assert payload["env"] == "local"
    assert payload["mode"] == "random"
    assert len(payload["episodes"]) == 1


def test_enforce_action_contract_rejects_duplicate_pending_forensics_target() -> None:
    observation = _obs(
        {
            "available_actions": ["REQUEST_FORENSICS"],
            "step_contract": "observation_reward_done_info",
        }
    )
    observation.valid_targets = {
        "assets": ["ws-dev-12"],
        "identities": [],
        "pending_forensics_targets": ["ws-dev-12"],
    }
    action = CybersecAction(action_type="REQUEST_FORENSICS", target="ws-dev-12")
    assert inference._enforce_action_contract(observation, action) is None
