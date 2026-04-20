from __future__ import annotations

try:
    from cybersec.models import CybersecAction
    from cybersec.server.cybersec_environment import CybersecEnvironment
except ModuleNotFoundError:
    from models import CybersecAction
    from server.cybersec_environment import CybersecEnvironment


def _new_env() -> CybersecEnvironment:
    return CybersecEnvironment(scenario_id="supply_chain_token_drift", seed=7)


def test_step_before_reset_ignores_action_and_exposes_machine_error() -> None:
    env = _new_env()

    obs = env.step(CybersecAction(action_type="MONITOR"))

    assert obs.done is False
    assert obs.reward is None
    assert obs.info.get("invalid_action") == "step_called_before_reset_action_ignored"
    assert (
        obs.info.get("last_action_error") == "step_called_before_reset_action_ignored"
    )


def test_step_after_terminal_returns_terminal_observation() -> None:
    env = CybersecEnvironment(
        scenario_id="federated_identity_takeover",
        seed=13,
        horizon=1,
    )
    env.reset()

    terminal = env.step(CybersecAction(action_type="MONITOR"))
    assert terminal.done is True

    post_terminal = env.step(CybersecAction(action_type="MONITOR"))
    assert post_terminal.done is True
    assert post_terminal.reward == 0.0
    assert (
        post_terminal.info.get("invalid_action")
        == "episode_already_terminated_call_reset"
    )
    assert post_terminal.info.get("terminal_reason") == "horizon_reached"


def test_reset_is_deterministic_for_same_seed_and_scenario() -> None:
    env_a = CybersecEnvironment(scenario_id="insider_repo_pivot")
    env_b = CybersecEnvironment(scenario_id="insider_repo_pivot")

    obs_a = env_a.reset(seed=111)
    obs_b = env_b.reset(seed=111)

    assert obs_a.scenario_id == obs_b.scenario_id
    assert obs_a.enterprise_risk_score == obs_b.enterprise_risk_score
    assert obs_a.horizon == obs_b.horizon


def test_horizon_override_terminates_episode_and_sets_terminal_fields() -> None:
    env = CybersecEnvironment(
        scenario_id="supply_chain_token_drift",
        seed=5,
        horizon=2,
    )
    env.reset()

    env.step(CybersecAction(action_type="MONITOR"))
    terminal = env.step(CybersecAction(action_type="MONITOR"))

    assert terminal.done is True
    assert terminal.info.get("terminal_reason") == "horizon_reached"
    assert isinstance(terminal.info.get("grader_score"), float)
    assert isinstance(terminal.info.get("grader_success"), bool)


def test_open_ticket_then_execute_after_ready() -> None:
    env = _new_env()
    obs = env.reset(seed=9)

    obs = env.step(
        CybersecAction(
            action_type="OPEN_TICKET",
            target="edge-egress",
            parameter="BLOCK_EGRESS",
            urgency="high",
        )
    )
    assert obs.done is False
    assert len(obs.open_tickets) >= 1
    ticket_id = obs.open_tickets[0].ticket_id

    ready = False
    for _ in range(6):
        ready_obs = env.step(CybersecAction(action_type="MONITOR"))
        ticket = next(
            (t for t in ready_obs.open_tickets if t.ticket_id == ticket_id), None
        )
        if ticket and ticket.status == "ready":
            ready = True
            break

    assert ready is True

    executed = env.step(CybersecAction(action_type="EXECUTE_TICKET", target=ticket_id))
    assert executed.done in {False, True}
    assert any("Executed ticket" in event for event in executed.recent_activity)


def test_execute_ticket_before_ready_returns_invalid_action() -> None:
    env = _new_env()
    env.reset(seed=9)
    obs = env.step(
        CybersecAction(
            action_type="OPEN_TICKET",
            target="edge-egress",
            parameter="BLOCK_EGRESS",
            urgency="high",
        )
    )
    ticket_id = obs.open_tickets[0].ticket_id

    invalid = env.step(CybersecAction(action_type="EXECUTE_TICKET", target=ticket_id))
    assert invalid.done is False
    assert "not executable" in str(invalid.info.get("invalid_action", ""))


def test_reward_breakdown_and_valid_targets_present() -> None:
    env = _new_env()
    env.reset(seed=17)

    obs = env.step(CybersecAction(action_type="QUERY_LOGS", parameter="cloud"))

    assert isinstance(obs.reward, float)
    breakdown = obs.info.get("reward_breakdown")
    assert isinstance(breakdown, dict)
    assert "detection" in breakdown
    assert "containment" in breakdown
    assert "total" in breakdown

    assert "assets" in obs.valid_targets
    assert "identities" in obs.valid_targets
    assert "query_log_sources" in obs.valid_targets
    assert "ticketable_actions" in obs.valid_targets
    assert "pending_forensics_targets" in obs.valid_targets


def test_reset_supports_scenario_switch() -> None:
    env = _new_env()
    obs = env.reset(seed=10, scenario_id="federated_identity_takeover")

    assert obs.scenario_id == "federated_identity_takeover"
    assert "identity" in obs.valid_targets.get("query_log_sources", [])


def test_duplicate_forensics_request_returns_invalid_action() -> None:
    env = _new_env()
    env.reset(seed=7)

    first = env.step(
        CybersecAction(action_type="REQUEST_FORENSICS", target="ws-dev-12")
    )
    assert first.done is False

    second = env.step(
        CybersecAction(action_type="REQUEST_FORENSICS", target="ws-dev-12")
    )
    assert second.done is False
    assert "already pending" in str(second.info.get("invalid_action", ""))
