"""End-to-end contract tests for :class:`CybersecEnvironment`.

These tests guarantee:

  * The OpenEnv ``Environment`` interface is satisfied (reset / step / state).
  * Reset is deterministic for a given (seed, scenario_id, personality).
  * Episodes always terminate within ``horizon`` ticks.
  * Reward components are present in the observation's ``info`` dict.
  * The state endpoint never leaks ground-truth attacker progress while the
    episode is still running.
"""

from __future__ import annotations

import pytest

from cybersec import (
    ActionType,
    AttackerPersonality,
    CybersecAction,
    CybersecEnvironment,
    CybersecObservation,
    list_scenarios,
)
from cybersec.baselines import HeuristicPolicy, RandomPolicy, run_episode


def _is_obs(obj) -> bool:
    return isinstance(obj, CybersecObservation)


# ---------------------------------------------------------------------------
# reset / state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_reset_returns_well_formed_observation(env: CybersecEnvironment, scenario_id: str) -> None:
    obs = env.reset(seed=7, scenario_id=scenario_id)
    assert _is_obs(obs)
    assert obs.tick == 0
    assert obs.scenario_id == scenario_id
    assert obs.horizon >= 60
    assert isinstance(obs.attacker_personality, AttackerPersonality)
    assert "assets" in obs.valid_targets and "identities" in obs.valid_targets
    assert obs.valid_targets["assets"], "scenario should expose at least one asset"
    assert obs.valid_targets["identities"], "scenario should expose at least one identity"
    assert set(obs.available_actions) == set(ActionType)


def test_reset_is_deterministic_under_same_seed(env: CybersecEnvironment) -> None:
    obs_a = env.reset(seed=42, scenario_id="supply_chain_token_drift")
    obs_b = env.reset(seed=42, scenario_id="supply_chain_token_drift")
    # Compare via model_dump - lists/sets serialize stably for identical state.
    assert obs_a.model_dump() == obs_b.model_dump()


def test_state_endpoint_does_not_leak_attacker_internals(env: CybersecEnvironment) -> None:
    env.reset(seed=0, scenario_id="federated_identity_takeover")
    state = env.state
    # CybersecState only ships the *succeeded* stage IDs, never the in-progress one.
    assert state.completed_attack_stages == [] or all(
        isinstance(s, str) for s in state.completed_attack_stages
    )
    # No private attacker fields should be on the public state.
    blocked_keys = {"runtimes", "_compromised_assets", "_compromised_identities"}
    assert not (set(state.model_dump().keys()) & blocked_keys)


# ---------------------------------------------------------------------------
# step termination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("personality", list(AttackerPersonality))
def test_episode_terminates_within_horizon(env: CybersecEnvironment, personality: AttackerPersonality) -> None:
    result = run_episode(
        env,
        HeuristicPolicy(),
        seed=11,
        scenario_id="insider_repo_pivot",
        attacker_personality=personality,
    )
    assert result.steps <= 80, f"episode ran past horizon: {result.steps}"
    assert result.terminal_reason in {
        "exfil_completed",
        "horizon_reached",
        "attacker_done",
    }


def test_step_returns_reward_breakdown(env: CybersecEnvironment) -> None:
    env.reset(seed=3, scenario_id="supply_chain_token_drift")
    obs = env.step(CybersecAction(action_type=ActionType.MONITOR))
    breakdown = obs.info.get("reward_breakdown")
    assert breakdown is not None, "every step must populate info.reward_breakdown"
    assert {"detection", "containment", "false_positive_penalty", "disruption_penalty",
            "invalid_action_penalty", "terminal_score", "total"} <= breakdown.keys()


def test_terminal_step_carries_terminal_info(env: CybersecEnvironment) -> None:
    result = run_episode(env, RandomPolicy(seed=5), seed=5, scenario_id="supply_chain_token_drift")
    # Confirm the runner picked the terminal info up; if it's missing the runner
    # would have produced 0 for these fields and run_episode would have fallen
    # back to the manual sum.
    assert result.terminal_reason is not None
    assert result.total_stage_count > 0


# ---------------------------------------------------------------------------
# step against a terminated env
# ---------------------------------------------------------------------------


def test_stepping_after_done_raises(env: CybersecEnvironment) -> None:
    env.reset(seed=0, scenario_id="supply_chain_token_drift")
    # Drive episode to completion using all-MONITOR (will hit horizon eventually).
    obs = None
    for _ in range(200):  # generous upper bound
        obs = env.step(CybersecAction(action_type=ActionType.MONITOR))
        if obs.done:
            break
    assert obs is not None and obs.done, "episode should terminate within 200 ticks"
    with pytest.raises(RuntimeError):
        env.step(CybersecAction(action_type=ActionType.MONITOR))


# ---------------------------------------------------------------------------
# baselines sanity: heuristic should beat random on average
# ---------------------------------------------------------------------------


def test_heuristic_outperforms_random_on_average() -> None:
    env = CybersecEnvironment()
    seeds = list(range(15))
    rand = [run_episode(env, RandomPolicy(seed=s), seed=s) for s in seeds]
    heur = [run_episode(env, HeuristicPolicy(), seed=s) for s in seeds]
    rand_mean = sum(r.cumulative_reward for r in rand) / len(rand)
    heur_mean = sum(r.cumulative_reward for r in heur) / len(heur)
    assert heur_mean > rand_mean, (
        f"heuristic ({heur_mean:.2f}) should beat random ({rand_mean:.2f}); "
        "if this fails, the reward shaping has regressed."
    )
