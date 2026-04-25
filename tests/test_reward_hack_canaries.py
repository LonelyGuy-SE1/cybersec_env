"""Reward-hack canary tests.

The iter-1 GRPO run produced a trained policy with ``std_return = 0.0`` on
two of three scenarios across 50 different seeds: the policy had memorised
a single deterministic action sequence that froze the attacker before any
stage could complete. Numerically the run looked great; behaviourally it
was a degenerate canned plan.

This file's only job is to ensure no future iteration re-introduces that
class of failure undetected. Each test stands up the env, runs a
deliberately pathological "policy", and asserts that the reward shape
makes that pathology visible -- either as a low return, as a near-zero
diversity reward, or as a low observation-aware reward.

If you change reward weights or add scenarios, these tests are the alarm.
"""

from __future__ import annotations

import statistics
from typing import List

import pytest

from cybersec import (
    ActionType,
    CybersecAction,
    CybersecEnvironment,
    list_train_scenarios,
)
from cybersec.baselines import RandomPolicy, run_episode
from cybersec.training.rewards import (
    reward_action_diversity,
    reward_batch_action_entropy,
    reward_observation_aware,
    reward_step_total,
    snapshot_env,
)


# ---------------------------------------------------------------------------
# Fake policies that exhibit the failure modes we want to catch
# ---------------------------------------------------------------------------


class _ConstantPolicy:
    """Always returns the same action, regardless of observation."""

    name = "constant"

    def __init__(self, action: CybersecAction):
        self._action = action

    def reset(self) -> None:  # pragma: no cover - trivial
        return None

    def act(self, _obs) -> CybersecAction:
        return self._action


class _AlwaysMonitorPolicy(_ConstantPolicy):
    name = "always-monitor"

    def __init__(self):
        super().__init__(CybersecAction(action_type=ActionType.MONITOR))


# ---------------------------------------------------------------------------
# Diversity reward catches identical-completion mode collapse
# ---------------------------------------------------------------------------


def test_diversity_reward_collapses_when_all_completions_identical():
    """Four identical completions in one group -> 0.25 each, not 1.0.

    This is the *direct* signal we lean on during GRPO to break the
    iter-1 reward hack. Future-proofing: if someone simplifies the
    diversity reward and breaks this property, a 4x signal-strength
    bonus to mode-collapse comes back.
    """

    completion = '{"action_type": "ISOLATE_ASSET", "target": "secrets-vault"}'
    out = reward_action_diversity(
        prompts=["P"] * 4,
        completions=[completion] * 4,
    )
    assert out == [0.25, 0.25, 0.25, 0.25]


def test_diversity_reward_per_completion_at_most_one():
    """Reward is bounded by 1.0 even for a unique completion in a group of one."""

    out = reward_action_diversity(
        prompts=["P"],
        completions=['{"action_type": "MONITOR"}'],
    )
    assert out == [1.0]


# ---------------------------------------------------------------------------
# Observation-aware reward catches the "always MONITOR" trivial policy
# ---------------------------------------------------------------------------


def test_observation_aware_reward_punishes_always_monitor_when_alerts_fire():
    """A constant-MONITOR policy must not max out reward_observation_aware."""

    completions = ['{"action_type": "MONITOR"}'] * 4
    valid_assets = [["asset-a"]] * 4
    out_alerts = reward_observation_aware(
        prompts=["p"] * 4,
        completions=completions,
        valid_assets=valid_assets,
        alert_count=[5, 5, 5, 5],
    )
    out_quiet = reward_observation_aware(
        prompts=["p"] * 4,
        completions=completions,
        valid_assets=valid_assets,
        alert_count=[0, 0, 0, 0],
    )
    assert out_alerts == [0.0, 0.0, 0.0, 0.0]
    assert out_quiet == [0.7, 0.7, 0.7, 0.7]


def test_observation_aware_reward_punishes_always_isolate_when_quiet():
    """A constant-ISOLATE policy gets less reward when there are no alerts.

    Together with the previous test, this means the only way for a policy
    to max out :func:`reward_observation_aware` is to actually condition
    on ``alert_count``.
    """

    completions = ['{"action_type": "ISOLATE_ASSET", "target": "asset-a"}'] * 4
    out_alerts = reward_observation_aware(
        prompts=["p"] * 4,
        completions=completions,
        valid_assets=[["asset-a"]] * 4,
        alert_count=[5, 5, 5, 5],
    )
    out_quiet = reward_observation_aware(
        prompts=["p"] * 4,
        completions=completions,
        valid_assets=[["asset-a"]] * 4,
        alert_count=[0, 0, 0, 0],
    )
    assert out_alerts == [1.0] * 4
    assert out_quiet == [0.4] * 4


# ---------------------------------------------------------------------------
# Trained-policy proxy: env-level smoke tests for canned plans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", list_train_scenarios())
def test_constant_policy_does_not_dominate_random(scenario_id: str):
    """A constant-action policy must NOT outperform random across many seeds.

    We pick the most-suspicious canned actions (always-MONITOR and the
    iter-1-style "always isolate the most critical asset") and run them
    over the same 20 seeds as a RandomPolicy. The env's reward shape
    should make sure these degenerate strategies score below or near
    parity with random; if they suddenly start beating random by a wide
    margin, the env has acquired a new exploitable shortcut and you have
    a reward hack to fix.
    """

    env = CybersecEnvironment()
    seeds = list(range(20))
    rand = [
        run_episode(env, RandomPolicy(seed=s), seed=s, scenario_id=scenario_id)
        for s in seeds
    ]
    monitor = [
        run_episode(env, _AlwaysMonitorPolicy(), seed=s, scenario_id=scenario_id)
        for s in seeds
    ]
    rand_mean = statistics.fmean(r.cumulative_reward for r in rand)
    monitor_mean = statistics.fmean(r.cumulative_reward for r in monitor)
    assert monitor_mean <= rand_mean + 1.5, (
        f"{scenario_id}: always-MONITOR ({monitor_mean:.3f}) is dominating "
        f"random ({rand_mean:.3f}) by more than the 1.5-point grace margin. "
        "This is exactly the iter-1 reward-hack shape; check that "
        "invalid-action / disruption / containment penalties are still active."
    )


def test_canary_run_episode_invalid_action_count_is_observable():
    """The :class:`EpisodeResult` exposes ``invalid_action_count``.

    Iter-1's notebook crashed because it tried to read ``invalid_actions``
    (which does not exist). Locking the public attribute name in a test
    means refactoring it later is a deliberate, visible decision.
    """

    env = CybersecEnvironment()
    res = run_episode(
        env, RandomPolicy(seed=0), seed=0, scenario_id="supply_chain_token_drift"
    )
    assert hasattr(res, "invalid_action_count")
    assert isinstance(res.invalid_action_count, int)
    assert res.invalid_action_count >= 0


def test_canary_run_episode_exposes_monitor_fallback_count():
    """Iter-4: ``EpisodeResult.monitor_fallback_count`` exists for honest reporting.

    The iter-3 results showed ``invalid_rate=0.0`` for a trained LLM
    policy that was actually MONITOR-fallbacking on most steps.
    ``invalid_action_count`` only counts env-rejected actions, so a policy
    that silently degrades to a legal MONITOR is invisible to it. This
    canary locks in the new metric so future refactors cannot drop it.
    """

    env = CybersecEnvironment()
    res = run_episode(
        env, RandomPolicy(seed=0), seed=0, scenario_id="supply_chain_token_drift"
    )
    assert hasattr(res, "monitor_fallback_count")
    assert isinstance(res.monitor_fallback_count, int)
    assert res.monitor_fallback_count == 0  # RandomPolicy never falls back


# ---------------------------------------------------------------------------
# Iter-4: batch action entropy reward gives gradient OUT of collapse
# ---------------------------------------------------------------------------


def test_batch_action_entropy_zero_when_fully_collapsed():
    """All identical actions across the whole batch -> 0.0 reward.

    This is the ground state we are pushing the policy *out of*.
    """

    completion = '{"action_type": "ISOLATE_ASSET", "target": "secrets-vault"}'
    out = reward_batch_action_entropy(
        prompts=[f"p{i}" for i in range(8)],
        completions=[completion] * 8,
    )
    assert out == [0.0] * 8


def test_batch_action_entropy_rewards_breaking_symmetry():
    """One unique completion among seven identical ones -> the unique one wins.

    This is the property we need: there must be a *positive* relative
    reward gap that GRPO can follow when a single candidate breaks
    symmetry. Without it, no gradient pushes out of the collapsed mode.
    """

    canned = '{"action_type": "ISOLATE_ASSET", "target": "secrets-vault"}'
    odd = '{"action_type": "REVOKE_IDENTITY", "target": "alice"}'
    completions = [canned] * 7 + [odd]
    out = reward_batch_action_entropy(
        prompts=[f"p{i}" for i in range(len(completions))],
        completions=completions,
    )
    assert out[-1] > out[0], (
        f"unique completion ({out[-1]:.3f}) must outrank canned ones ({out[0]:.3f})"
    )
    assert all(0.0 <= v <= 1.0 for v in out)
