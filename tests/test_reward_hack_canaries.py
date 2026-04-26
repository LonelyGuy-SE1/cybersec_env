"""Regression tests for reward shaping and episode metrics.

Exercises pathological policies (constant actions, always-MONITOR, etc.) and
asserts that reward terms and ``EpisodeResult`` fields expose undesirable
behaviour. Extend these tests when changing reward weights or scenarios.
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
# Environment-level checks for degenerate constant policies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", list_train_scenarios())
def test_constant_policy_does_not_dominate_random(scenario_id: str):
    """Constant policies should not strongly dominate random under the reward model.

    Compares always-MONITOR against ``RandomPolicy`` over shared seeds. A large
    systematic advantage for the degenerate policy indicates unintended reward
    shaping.
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
        "Check invalid-action, disruption, and containment penalties are still active."
    )


def test_canary_run_episode_invalid_action_count_is_observable():
    """The :class:`EpisodeResult` exposes ``invalid_action_count``.

    Locks the public ``invalid_action_count`` attribute for stable downstream use.
    """

    env = CybersecEnvironment()
    res = run_episode(
        env, RandomPolicy(seed=0), seed=0, scenario_id="supply_chain_token_drift"
    )
    assert hasattr(res, "invalid_action_count")
    assert isinstance(res.invalid_action_count, int)
    assert res.invalid_action_count >= 0


def test_canary_run_episode_exposes_monitor_fallback_count():
    """``EpisodeResult.monitor_fallback_count`` is present and typed.

    Distinguishes parse-or-policy fallbacks to MONITOR from env-rejected
    actions counted in ``invalid_action_count``.
    """

    env = CybersecEnvironment()
    res = run_episode(
        env, RandomPolicy(seed=0), seed=0, scenario_id="supply_chain_token_drift"
    )
    assert hasattr(res, "monitor_fallback_count")
    assert isinstance(res.monitor_fallback_count, int)
    assert res.monitor_fallback_count == 0  # RandomPolicy never falls back
