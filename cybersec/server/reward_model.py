"""Reward decomposition and scoring for CybersecEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .world_state import WorldState


@dataclass
class StepSignals:
    """Signals produced by defender and environment interactions at one step."""

    detection_gain: float = 0.0
    containment_gain: float = 0.0
    disruption_cost: float = 0.0
    governance_cost: float = 0.0
    efficiency_cost: float = 0.0
    adversary_progress_delta: float = 0.0
    false_positive_penalty: float = 0.0
    invalid_action_penalty: float = 0.0


class RewardModel:
    """Computes dense and terminal reward signals with diagnostics."""

    def __init__(self, horizon: int):
        self._horizon = max(1, horizon)

    def step_reward(
        self, *, world: WorldState, signals: StepSignals, done: bool
    ) -> tuple[float, Dict[str, float]]:
        detection = _clamp01(signals.detection_gain)
        containment = _clamp01(signals.containment_gain)

        business_continuity = _clamp01(1.0 - signals.disruption_cost)
        governance = _clamp01(1.0 - signals.governance_cost)
        efficiency = _clamp01(1.0 - signals.efficiency_cost)

        disruption_penalty = _clamp01(signals.disruption_cost)
        governance_penalty = _clamp01(signals.governance_cost)
        efficiency_penalty = _clamp01(signals.efficiency_cost)

        adversary_pressure = _clamp01(signals.adversary_progress_delta)
        false_positive = _clamp01(signals.false_positive_penalty)
        invalid_action = _clamp01(signals.invalid_action_penalty)

        terminal_outcome = 0.0
        if done:
            terminal_outcome = self._terminal_outcome(world)

        total = 0.0
        total += 0.22 * detection
        total += 0.30 * containment
        total += 0.28 * terminal_outcome
        total -= 0.22 * adversary_pressure
        total -= 0.16 * disruption_penalty
        total -= 0.10 * governance_penalty
        total -= 0.08 * efficiency_penalty
        total -= 0.10 * false_positive
        total -= 0.10 * invalid_action
        if not done:
            total -= 0.02

        total = _clamp(total, -1.0, 1.0)

        breakdown = {
            "detection": detection,
            "containment": containment,
            "business_continuity": business_continuity,
            "governance": governance,
            "efficiency": efficiency,
            "disruption_penalty": disruption_penalty,
            "governance_penalty": governance_penalty,
            "efficiency_penalty": efficiency_penalty,
            "adversary_pressure": adversary_pressure,
            "terminal_outcome": terminal_outcome,
            "false_positive_penalty": false_positive,
            "invalid_action_penalty": invalid_action,
            "total": total,
        }
        return float(total), breakdown

    def terminal_score(self, world: WorldState) -> float:
        """Score in [0,1] used for benchmark grading and demo plots."""
        if world.exfiltration_succeeded:
            base = 0.0
        else:
            base = 0.45

        progress_penalty = 0.30 * _clamp01(
            world.attacker_progress / max(1.0, world.max_progress)
        )
        risk_penalty = 0.20 * world.enterprise_risk_score()
        focus_bonus = 0.20 * world.protected_focus_ratio()
        blocked_bonus = min(0.12, 0.02 * world.blocked_attempts)
        switch_bonus = min(0.08, 0.02 * world.path_switches)

        score = (
            base
            + focus_bonus
            + blocked_bonus
            + switch_bonus
            - progress_penalty
            - risk_penalty
        )
        return _clamp01(score)

    def grader_success(self, world: WorldState) -> bool:
        return self.terminal_score(world) >= 0.72

    def _terminal_outcome(self, world: WorldState) -> float:
        score = self.terminal_score(world)
        return _clamp(-0.2 + 1.2 * score, 0.0, 1.0)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return float(value)
