"""Reward model.

Seven clean channels are exposed to the policy and to downstream RL training:

  * ``detection``              - bonus for confirming a real compromise via INVESTIGATE
  * ``containment``            - bonus for blocking a stage that would otherwise advance
  * ``evidence_bonus``         - bonus for containing a previously confirmed-compromised target
  * ``false_positive_penalty`` - cost for INVESTIGATE / ISOLATE / REVOKE on a clean target
  * ``disruption_penalty``     - per-tick cost of having business assets isolated/blocked
  * ``invalid_action_penalty`` - cost for an action whose target is not in valid_targets
  * ``terminal_score``         - large terminal payoff/penalty (only on the final tick)

Each channel is computed from a simple :class:`StepSignals` struct populated
by ``env.step``. The defender's per-step reward is the sum, clipped to a
fixed range so a single overpowered signal cannot dominate training.

Design safeguards:
  * Detection only fires on the *first* confirmation of a given target.
  * Containment only fires on attack-path targets (not arbitrary isolation).
  * Disruption scales with weighted isolation load; optional per-tick cap
   (``disruption_cap_per_tick``). Default is **uncapped** so mass-isolation
   pays linearly and cannot be arbitraged against a small fixed penalty.
  * Per-step total is clipped to ``[-step_clip, +step_clip]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

# Two-mode import: relative when imported as ``cybersec.reward``, absolute
# fallback when ``reward`` is loaded as a top-level module from CWD on HF.
try:
    from .models import RewardBreakdown
except ImportError:  # pragma: no cover - HF Spaces / docker runtime path
    from models import RewardBreakdown  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Tunable weights (kept in one place so RL pipelines can override)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    detection_per_target: float = 1.5
    containment_active_per_stage: float = 2.5
    containment_preemptive_per_stage: float = 0.4
    containment_action_cost: float = 0.15
    false_positive_per_action: float = 0.6
    invalid_action: float = 0.6
    disruption_per_active_control: float = 0.08
    # None = no cap (full -raw_disruption). Set to e.g. 0.6 for legacy capped shaping.
    disruption_cap_per_tick: float | None = None
    step_clip: float = 5.0
    terminal_clean_bonus: float = 5.0
    terminal_partial_per_stage: float = 0.6
    terminal_exfil_penalty: float = 6.0
    terminal_idle_penalty: float = 1.5  # if defender did nothing all episode
    evidence_bonus_per_target: float = 1.5  # bonus for contain after confirmed compromise


# ---------------------------------------------------------------------------
# Per-step signals (populated by env.step before reward computation)
# ---------------------------------------------------------------------------


@dataclass
class StepSignals:
    """Inputs for one call to :meth:`RewardModel.step_reward`."""

    new_confirmed_compromised: Set[str] = field(default_factory=set)
    contained_active_stage_ids: Set[str] = field(default_factory=set)
    contained_preemptive_stage_ids: Set[str] = field(default_factory=set)
    false_positive_count: int = 0
    invalid_action_count: int = 0
    containment_action_count: int = 0
    active_control_count: int = 0
    weighted_disruption: float = 0.0  # sum of criticality * indicator
    is_terminal: bool = False
    exfil_completed: bool = False
    succeeded_stage_count: int = 0
    total_stage_count: int = 0
    defender_acted_at_least_once: bool = True
    containment_on_confirmed: int = 0  # containment actions on confirmed-compromised targets


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------


class RewardModel:
    """Computes per-step reward + terminal score using :class:`StepSignals`."""

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    # -------------------------------------------------------------- step

    def step_reward(self, signals: StepSignals) -> RewardBreakdown:
        w = self.weights

        detection = w.detection_per_target * len(signals.new_confirmed_compromised)
        containment = (
            w.containment_active_per_stage * len(signals.contained_active_stage_ids)
            + w.containment_preemptive_per_stage * len(signals.contained_preemptive_stage_ids)
            - w.containment_action_cost * signals.containment_action_count
        )
        evidence = w.evidence_bonus_per_target * signals.containment_on_confirmed
        false_positive = -w.false_positive_per_action * signals.false_positive_count
        invalid = -w.invalid_action * signals.invalid_action_count

        raw_disruption = w.disruption_per_active_control * signals.weighted_disruption
        if w.disruption_cap_per_tick is None:
            disruption = -raw_disruption
        else:
            disruption = -min(raw_disruption, w.disruption_cap_per_tick)

        terminal = self._terminal_score(signals) if signals.is_terminal else 0.0

        total = detection + containment + evidence + false_positive + invalid + disruption + terminal
        # Step-level clip protects training from rare blowups; terminal is
        # added separately and *not* clipped (it is what defines win/lose).
        non_terminal = total - terminal
        non_terminal = max(-w.step_clip, min(w.step_clip, non_terminal))
        total = non_terminal + terminal

        return RewardBreakdown(
            detection=round(detection, 4),
            containment=round(containment, 4),
            evidence_bonus=round(evidence, 4),
            false_positive_penalty=round(false_positive, 4),
            disruption_penalty=round(disruption, 4),
            invalid_action_penalty=round(invalid, 4),
            terminal_score=round(terminal, 4),
            total=round(total, 4),
        )

    # -------------------------------------------------------------- terminal

    def _terminal_score(self, signals: StepSignals) -> float:
        w = self.weights
        if signals.exfil_completed:
            # Exfil happened: hard penalty, but defender still gets partial
            # credit for stages prevented before exfil.
            stages_prevented = max(0, signals.total_stage_count - signals.succeeded_stage_count)
            return -w.terminal_exfil_penalty + w.terminal_partial_per_stage * stages_prevented

        if signals.succeeded_stage_count == 0:
            return w.terminal_clean_bonus

        # Attacker made progress but never exfiltrated: scale with how much
        # of the kill chain they actually completed.
        stages_prevented = max(0, signals.total_stage_count - signals.succeeded_stage_count)
        score = w.terminal_partial_per_stage * stages_prevented
        if not signals.defender_acted_at_least_once:
            score -= w.terminal_idle_penalty
        return score


__all__ = ["RewardWeights", "StepSignals", "RewardModel"]
