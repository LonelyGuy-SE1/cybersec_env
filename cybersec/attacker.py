"""Scripted attacker policy.

The attacker is a small state machine that walks the per-scenario stage DAG.
It is intentionally *not* a learned agent - we want a deterministic-ish
adversary so reward signals stay attributable, and we want three legible
"personalities" for the storytelling layer:

  * STEALTHY     - longer dwell, quieter alerts, pauses after defender activity.
  * AGGRESSIVE   - shorter dwell, louder alerts, never pauses.
  * OPPORTUNISTIC- nominal dwell/noise, will reroute around blocked stages.

The attacker's step() runs once per environment tick and returns an
:class:`AttackerEvent` describing what happened: stages it started, stages it
completed (success or failure), stages that were blocked by defender action,
plus any alerts that should surface to the defender (with their per-alert
lag already applied).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .models import AlertEvent, AlertSignal, AttackerPersonality
from .scenarios import AttackStage, Scenario


# ---------------------------------------------------------------------------
# Personality knobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonalityProfile:
    """Numeric tweaks the attacker applies to every stage."""

    dwell_multiplier: float
    detection_multiplier: float
    success_bias: float  # added to (then clamped) per-stage success_prob
    pause_after_defender_action_prob: float
    reroute_on_block: bool


_PROFILES: Dict[AttackerPersonality, PersonalityProfile] = {
    AttackerPersonality.STEALTHY: PersonalityProfile(
        dwell_multiplier=1.5,
        detection_multiplier=0.55,
        success_bias=-0.05,
        pause_after_defender_action_prob=0.5,
        reroute_on_block=False,
    ),
    AttackerPersonality.AGGRESSIVE: PersonalityProfile(
        dwell_multiplier=0.6,
        detection_multiplier=1.3,
        success_bias=0.05,
        pause_after_defender_action_prob=0.0,
        reroute_on_block=False,
    ),
    AttackerPersonality.OPPORTUNISTIC: PersonalityProfile(
        dwell_multiplier=1.0,
        detection_multiplier=1.0,
        success_bias=0.0,
        pause_after_defender_action_prob=0.15,
        reroute_on_block=True,
    ),
}


def get_personality_profile(personality: AttackerPersonality) -> PersonalityProfile:
    return _PROFILES[personality]


# ---------------------------------------------------------------------------
# Per-stage runtime state
# ---------------------------------------------------------------------------


class StageStatus(str, Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class StageRuntime:
    """Mutable runtime state for one stage."""

    stage: AttackStage
    status: StageStatus = StageStatus.IDLE
    started_tick: Optional[int] = None
    target_dwell: int = 0
    completes_at: Optional[int] = None
    pending_alerts: List[Tuple[int, AlertEvent]] = field(default_factory=list)
    last_alert_tick: int = -1


# ---------------------------------------------------------------------------
# Defender view contract (passed in by env.py)
# ---------------------------------------------------------------------------


@dataclass
class DefenderView:
    """Slice of defender state the attacker reads when deciding what to do."""

    isolated_assets: Set[str]
    revoked_identities: Set[str]
    blocked_egress_assets: Set[str]
    patched_assets: Set[str]
    defender_acted_this_tick: bool


# ---------------------------------------------------------------------------
# AttackerEvent: what happened this tick
# ---------------------------------------------------------------------------


@dataclass
class AttackerEvent:
    started: List[str] = field(default_factory=list)
    succeeded: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    blocked_active: List[str] = field(default_factory=list)
    blocked_preemptive: List[str] = field(default_factory=list)
    surfaced_alerts: List[AlertEvent] = field(default_factory=list)
    exfil_completed: bool = False

    @property
    def blocked(self) -> List[str]:
        """Convenience: union of active + preemptive blocks (read-only)."""

        return self.blocked_active + self.blocked_preemptive


# ---------------------------------------------------------------------------
# ScriptedAttacker
# ---------------------------------------------------------------------------


class ScriptedAttacker:
    """Step-driven attacker walking a scenario's stage DAG.

    Update order each tick (called by ``env.step`` after the defender action
    has already been applied):

      1. Drain any pending alerts that have reached their visibility tick.
      2. Resolve in-progress stages whose dwell timer fires this tick.
      3. Possibly pause (stealthy / opportunistic personalities).
      4. Pick at most one new stage to start, respecting prereqs and
         defender containment.
      5. Roll lag and severity for any alerts the new stage will produce.

    The attacker only ever has one stage in_progress at a time. This keeps
    the long-horizon planning challenge legible: the defender is always
    racing one specific dwell timer.
    """

    def __init__(
        self,
        scenario: Scenario,
        personality: AttackerPersonality,
        rng: random.Random,
    ):
        self.scenario = scenario
        self.personality = personality
        self.profile = get_personality_profile(personality)
        self.rng = rng
        self.runtimes: Dict[str, StageRuntime] = {
            stage.stage_id: StageRuntime(stage=stage) for stage in scenario.stages
        }
        self._compromised_assets: Set[str] = set()
        self._compromised_identities: Set[str] = set()

    # ------------------------------------------------------------------ helpers

    def _all_prereqs_met(self, stage: AttackStage) -> bool:
        return all(
            self.runtimes[p].status is StageStatus.SUCCEEDED
            for p in stage.prereq_stages
        )

    def _stage_blocked_by_defender(
        self, stage: AttackStage, defender: DefenderView
    ) -> bool:
        if stage.target_asset and stage.target_asset in defender.isolated_assets:
            return True
        if stage.target_identity and stage.target_identity in defender.revoked_identities:
            return True
        if stage.is_exfil and stage.target_asset in defender.blocked_egress_assets:
            return True
        return False

    def _pickable_stages(self, defender: DefenderView) -> List[StageRuntime]:
        out: List[StageRuntime] = []
        for rt in self.runtimes.values():
            if rt.status is not StageStatus.IDLE:
                continue
            if not self._all_prereqs_met(rt.stage):
                continue
            if self._stage_blocked_by_defender(rt.stage, defender):
                continue
            out.append(rt)
        return out

    def _has_active_stage(self) -> bool:
        return any(rt.status is StageStatus.IN_PROGRESS for rt in self.runtimes.values())

    def _build_alerts_for_stage(self, stage: AttackStage, start_tick: int) -> List[Tuple[int, AlertEvent]]:
        """Roll the alert plan a stage will produce while it dwells.

        We emit between 0 and 2 alerts per stage. Alert visibility is
        ``start_tick + lag``; lag is sampled inside ``alert_lag_range``.
        """

        adjusted_strength = max(
            0.0, min(1.0, stage.detection_strength * self.profile.detection_multiplier)
        )
        # Expected alert count grows with detection strength, capped at 2.
        n_alerts = 0
        for _ in range(2):
            if self.rng.random() < adjusted_strength:
                n_alerts += 1

        signal = self._signal_for_tactic(stage.mitre_tactic)
        plans: List[Tuple[int, AlertEvent]] = []
        for _ in range(n_alerts):
            lag = self.rng.randint(*stage.alert_lag_range)
            severity = max(0.05, min(1.0, adjusted_strength + self.rng.uniform(-0.15, 0.2)))
            event = AlertEvent(
                tick=start_tick + lag,
                signal=signal,
                asset=stage.target_asset or None,
                identity=stage.target_identity or None,
                severity=round(severity, 3),
                description=f"{stage.mitre_tactic} :: {stage.mitre_technique}",
            )
            plans.append((start_tick + lag, event))
        return plans

    @staticmethod
    def _signal_for_tactic(tactic: str) -> AlertSignal:
        # Keep this mapping coarse-grained so the policy gets a stable signal.
        if "Initial Access" in tactic or "Credential" in tactic:
            return AlertSignal.AUTH_ANOMALY
        if "Lateral Movement" in tactic or "Execution" in tactic:
            return AlertSignal.LATERAL_MOVEMENT
        if "Collection" in tactic or "Discovery" in tactic:
            return AlertSignal.DATA_STAGING
        if "Exfiltration" in tactic:
            return AlertSignal.EGRESS_ANOMALY
        return AlertSignal.AUTH_ANOMALY

    def _start_stage(self, rt: StageRuntime, tick: int) -> None:
        lo, hi = rt.stage.dwell_range
        scaled_lo = max(1, int(round(lo * self.profile.dwell_multiplier)))
        scaled_hi = max(scaled_lo, int(round(hi * self.profile.dwell_multiplier)))
        rt.target_dwell = self.rng.randint(scaled_lo, scaled_hi)
        rt.started_tick = tick
        rt.completes_at = tick + rt.target_dwell
        rt.status = StageStatus.IN_PROGRESS
        rt.pending_alerts = self._build_alerts_for_stage(rt.stage, tick)

    def _resolve_stage(self, rt: StageRuntime, defender: DefenderView) -> str:
        """Resolve a stage whose dwell timer just expired.

        Returns one of: 'succeeded', 'failed', 'blocked'.
        """

        if self._stage_blocked_by_defender(rt.stage, defender):
            rt.status = StageStatus.BLOCKED
            return "blocked"

        success_prob = max(
            0.05,
            min(0.99, rt.stage.success_prob + self.profile.success_bias),
        )
        # Patching the target asset cuts success probability noticeably.
        if rt.stage.target_asset and rt.stage.target_asset in defender.patched_assets:
            success_prob = max(0.05, success_prob - 0.3)

        if self.rng.random() < success_prob:
            rt.status = StageStatus.SUCCEEDED
            if rt.stage.compromises_asset and rt.stage.target_asset:
                self._compromised_assets.add(rt.stage.target_asset)
            if rt.stage.compromises_identity and rt.stage.target_identity:
                self._compromised_identities.add(rt.stage.target_identity)
            return "succeeded"

        rt.status = StageStatus.FAILED
        return "failed"

    # ------------------------------------------------------------------ main step

    def step(self, tick: int, defender: DefenderView) -> AttackerEvent:
        """Advance the attacker by one tick and return what happened."""

        ev = AttackerEvent()

        # 1. Surface alerts that have reached their visibility tick. Even
        #    for blocked stages, alerts in flight before the block still fire
        #    (defenders shouldn't be retroactively rewarded for late alerts).
        for rt in self.runtimes.values():
            if not rt.pending_alerts:
                continue
            still_pending: List[Tuple[int, AlertEvent]] = []
            for visible_at, alert in rt.pending_alerts:
                if visible_at <= tick:
                    ev.surfaced_alerts.append(alert)
                    rt.last_alert_tick = tick
                else:
                    still_pending.append((visible_at, alert))
            rt.pending_alerts = still_pending

        # 2. Resolve any in-progress stage that has now reached its dwell.
        for rt in self.runtimes.values():
            if rt.status is not StageStatus.IN_PROGRESS:
                continue
            # Mark blocked early so the defender gets credit even if the
            # dwell hasn't fully elapsed.
            if self._stage_blocked_by_defender(rt.stage, defender):
                rt.status = StageStatus.BLOCKED
                ev.blocked_active.append(rt.stage.stage_id)
                continue
            if rt.completes_at is not None and tick >= rt.completes_at:
                outcome = self._resolve_stage(rt, defender)
                if outcome == "succeeded":
                    ev.succeeded.append(rt.stage.stage_id)
                    if rt.stage.is_exfil:
                        ev.exfil_completed = True
                elif outcome == "blocked":
                    ev.blocked_active.append(rt.stage.stage_id)
                else:
                    ev.failed.append(rt.stage.stage_id)

        # 3. Personality-driven gating: maybe pause this tick.
        skip_start = False
        if self._has_active_stage():
            # Already busy; nothing else to start.
            skip_start = True
        elif (
            defender.defender_acted_this_tick
            and self.rng.random() < self.profile.pause_after_defender_action_prob
        ):
            skip_start = True

        # 4. Pick the next stage. Prefer canonical DAG order; opportunistic
        #    will reroute around dead-ends by skipping permanently-blocked stages.
        if not skip_start:
            candidates = self._pickable_stages(defender)
            if candidates:
                if self.profile.reroute_on_block:
                    # Prefer the earliest-defined stage among candidates (DAG
                    # order is preserved by Python dict insertion).
                    choice = candidates[0]
                else:
                    # Aggressive / stealthy stick to canonical order even if
                    # the earliest available has a contained target.
                    choice = candidates[0]
                self._start_stage(choice, tick)
                ev.started.append(choice.stage.stage_id)

        # 5. Flip any IDLE stage whose target the defender has already
        #    contained into BLOCKED (preemptive containment). This is what
        #    rewards good "yank the crown jewels offline" defender play even
        #    if the attacker never reaches that part of the chain. Preemptive
        #    blocks are tracked separately because they pay much less reward.
        for rt in self.runtimes.values():
            if rt.status is not StageStatus.IDLE:
                continue
            if self._stage_blocked_by_defender(rt.stage, defender):
                rt.status = StageStatus.BLOCKED
                ev.blocked_preemptive.append(rt.stage.stage_id)

        return ev

    # ------------------------------------------------------------------ public read-only

    @property
    def compromised_assets(self) -> Set[str]:
        return set(self._compromised_assets)

    @property
    def compromised_identities(self) -> Set[str]:
        return set(self._compromised_identities)

    def succeeded_stage_ids(self) -> List[str]:
        return [rt.stage.stage_id for rt in self.runtimes.values() if rt.status is StageStatus.SUCCEEDED]

    def in_progress_stage(self) -> Optional[StageRuntime]:
        for rt in self.runtimes.values():
            if rt.status is StageStatus.IN_PROGRESS:
                return rt
        return None

    def is_done(self) -> bool:
        """All stages are terminal (no idle, no in-progress, no resumable)."""

        for rt in self.runtimes.values():
            if rt.status is StageStatus.IN_PROGRESS:
                return False
            if rt.status is StageStatus.IDLE:
                # If prereqs can still be satisfied later, attacker isn't done.
                if all(
                    self.runtimes[p].status is StageStatus.SUCCEEDED
                    for p in rt.stage.prereq_stages
                ):
                    return False
        return True


__all__ = [
    "PersonalityProfile",
    "get_personality_profile",
    "StageStatus",
    "StageRuntime",
    "DefenderView",
    "AttackerEvent",
    "ScriptedAttacker",
]
