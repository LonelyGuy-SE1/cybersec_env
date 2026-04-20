"""Adaptive attacker policy for CybersecEnv multi-agent simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from .scenario_loader import (
    AttackPathDefinition,
    AttackStepDefinition,
    ScenarioDefinition,
)
from .world_state import WorldState


@dataclass
class AttackerEvent:
    """Single attacker-side event emitted each simulation tick."""

    attempted: bool
    success: bool
    blocked: bool
    path_id: Optional[str]
    step_id: Optional[str]
    stage: Optional[str]
    target: Optional[str]
    source: Optional[str]
    severity: Optional[str]
    description: str
    progress_delta: float
    detection_strength: float
    compromise_confirmed: bool
    exfiltration_succeeded: bool


class AttackerPolicy:
    """Stochastic attacker with path switching under defender pressure."""

    def __init__(self, scenario: ScenarioDefinition, rng: random.Random):
        self._scenario = scenario
        self._rng = rng
        self._active_path_index = 0
        self._next_step_index = 0
        self._stall_counter = 0
        self._failed_recently = False

        if not scenario.attack_paths:
            raise ValueError("Scenario must define at least one attack path")

    def progress(self, world: WorldState) -> AttackerEvent:
        if world.exfiltration_succeeded:
            return AttackerEvent(
                attempted=False,
                success=False,
                blocked=False,
                path_id=self.current_path.path_id,
                step_id=None,
                stage=None,
                target=None,
                source=None,
                severity=None,
                description="Attacker campaign already completed",
                progress_delta=0.0,
                detection_strength=0.0,
                compromise_confirmed=False,
                exfiltration_succeeded=True,
            )

        if self._next_step_index >= len(self.current_path.steps):
            switched = self._switch_path(world)
            if not switched:
                return AttackerEvent(
                    attempted=False,
                    success=False,
                    blocked=False,
                    path_id=self.current_path.path_id,
                    step_id=None,
                    stage=None,
                    target=None,
                    source=None,
                    severity=None,
                    description="Attacker paused to reassess after terminal path",
                    progress_delta=0.0,
                    detection_strength=0.0,
                    compromise_confirmed=False,
                    exfiltration_succeeded=False,
                )

        step = self.current_step
        if step is None:
            return AttackerEvent(
                attempted=False,
                success=False,
                blocked=False,
                path_id=self.current_path.path_id,
                step_id=None,
                stage=None,
                target=None,
                source=None,
                severity=None,
                description="Attacker stalled with no executable step",
                progress_delta=0.0,
                detection_strength=0.0,
                compromise_confirmed=False,
                exfiltration_succeeded=False,
            )

        if not self._prerequisites_satisfied(step, world):
            self._stall_counter += 1
            if self._stall_counter >= 2:
                self._switch_path(world)
            return AttackerEvent(
                attempted=False,
                success=False,
                blocked=True,
                path_id=self.current_path.path_id,
                step_id=step.step_id,
                stage=step.stage,
                target=step.target_id,
                source=step.source,
                severity=step.severity,
                description=(
                    f"Prerequisites for attacker step {step.step_id} not satisfied"
                    " due to defender controls"
                ),
                progress_delta=0.0,
                detection_strength=step.detection_strength * 0.6,
                compromise_confirmed=False,
                exfiltration_succeeded=False,
            )

        if self._is_blocked_by_controls(step, world):
            self._failed_recently = True
            self._stall_counter += 1
            world.blocked_attempts += 1
            if self._stall_counter >= 2:
                self._switch_path(world)
            return AttackerEvent(
                attempted=True,
                success=False,
                blocked=True,
                path_id=self.current_path.path_id,
                step_id=step.step_id,
                stage=step.stage,
                target=step.target_id,
                source=step.source,
                severity=step.severity,
                description=(f"Defender control blocked attacker step {step.step_id}"),
                progress_delta=0.0,
                detection_strength=min(1.0, step.detection_strength + 0.15),
                compromise_confirmed=False,
                exfiltration_succeeded=False,
            )

        success_prob = self._effective_success_probability(step, world)
        rolled = self._rng.random()
        success = rolled <= success_prob
        if not success:
            self._failed_recently = True
            self._stall_counter += 1
            if self._stall_counter >= 3:
                self._switch_path(world)
            return AttackerEvent(
                attempted=True,
                success=False,
                blocked=False,
                path_id=self.current_path.path_id,
                step_id=step.step_id,
                stage=step.stage,
                target=step.target_id,
                source=step.source,
                severity=step.severity,
                description=(
                    f"Attacker attempted {step.step_id} but failed due to operational"
                    " instability"
                ),
                progress_delta=0.03,
                detection_strength=min(1.0, step.detection_strength + 0.08),
                compromise_confirmed=False,
                exfiltration_succeeded=False,
            )

        compromise_confirmed = self._apply_compromise(step, world)
        progress_delta = max(0.02, step.progress_weight / max(1.0, world.max_progress))
        world.attacker_progress = min(
            world.max_progress, world.attacker_progress + step.progress_weight
        )

        self._next_step_index += 1
        self._stall_counter = 0
        self._failed_recently = False

        exfiltration = bool(step.exfiltration_step)
        if exfiltration:
            world.exfiltration_succeeded = True

        return AttackerEvent(
            attempted=True,
            success=True,
            blocked=False,
            path_id=self.current_path.path_id,
            step_id=step.step_id,
            stage=step.stage,
            target=step.target_id,
            source=step.source,
            severity=step.severity,
            description=f"Attacker completed {step.step_id}: {step.description}",
            progress_delta=progress_delta,
            detection_strength=step.detection_strength,
            compromise_confirmed=compromise_confirmed,
            exfiltration_succeeded=exfiltration,
        )

    @property
    def current_path(self) -> AttackPathDefinition:
        return self._scenario.attack_paths[self._active_path_index]

    @property
    def current_step(self) -> Optional[AttackStepDefinition]:
        if self._next_step_index >= len(self.current_path.steps):
            return None
        return self.current_path.steps[self._next_step_index]

    def campaign_summary(self) -> dict:
        return {
            "active_path": self.current_path.path_id,
            "active_path_label": self.current_path.label,
            "next_step_index": self._next_step_index,
            "path_count": len(self._scenario.attack_paths),
        }

    def _effective_success_probability(
        self,
        step: AttackStepDefinition,
        world: WorldState,
    ) -> float:
        probability = step.success_probability
        probability -= 0.20 if self._failed_recently else 0.0

        target_type = world.target_type(step.target_id)
        if target_type == "asset":
            asset = world.assets[step.target_id]
            if asset.patched:
                probability -= step.patch_resistance
            if asset.isolated:
                probability -= 0.35
            if asset.egress_blocked and step.exfiltration_step:
                probability -= 0.50
        elif target_type == "identity":
            identity = world.identities[step.target_id]
            if identity.revoked:
                probability -= 0.65
            if (
                identity.secret_rotated_at is not None
                and identity.secret_rotated_at >= world.tick - 2
            ):
                probability -= 0.45

        evidence = world.evidence_for(step.target_id)
        probability -= 0.20 * evidence
        return _clamp(probability, 0.05, 0.98)

    def _prerequisites_satisfied(
        self, step: AttackStepDefinition, world: WorldState
    ) -> bool:
        for asset_id in step.required_assets:
            asset = world.assets.get(asset_id)
            if asset is None:
                return False
            if not asset.compromised:
                return False
            if asset.isolated:
                return False

        for identity_id in step.required_identities:
            identity = world.identities.get(identity_id)
            if identity is None:
                return False
            if not identity.compromised:
                return False
            if identity.revoked:
                return False

        return True

    def _is_blocked_by_controls(
        self, step: AttackStepDefinition, world: WorldState
    ) -> bool:
        target_type = world.target_type(step.target_id)
        if target_type == "asset":
            asset = world.assets[step.target_id]
            if asset.isolated and step.stage in {
                "lateral_movement",
                "collection",
                "exfiltration",
            }:
                return True
            if step.exfiltration_step and asset.egress_blocked:
                return True
            if asset.patched and step.stage in {"persistence", "privilege_escalation"}:
                return self._rng.random() < 0.75
        elif target_type == "identity":
            identity = world.identities[step.target_id]
            if identity.revoked:
                return True
            if (
                identity.secret_rotated_at is not None
                and identity.secret_rotated_at >= world.tick - 1
            ):
                return self._rng.random() < 0.65
        return False

    def _apply_compromise(self, step: AttackStepDefinition, world: WorldState) -> bool:
        changed = False
        if step.compromise_asset and step.target_id in world.assets:
            asset = world.assets[step.target_id]
            if not asset.compromised:
                asset.compromised = True
                changed = True
        if step.compromise_identity and step.target_id in world.identities:
            identity = world.identities[step.target_id]
            if not identity.compromised:
                identity.compromised = True
                changed = True
        return changed

    def _switch_path(self, world: WorldState) -> bool:
        alternatives = [
            idx
            for idx in range(len(self._scenario.attack_paths))
            if idx != self._active_path_index
        ]
        if not alternatives:
            return False

        next_index = self._rng.choice(alternatives)
        self._active_path_index = next_index
        self._next_step_index = 0
        self._stall_counter = 0
        world.path_switches += 1
        world.add_activity(
            f"Attacker switched campaign path to {self.current_path.path_id}"
        )
        return True


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return float(value)
