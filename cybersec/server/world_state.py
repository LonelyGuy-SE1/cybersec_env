"""Runtime world state for CybersecEnv campaign simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from .scenario_loader import ScenarioDefinition

ControlAction = Literal[
    "ISOLATE_ASSET",
    "REVOKE_IDENTITY",
    "ROTATE_SECRET",
    "BLOCK_EGRESS",
    "PATCH_ASSET",
]

Criticality = Literal["low", "medium", "high", "critical"]
Privilege = Literal["user", "service", "admin"]


@dataclass
class AssetRuntime:
    asset_id: str
    domain: str
    criticality: Criticality
    description: str
    egress_point: bool = False
    compromised: bool = False
    isolated: bool = False
    patched: bool = False
    egress_blocked: bool = False
    remediated: bool = False
    secret_rotated_at: Optional[int] = None


@dataclass
class IdentityRuntime:
    identity_id: str
    privilege: Privilege
    role: str
    compromised: bool = False
    revoked: bool = False
    secret_rotated_at: Optional[int] = None


@dataclass
class ControlApplication:
    action_type: ControlAction
    target: str
    success: bool
    message: str
    disruption_cost: float
    containment_gain: float
    target_was_compromised: bool
    via_ticket: bool


@dataclass
class WorldState:
    scenario: ScenarioDefinition
    assets: Dict[str, AssetRuntime]
    identities: Dict[str, IdentityRuntime]
    tick: int = 0
    attacker_progress: float = 0.0
    max_progress: float = 1.0
    exfiltration_succeeded: bool = False
    known_compromised_assets: Set[str] = field(default_factory=set)
    known_compromised_identities: Set[str] = field(default_factory=set)
    entity_evidence: Dict[str, float] = field(default_factory=dict)
    recent_activity: List[str] = field(default_factory=list)
    path_switches: int = 0
    blocked_attempts: int = 0

    @classmethod
    def from_scenario(cls, scenario: ScenarioDefinition) -> "WorldState":
        assets = {
            a.asset_id: AssetRuntime(
                asset_id=a.asset_id,
                domain=a.domain,
                criticality=a.criticality,
                description=a.description,
                egress_point=a.egress_point,
            )
            for a in scenario.assets
        }
        identities = {
            i.identity_id: IdentityRuntime(
                identity_id=i.identity_id,
                privilege=i.privilege,
                role=i.role,
            )
            for i in scenario.identities
        }
        max_progress = 1.0
        if scenario.attack_paths:
            max_progress = max(
                1.0,
                max(
                    sum(step.progress_weight for step in path.steps)
                    for path in scenario.attack_paths
                ),
            )
        return cls(
            scenario=scenario,
            assets=assets,
            identities=identities,
            max_progress=max_progress,
        )

    def add_activity(self, message: str) -> None:
        self.recent_activity.append(message)
        if len(self.recent_activity) > 24:
            self.recent_activity = self.recent_activity[-24:]

    def get_recent_activity(self, limit: int = 10) -> List[str]:
        if limit <= 0:
            return []
        return self.recent_activity[-limit:]

    def has_asset(self, asset_id: str) -> bool:
        return asset_id in self.assets

    def has_identity(self, identity_id: str) -> bool:
        return identity_id in self.identities

    def target_type(self, target: str) -> Optional[Literal["asset", "identity"]]:
        if target in self.assets:
            return "asset"
        if target in self.identities:
            return "identity"
        return None

    def is_compromised(self, target: str) -> bool:
        if target in self.assets:
            return self.assets[target].compromised
        if target in self.identities:
            return self.identities[target].compromised
        return False

    def is_asset_contained(self, asset_id: str) -> bool:
        asset = self.assets.get(asset_id)
        if asset is None:
            return True
        if not asset.compromised:
            return True
        return asset.isolated or asset.remediated

    def is_identity_contained(self, identity_id: str) -> bool:
        identity = self.identities.get(identity_id)
        if identity is None:
            return True
        if not identity.compromised:
            return True
        return identity.revoked

    def evidence_for(self, target: str) -> float:
        return float(self.entity_evidence.get(target, 0.0))

    def adjust_evidence(self, target: str, delta: float) -> float:
        current = self.entity_evidence.get(target, 0.0)
        updated = _clamp01(current + delta)
        self.entity_evidence[target] = updated
        return updated

    def register_discovery(self, target: str) -> bool:
        if target in self.assets:
            if target not in self.known_compromised_assets:
                self.known_compromised_assets.add(target)
                return True
            return False
        if target in self.identities:
            if target not in self.known_compromised_identities:
                self.known_compromised_identities.add(target)
                return True
            return False
        return False

    def clear_compromise(self, target: str) -> None:
        if target in self.assets:
            self.assets[target].compromised = False
            self.assets[target].remediated = True
            self.known_compromised_assets.discard(target)
            return
        if target in self.identities:
            self.identities[target].compromised = False
            self.known_compromised_identities.discard(target)

    def apply_control(
        self,
        action_type: ControlAction,
        target: str,
        *,
        tick: int,
        via_ticket: bool,
    ) -> ControlApplication:
        if action_type in {"ISOLATE_ASSET", "BLOCK_EGRESS", "PATCH_ASSET"}:
            asset = self.assets.get(target)
            if asset is None:
                return ControlApplication(
                    action_type=action_type,
                    target=target,
                    success=False,
                    message=f"Target asset '{target}' not found",
                    disruption_cost=0.0,
                    containment_gain=0.0,
                    target_was_compromised=False,
                    via_ticket=via_ticket,
                )
            return self._apply_asset_control(
                action_type, asset, tick=tick, via_ticket=via_ticket
            )

        identity = self.identities.get(target)
        if identity is None:
            return ControlApplication(
                action_type=action_type,
                target=target,
                success=False,
                message=f"Target identity '{target}' not found",
                disruption_cost=0.0,
                containment_gain=0.0,
                target_was_compromised=False,
                via_ticket=via_ticket,
            )
        return self._apply_identity_control(
            action_type, identity, tick=tick, via_ticket=via_ticket
        )

    def _apply_asset_control(
        self,
        action_type: ControlAction,
        asset: AssetRuntime,
        *,
        tick: int,
        via_ticket: bool,
    ) -> ControlApplication:
        compromised = asset.compromised

        if action_type == "ISOLATE_ASSET":
            if asset.isolated:
                return self._noop_control(
                    action_type, asset.asset_id, via_ticket, "Asset already isolated"
                )
            asset.isolated = True
            disruption = 0.16 * _criticality_weight(asset.criticality)
            containment_gain = 0.55 if compromised else 0.05
            return ControlApplication(
                action_type=action_type,
                target=asset.asset_id,
                success=True,
                message=f"Asset {asset.asset_id} isolated from production network",
                disruption_cost=disruption,
                containment_gain=containment_gain,
                target_was_compromised=compromised,
                via_ticket=via_ticket,
            )

        if action_type == "BLOCK_EGRESS":
            if not asset.egress_point:
                return self._noop_control(
                    action_type,
                    asset.asset_id,
                    via_ticket,
                    "Asset is not configured as an egress control point",
                )
            if asset.egress_blocked:
                return self._noop_control(
                    action_type, asset.asset_id, via_ticket, "Egress already blocked"
                )
            asset.egress_blocked = True
            disruption = 0.12 * _criticality_weight(asset.criticality)
            containment_gain = 0.45 if compromised else 0.10
            return ControlApplication(
                action_type=action_type,
                target=asset.asset_id,
                success=True,
                message=f"Outbound traffic blocked on {asset.asset_id}",
                disruption_cost=disruption,
                containment_gain=containment_gain,
                target_was_compromised=compromised,
                via_ticket=via_ticket,
            )

        # PATCH_ASSET
        if asset.patched and asset.remediated:
            return self._noop_control(
                action_type,
                asset.asset_id,
                via_ticket,
                "Asset already patched and remediated",
            )
        asset.patched = True
        disruption = 0.10 * _criticality_weight(asset.criticality)
        containment_gain = 0.20
        message = f"Patch rollout applied on {asset.asset_id}"
        if compromised and asset.isolated:
            asset.compromised = False
            asset.remediated = True
            self.known_compromised_assets.discard(asset.asset_id)
            containment_gain = 0.75
            message = f"Patch and clean-up completed on isolated asset {asset.asset_id}"
        elif compromised:
            message = (
                f"Patch applied on {asset.asset_id}; compromise not fully removed"
                " because asset remains online"
            )

        return ControlApplication(
            action_type=action_type,
            target=asset.asset_id,
            success=True,
            message=message,
            disruption_cost=disruption,
            containment_gain=containment_gain,
            target_was_compromised=compromised,
            via_ticket=via_ticket,
        )

    def _apply_identity_control(
        self,
        action_type: ControlAction,
        identity: IdentityRuntime,
        *,
        tick: int,
        via_ticket: bool,
    ) -> ControlApplication:
        compromised = identity.compromised

        if action_type == "REVOKE_IDENTITY":
            if identity.revoked:
                return self._noop_control(
                    action_type,
                    identity.identity_id,
                    via_ticket,
                    "Identity already revoked",
                )
            identity.revoked = True
            disruption = 0.11 * _privilege_weight(identity.privilege)
            containment_gain = 0.65 if compromised else 0.08
            if compromised:
                identity.compromised = False
                self.known_compromised_identities.discard(identity.identity_id)
            return ControlApplication(
                action_type=action_type,
                target=identity.identity_id,
                success=True,
                message=f"Identity {identity.identity_id} revoked",
                disruption_cost=disruption,
                containment_gain=containment_gain,
                target_was_compromised=compromised,
                via_ticket=via_ticket,
            )

        # ROTATE_SECRET
        identity.secret_rotated_at = tick
        disruption = 0.06 * _privilege_weight(identity.privilege)
        containment_gain = 0.30 if compromised else 0.06
        if compromised:
            identity.compromised = False
            self.known_compromised_identities.discard(identity.identity_id)
        return ControlApplication(
            action_type="ROTATE_SECRET",
            target=identity.identity_id,
            success=True,
            message=f"Credential material rotated for {identity.identity_id}",
            disruption_cost=disruption,
            containment_gain=containment_gain,
            target_was_compromised=compromised,
            via_ticket=via_ticket,
        )

    @staticmethod
    def _noop_control(
        action_type: ControlAction,
        target: str,
        via_ticket: bool,
        message: str,
    ) -> ControlApplication:
        return ControlApplication(
            action_type=action_type,
            target=target,
            success=False,
            message=message,
            disruption_cost=0.0,
            containment_gain=0.0,
            target_was_compromised=False,
            via_ticket=via_ticket,
        )

    def compromised_assets(self) -> List[str]:
        return [
            asset.asset_id
            for asset in self.assets.values()
            if asset.compromised and not self.is_asset_contained(asset.asset_id)
        ]

    def compromised_identities(self) -> List[str]:
        return [
            identity.identity_id
            for identity in self.identities.values()
            if identity.compromised
            and not self.is_identity_contained(identity.identity_id)
        ]

    def enterprise_risk_score(self) -> float:
        progress_term = self.attacker_progress / max(1.0, self.max_progress)
        open_asset_compromise = 0.0
        critical_pool = 0
        for asset in self.assets.values():
            if asset.criticality in {"high", "critical"}:
                critical_pool += 1
                if asset.compromised and not self.is_asset_contained(asset.asset_id):
                    open_asset_compromise += 1.0
        asset_term = (
            (open_asset_compromise / critical_pool) if critical_pool > 0 else 0.0
        )

        identity_term = 0.0
        admin_total = 0
        for identity in self.identities.values():
            if identity.privilege == "admin":
                admin_total += 1
                if identity.compromised and not self.is_identity_contained(
                    identity.identity_id
                ):
                    identity_term += 1.0
        if admin_total > 0:
            identity_term /= admin_total

        score = 0.12
        score += 0.36 * _clamp01(progress_term)
        score += 0.30 * _clamp01(asset_term)
        score += 0.16 * _clamp01(identity_term)
        if self.exfiltration_succeeded:
            score += 0.45
        return _clamp01(score)

    def protected_focus_ratio(self) -> float:
        focus = [a for a in self.scenario.evaluation_focus_assets if a in self.assets]
        if not focus:
            return 1.0
        protected = 0
        for asset_id in focus:
            asset = self.assets[asset_id]
            if (not asset.compromised) or self.is_asset_contained(asset_id):
                protected += 1
        return protected / len(focus)

    def target_catalog(self) -> Dict[str, List[str]]:
        return {
            "assets": sorted(self.assets.keys()),
            "identities": sorted(self.identities.keys()),
            "known_compromised_assets": sorted(self.known_compromised_assets),
            "known_compromised_identities": sorted(self.known_compromised_identities),
        }


def _criticality_weight(level: Criticality) -> float:
    if level == "low":
        return 0.7
    if level == "medium":
        return 1.0
    if level == "high":
        return 1.3
    return 1.6


def _privilege_weight(privilege: Privilege) -> float:
    if privilege == "user":
        return 0.8
    if privilege == "service":
        return 1.0
    return 1.4


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
