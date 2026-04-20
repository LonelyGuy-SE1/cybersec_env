"""Baseline defender policies for CybersecEnv benchmarking."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

try:
    from cybersec.models import CybersecAction, CybersecObservation
except ModuleNotFoundError:
    from models import CybersecAction, CybersecObservation


class DefenderPolicy(Protocol):
    """Policy interface used by benchmark runner."""

    name: str

    def reset_episode(self, observation: CybersecObservation) -> None:
        """Optional hook called after each environment reset."""

    def act(self, observation: CybersecObservation) -> CybersecAction:
        """Choose one action for the current observation."""


class RandomPolicy:
    """Uniform random baseline constrained by advertised valid targets."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def reset_episode(self, observation: CybersecObservation) -> None:
        del observation

    def act(self, observation: CybersecObservation) -> CybersecAction:
        available = observation.available_actions or ["MONITOR"]
        action_type = self._rng.choice(available)
        return _action_from_observation(
            observation=observation,
            action_type=action_type,
            rng=self._rng,
        )


@dataclass(frozen=True)
class HeuristicConfig:
    triage_confidence_threshold: float = 0.45
    ticket_confidence_threshold: float = 0.55
    forensics_confidence_threshold: float = 0.50
    prioritize_egress_blocking: bool = True
    max_open_tickets: int = 3
    max_forensics_per_episode: int = 5
    forensics_repeat_window: int = 4
    emergency_risk_threshold: float = 0.70
    emergency_alert_confidence: float = 0.78


@dataclass
class HeuristicState:
    blocked_assets: set[str] = field(default_factory=set)
    blocked_egress_assets: set[str] = field(default_factory=set)
    revoked_identities: set[str] = field(default_factory=set)
    forensics_requests: Dict[str, int] = field(default_factory=dict)
    forensics_count: int = 0


class HeuristicPolicy:
    """Structured rule-based policy for deterministic baseline comparison."""

    name = "heuristic"

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self._cfg = config or HeuristicConfig()
        self._state = HeuristicState()

    def reset_episode(self, observation: CybersecObservation) -> None:
        del observation
        self._state = HeuristicState()

    def act(self, observation: CybersecObservation) -> CybersecAction:
        valid = observation.valid_targets
        self._refresh_state_from_observation(observation)

        ready_tickets = list(valid.get("ready_ticket_ids", []))
        if ready_tickets:
            return CybersecAction(action_type="EXECUTE_TICKET", target=ready_tickets[0])

        emergency_action = self._emergency_control(observation)
        if emergency_action is not None:
            return emergency_action

        pending_alerts = sorted(
            [alert for alert in observation.alerts if alert.triage_status == "pending"],
            key=lambda alert: (alert.confidence, _severity_rank(alert.severity)),
            reverse=True,
        )

        forensic_target = self._select_forensics_target(observation, pending_alerts)
        if forensic_target is not None:
            return CybersecAction(
                action_type="REQUEST_FORENSICS", target=forensic_target
            )

        for alert in pending_alerts:
            if (
                alert.confidence >= self._cfg.triage_confidence_threshold
                or alert.severity in {"high", "critical"}
            ):
                return CybersecAction(action_type="TRIAGE_ALERT", target=alert.alert_id)

        known_assets = list(observation.known_compromised_assets)
        known_ids = list(observation.known_compromised_identities)

        if known_assets:
            top_asset = known_assets[0]

            if (
                self._cfg.prioritize_egress_blocking
                and "edge-egress" in valid.get("assets", [])
                and "edge-egress" not in self._state.blocked_egress_assets
                and not _has_open_ticket_for_target(observation, "edge-egress")
            ):
                return CybersecAction(
                    action_type="OPEN_TICKET",
                    target="edge-egress",
                    parameter="BLOCK_EGRESS",
                    urgency="high",
                )

            if (
                len(observation.open_tickets) < self._cfg.max_open_tickets
                and top_asset not in self._state.blocked_assets
                and not _has_open_ticket_for_target(observation, top_asset)
            ):
                return CybersecAction(
                    action_type="OPEN_TICKET",
                    target=top_asset,
                    parameter="ISOLATE_ASSET",
                    urgency="high",
                )

            if top_asset in valid.get("assets", []) and _risk_is_high(observation):
                if top_asset in self._state.blocked_assets:
                    return CybersecAction(action_type="PATCH_ASSET", target=top_asset)
                return CybersecAction(action_type="ISOLATE_ASSET", target=top_asset)

        if known_ids:
            top_identity = known_ids[0]
            if top_identity in valid.get("identities", []):
                if top_identity in self._state.revoked_identities:
                    return CybersecAction(
                        action_type="ROTATE_SECRET", target=top_identity
                    )
                if _risk_is_high(observation):
                    return CybersecAction(
                        action_type="REVOKE_IDENTITY", target=top_identity
                    )
                return CybersecAction(action_type="ROTATE_SECRET", target=top_identity)

        high_signal_entity = _highest_signal_entity(observation)
        if high_signal_entity is not None:
            all_targets = set(valid.get("assets", [])) | set(
                valid.get("identities", [])
            )
            if high_signal_entity in all_targets:
                return CybersecAction(
                    action_type="REQUEST_FORENSICS", target=high_signal_entity
                )

        if _risk_is_rising(observation):
            source = _best_log_source(observation)
            return CybersecAction(action_type="QUERY_LOGS", parameter=source)

        return CybersecAction(action_type="MONITOR")

    def _select_forensics_target(
        self,
        observation: CybersecObservation,
        pending_alerts: List,
    ) -> Optional[str]:
        if self._state.forensics_count >= self._cfg.max_forensics_per_episode:
            return None

        valid_targets = set(observation.valid_targets.get("assets", [])) | set(
            observation.valid_targets.get("identities", [])
        )
        if not valid_targets:
            return None

        for alert in pending_alerts:
            if alert.entity not in valid_targets:
                continue
            if alert.confidence < self._cfg.forensics_confidence_threshold:
                continue
            last_tick = self._state.forensics_requests.get(alert.entity)
            if (
                last_tick is not None
                and observation.tick - last_tick <= self._cfg.forensics_repeat_window
            ):
                continue
            self._state.forensics_requests[alert.entity] = observation.tick
            self._state.forensics_count += 1
            return alert.entity
        return None

    def _emergency_control(
        self, observation: CybersecObservation
    ) -> Optional[CybersecAction]:
        risk = observation.enterprise_risk_score
        if risk < self._cfg.emergency_risk_threshold:
            return None

        valid = observation.valid_targets
        assets = set(valid.get("assets", []))
        identities = set(valid.get("identities", []))

        high_alerts = sorted(
            [
                alert
                for alert in observation.alerts
                if alert.confidence >= self._cfg.emergency_alert_confidence
            ],
            key=lambda alert: (alert.confidence, _severity_rank(alert.severity)),
            reverse=True,
        )

        if (
            "edge-egress" in assets
            and "edge-egress" not in self._state.blocked_egress_assets
        ):
            if not _has_open_ticket_for_target(observation, "edge-egress"):
                return CybersecAction(action_type="BLOCK_EGRESS", target="edge-egress")

        for alert in high_alerts:
            if (
                alert.entity in assets
                and alert.entity not in self._state.blocked_assets
            ):
                return CybersecAction(action_type="ISOLATE_ASSET", target=alert.entity)
            if (
                alert.entity in identities
                and alert.entity not in self._state.revoked_identities
            ):
                return CybersecAction(
                    action_type="REVOKE_IDENTITY", target=alert.entity
                )

        for compromised in observation.known_compromised_assets:
            if compromised in assets and compromised not in self._state.blocked_assets:
                return CybersecAction(action_type="ISOLATE_ASSET", target=compromised)
        for compromised in observation.known_compromised_identities:
            if (
                compromised in identities
                and compromised not in self._state.revoked_identities
            ):
                return CybersecAction(action_type="REVOKE_IDENTITY", target=compromised)

        return None

    def _refresh_state_from_observation(self, observation: CybersecObservation) -> None:
        for ticket in observation.open_tickets:
            if ticket.status != "executed":
                continue
            if ticket.requested_action == "ISOLATE_ASSET":
                self._state.blocked_assets.add(ticket.target)
            elif ticket.requested_action == "BLOCK_EGRESS":
                self._state.blocked_egress_assets.add(ticket.target)
            elif ticket.requested_action == "REVOKE_IDENTITY":
                self._state.revoked_identities.add(ticket.target)

        for event in observation.recent_activity:
            lowered = event.lower()
            direct = _parse_direct_control_event(event)
            if direct is not None:
                action_type, target = direct
                if action_type == "ISOLATE_ASSET":
                    self._state.blocked_assets.add(target)
                elif action_type == "BLOCK_EGRESS":
                    self._state.blocked_egress_assets.add(target)
                elif action_type == "REVOKE_IDENTITY":
                    self._state.revoked_identities.add(target)

            if "outbound traffic blocked on" in lowered:
                target = _extract_last_token(event)
                if target:
                    self._state.blocked_egress_assets.add(target)
            elif "isolated from production network" in lowered:
                target = _extract_target_from_phrase(event, "Asset ", " isolated")
                if target:
                    self._state.blocked_assets.add(target)
            elif "Identity " in event and " revoked" in event:
                target = _extract_target_from_phrase(event, "Identity ", " revoked")
                if target:
                    self._state.revoked_identities.add(target)


def _action_from_observation(
    *,
    observation: CybersecObservation,
    action_type: str,
    rng: random.Random,
) -> CybersecAction:
    valid = observation.valid_targets

    if action_type == "MONITOR":
        return CybersecAction(action_type="MONITOR")

    if action_type == "QUERY_LOGS":
        sources = valid.get("query_log_sources", [])
        source = rng.choice(sources) if sources else "cloud"
        return CybersecAction(action_type="QUERY_LOGS", parameter=source)

    if action_type == "OPEN_TICKET":
        candidates = valid.get("assets", []) + valid.get("identities", [])
        target = rng.choice(candidates) if candidates else "edge-egress"
        actions = valid.get("ticketable_actions", [])
        parameter = rng.choice(actions) if actions else "ISOLATE_ASSET"
        urgency = rng.choice(["low", "normal", "high"])
        return CybersecAction(
            action_type="OPEN_TICKET",
            target=target,
            parameter=parameter,
            urgency=urgency,
        )

    if action_type == "EXECUTE_TICKET":
        ticket_ids = valid.get("ticket_ids", [])
        if ticket_ids:
            return CybersecAction(
                action_type="EXECUTE_TICKET", target=rng.choice(ticket_ids)
            )
        return CybersecAction(action_type="MONITOR")

    if action_type == "TRIAGE_ALERT":
        alert_ids = valid.get("alert_ids", [])
        if alert_ids:
            return CybersecAction(
                action_type="TRIAGE_ALERT", target=rng.choice(alert_ids)
            )
        return CybersecAction(action_type="MONITOR")

    if action_type == "REQUEST_FORENSICS":
        candidates = valid.get("assets", []) + valid.get("identities", [])
        if candidates:
            return CybersecAction(
                action_type="REQUEST_FORENSICS", target=rng.choice(candidates)
            )
        return CybersecAction(action_type="MONITOR")

    if action_type in {"ISOLATE_ASSET", "BLOCK_EGRESS", "PATCH_ASSET"}:
        assets = valid.get("assets", [])
        if assets:
            return CybersecAction(action_type=action_type, target=rng.choice(assets))
        return CybersecAction(action_type="MONITOR")

    if action_type in {"REVOKE_IDENTITY", "ROTATE_SECRET"}:
        ids = valid.get("identities", [])
        if ids:
            return CybersecAction(action_type=action_type, target=rng.choice(ids))
        return CybersecAction(action_type="MONITOR")

    return CybersecAction(action_type="MONITOR")


def _severity_rank(value: str) -> int:
    if value == "critical":
        return 4
    if value == "high":
        return 3
    if value == "medium":
        return 2
    return 1


def _risk_is_high(observation: CybersecObservation) -> bool:
    return observation.enterprise_risk_score >= 0.62


def _risk_is_rising(observation: CybersecObservation) -> bool:
    risk = observation.enterprise_risk_score
    if risk >= 0.50:
        return True
    if observation.alerts:
        high_alerts = [
            a for a in observation.alerts if a.severity in {"high", "critical"}
        ]
        return bool(high_alerts)
    return False


def _has_open_ticket_for_target(observation: CybersecObservation, target: str) -> bool:
    for ticket in observation.open_tickets:
        if ticket.target == target and ticket.status not in {"executed", "rejected"}:
            return True
    return False


def _highest_signal_entity(observation: CybersecObservation) -> Optional[str]:
    candidates = sorted(
        observation.alerts,
        key=lambda alert: (alert.confidence, _severity_rank(alert.severity)),
        reverse=True,
    )
    for alert in candidates:
        if alert.confidence >= 0.50:
            return alert.entity
    return None


def _best_log_source(observation: CybersecObservation) -> str:
    if any(alert.source == "network" for alert in observation.alerts):
        return "network"
    if any(alert.source == "cloud" for alert in observation.alerts):
        return "cloud"
    if any(alert.source == "identity" for alert in observation.alerts):
        return "identity"
    return "cloud"


def _extract_last_token(text: str) -> Optional[str]:
    parts = text.strip().split()
    if not parts:
        return None
    return parts[-1].strip().strip(".:")


def _extract_target_from_phrase(text: str, prefix: str, suffix: str) -> Optional[str]:
    start = text.find(prefix)
    if start < 0:
        return None
    start += len(prefix)
    end = text.find(suffix, start)
    if end < 0:
        return None
    target = text[start:end].strip()
    if not target:
        return None
    return target


def _parse_direct_control_event(text: str) -> Optional[tuple[str, str]]:
    prefix = "Executed direct control "
    marker = " on "
    if not text.startswith(prefix):
        return None
    body = text[len(prefix) :]
    idx = body.find(marker)
    if idx < 0:
        return None
    action_type = body[:idx].strip()
    target_body = body[idx + len(marker) :]
    end = target_body.find(" ")
    target = target_body if end < 0 else target_body[:end]
    target = target.strip().strip(".:")
    if not action_type or not target:
        return None
    return action_type, target
