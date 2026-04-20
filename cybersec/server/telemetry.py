"""Telemetry synthesis for partially observable CybersecEnv observations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Optional

from .attacker_policy import AttackerEvent
from .world_state import WorldState

Severity = Literal["low", "medium", "high", "critical"]


@dataclass
class AlertEvent:
    """Alert object emitted to the defender observation stream."""

    alert_id: str
    source: str
    severity: Severity
    confidence: float
    entity: str
    title: str
    hint: str
    linked_attacker_step: Optional[str]
    true_positive: bool
    tick: int
    triage_status: Literal["pending", "triaged"] = "pending"


@dataclass
class ForensicsJob:
    """Asynchronous forensic collection request."""

    job_id: str
    target: str
    target_type: Literal["asset", "identity"]
    requested_step: int
    ready_step: int
    confidence: float
    finding: str
    completed: bool = False


class TelemetryEngine:
    """Transforms latent world/attacker events into noisy defender telemetry."""

    def __init__(
        self, *, rng: random.Random, false_positive_rate: float, max_alerts: int
    ):
        self._rng = rng
        self._false_positive_rate = false_positive_rate
        self._max_alerts = max(12, max_alerts)
        self._alerts: List[AlertEvent] = []
        self._alert_counter = 0
        self._job_counter = 0
        self._forensics_jobs: List[ForensicsJob] = []

    def ingest_attacker_event(
        self, event: AttackerEvent, *, tick: int, world: WorldState
    ) -> List[AlertEvent]:
        generated: List[AlertEvent] = []
        if not event.attempted:
            return generated

        base_conf = 0.30 + 0.55 * max(0.0, min(1.0, event.detection_strength))
        if event.success:
            base_conf += 0.05
        if event.blocked:
            base_conf += 0.08

        emit_prob = min(0.95, 0.25 + event.detection_strength)
        if self._rng.random() <= emit_prob:
            entity = event.target or "unknown"
            confidence = _clamp(base_conf + self._noise(0.08), 0.05, 0.98)
            severity = event.severity or "medium"
            title = _title_for_event(event)
            hint = _hint_for_event(event)
            alert = self._create_alert(
                source=event.source or "endpoint",
                severity=severity,
                confidence=confidence,
                entity=entity,
                title=title,
                hint=hint,
                linked_step=event.step_id,
                true_positive=True,
                tick=tick,
            )
            generated.append(alert)
            world.adjust_evidence(entity, 0.22 + 0.20 * confidence)

        return generated

    def maybe_emit_background_noise(self, *, tick: int) -> List[AlertEvent]:
        generated: List[AlertEvent] = []
        if self._rng.random() > self._false_positive_rate:
            return generated

        source = self._rng.choice(["endpoint", "network", "cloud", "code"])  # noqa: S311
        severity = self._rng.choice(["low", "medium", "high"])  # noqa: S311
        entity = self._rng.choice(
            [
                "load-balancer",
                "ci-runner-a",
                "vpn-gateway",
                "repo-core",
                "helpdesk-portal",
            ]
        )
        confidence = _clamp(0.15 + self._rng.random() * 0.35, 0.05, 0.55)
        title = f"Anomalous {source} pattern"
        hint = "Signal may represent normal operational variance"
        generated.append(
            self._create_alert(
                source=source,
                severity=severity,
                confidence=confidence,
                entity=entity,
                title=title,
                hint=hint,
                linked_step=None,
                true_positive=False,
                tick=tick,
            )
        )
        return generated

    def list_alerts(self) -> List[AlertEvent]:
        return list(self._alerts)

    def get_alert(self, alert_id: str) -> Optional[AlertEvent]:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                return alert
        return None

    def triage_alert(self, alert_id: str, *, world: WorldState) -> tuple[bool, str]:
        alert = self.get_alert(alert_id)
        if alert is None:
            return False, f"Alert '{alert_id}' not found"
        if alert.triage_status == "triaged":
            return False, f"Alert {alert_id} already triaged"

        alert.triage_status = "triaged"
        if alert.true_positive:
            world.adjust_evidence(alert.entity, 0.18 + 0.22 * alert.confidence)
            return True, (
                f"Triage confirmed suspicious behavior on {alert.entity}"
                f" (confidence={alert.confidence:.2f})"
            )

        world.adjust_evidence(alert.entity, -0.04)
        return True, (f"Triage downgraded alert {alert_id}; likely benign activity")

    def create_forensics_job(
        self,
        *,
        target: str,
        target_type: Literal["asset", "identity"],
        tick: int,
        world: WorldState,
    ) -> ForensicsJob:
        self._job_counter += 1
        job_id = f"FRX-{self._job_counter:04d}"
        delay = 2 if target_type == "identity" else 3

        compromised = world.is_compromised(target)
        if compromised:
            confidence = _clamp(0.65 + self._rng.random() * 0.30, 0.65, 0.98)
            finding = "Forensics found confirmed compromise artifacts"
        else:
            confidence = _clamp(0.20 + self._rng.random() * 0.35, 0.15, 0.60)
            finding = "Forensics found weak or inconclusive compromise evidence"

        job = ForensicsJob(
            job_id=job_id,
            target=target,
            target_type=target_type,
            requested_step=tick,
            ready_step=tick + delay,
            confidence=confidence,
            finding=finding,
            completed=False,
        )
        self._forensics_jobs.append(job)
        return job

    def collect_ready_forensics(
        self, *, tick: int, world: WorldState
    ) -> List[ForensicsJob]:
        ready: List[ForensicsJob] = []
        for job in self._forensics_jobs:
            if job.completed:
                continue
            if tick < job.ready_step:
                continue
            job.completed = True
            ready.append(job)
            if world.is_compromised(job.target):
                world.adjust_evidence(job.target, 0.25 + 0.20 * job.confidence)
                world.register_discovery(job.target)
            else:
                world.adjust_evidence(job.target, -0.03)
        return ready

    def _create_alert(
        self,
        *,
        source: str,
        severity: Severity,
        confidence: float,
        entity: str,
        title: str,
        hint: str,
        linked_step: Optional[str],
        true_positive: bool,
        tick: int,
    ) -> AlertEvent:
        self._alert_counter += 1
        alert = AlertEvent(
            alert_id=f"ALT-{self._alert_counter:04d}",
            source=source,
            severity=severity,
            confidence=confidence,
            entity=entity,
            title=title,
            hint=hint,
            linked_attacker_step=linked_step,
            true_positive=true_positive,
            tick=tick,
        )
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]
        return alert

    def _noise(self, scale: float) -> float:
        return (self._rng.random() * 2.0 - 1.0) * scale


def _title_for_event(event: AttackerEvent) -> str:
    stage = event.stage or "unknown-stage"
    return f"{stage.replace('_', ' ').title()} anomaly"


def _hint_for_event(event: AttackerEvent) -> str:
    if event.blocked:
        return "Defender controls may have disrupted this activity"
    if event.success:
        return "Correlated indicators suggest attacker progression"
    return "Transient activity detected; additional validation advised"


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return float(value)
