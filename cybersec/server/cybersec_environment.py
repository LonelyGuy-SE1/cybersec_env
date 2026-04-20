"""CybersecEnv multi-agent enterprise campaign environment."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import (
        ActionType,
        CybersecAction,
        CybersecObservation,
        CybersecRewardBreakdown,
        ForensicsUpdate,
        SecurityAlert,
        WorkflowTicket,
    )
except ImportError:
    from models import (
        ActionType,
        CybersecAction,
        CybersecObservation,
        CybersecRewardBreakdown,
        ForensicsUpdate,
        SecurityAlert,
        WorkflowTicket,
    )

from .attacker_policy import AttackerEvent, AttackerPolicy
from .reward_model import RewardModel, StepSignals
from .scenario_loader import get_scenario, list_scenarios
from .telemetry import ForensicsJob, TelemetryEngine
from .workflow import TicketRecord, WorkflowEngine
from .world_state import ControlApplication, WorldState

LogSource = str


class CybersecEnvironment(Environment[CybersecAction, CybersecObservation, State]):
    """Long-horizon cyber defense environment with hidden attacker dynamics."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    _ACTIONS: List[ActionType] = [
        "MONITOR",
        "QUERY_LOGS",
        "TRIAGE_ALERT",
        "REQUEST_FORENSICS",
        "OPEN_TICKET",
        "EXECUTE_TICKET",
        "ISOLATE_ASSET",
        "REVOKE_IDENTITY",
        "ROTATE_SECRET",
        "BLOCK_EGRESS",
        "PATCH_ASSET",
    ]

    def __init__(
        self,
        scenario_id: str = "supply_chain_token_drift",
        *,
        seed: Optional[int] = None,
        horizon: Optional[int] = None,
        false_positive_rate: Optional[float] = None,
    ):
        self._scenario_id = scenario_id
        self._default_seed = seed
        self._override_horizon = horizon
        self._override_false_positive_rate = false_positive_rate

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(seed)

        self._world: Optional[WorldState] = None
        self._telemetry: Optional[TelemetryEngine] = None
        self._workflow: Optional[WorkflowEngine] = None
        self._attacker: Optional[AttackerPolicy] = None
        self._reward_model: Optional[RewardModel] = None

        self._horizon = 0
        self._done = False
        self._terminal_reason = ""

        self._last_reward_breakdown: Dict[str, float] = {}
        self._invalid_action: Optional[str] = None
        self._step_events: List[str] = []
        self._attacker_summary: Dict[str, Any] = {}

    @classmethod
    def scenario_ids(cls) -> tuple[str, ...]:
        return list_scenarios()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        horizon: Optional[int] = None,
        **kwargs,
    ) -> CybersecObservation:
        del kwargs

        chosen_scenario_id = scenario_id or self._scenario_id
        scenario = get_scenario(chosen_scenario_id)
        self._scenario_id = chosen_scenario_id

        effective_seed = seed
        if effective_seed is None:
            effective_seed = self._default_seed
        if effective_seed is None:
            effective_seed = random.SystemRandom().randint(1, 10_000_000)

        self._rng = random.Random(effective_seed)

        self._world = WorldState.from_scenario(scenario)
        self._workflow = WorkflowEngine(rng=self._rng)

        false_positive_rate = (
            scenario.false_positive_rate
            if self._override_false_positive_rate is None
            else self._override_false_positive_rate
        )
        self._telemetry = TelemetryEngine(
            rng=self._rng,
            false_positive_rate=false_positive_rate,
            max_alerts=scenario.max_alerts,
        )
        self._attacker = AttackerPolicy(scenario=scenario, rng=self._rng)

        self._horizon = horizon or self._override_horizon or scenario.horizon
        self._reward_model = RewardModel(horizon=self._horizon)

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._done = False
        self._terminal_reason = ""
        self._invalid_action = None
        self._step_events = [
            f"Scenario initialized: {scenario.scenario_id}",
            f"Objective: {scenario.objective}",
        ]
        self._last_reward_breakdown = {
            "detection": 0.0,
            "containment": 0.0,
            "business_continuity": 1.0,
            "governance": 1.0,
            "efficiency": 1.0,
            "disruption_penalty": 0.0,
            "governance_penalty": 0.0,
            "efficiency_penalty": 0.0,
            "adversary_pressure": 0.0,
            "terminal_outcome": 0.0,
            "false_positive_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "total": 0.0,
        }

        self._attacker_summary = self._attacker.campaign_summary()

        return self._build_observation(
            reward=None,
            done=False,
            info={
                "phase": "initial",
                "scenario_id": scenario.scenario_id,
                "scenario_title": scenario.title,
                "seed": effective_seed,
                "horizon": self._horizon,
                "available_actions": list(self._ACTIONS),
                "step_contract": "observation_reward_done_info",
            },
        )

    def step(
        self,
        action: CybersecAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CybersecObservation:
        del timeout_s, kwargs

        if (
            self._world is None
            or self._telemetry is None
            or self._workflow is None
            or self._attacker is None
            or self._reward_model is None
        ):
            obs = self.reset()
            obs.info = {
                **obs.info,
                "invalid_action": "step_called_before_reset_action_ignored",
                "last_action_error": "step_called_before_reset_action_ignored",
            }
            return obs

        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                info={
                    "phase": "terminal",
                    "termination_reason": self._terminal_reason,
                    "invalid_action": "episode_already_terminated_call_reset",
                    "last_action_error": "episode_already_terminated_call_reset",
                    "available_actions": [],
                    "reward_breakdown": self._last_reward_breakdown,
                },
            )

        self._state.step_count += 1
        self._world.tick = self._state.step_count
        self._invalid_action = None
        self._step_events = []

        signals = StepSignals()
        self._apply_defender_action(action=action, signals=signals)

        workflow_updates = self._workflow.update(
            tick=self._world.tick,
            evidence_by_target=dict(self._world.entity_evidence),
        )
        if workflow_updates:
            self._step_events.extend(update.message for update in workflow_updates)

        attacker_event = self._attacker.progress(self._world)
        self._attacker_summary = self._attacker.campaign_summary()
        self._handle_attacker_event(attacker_event=attacker_event, signals=signals)

        alerts_from_attack = self._telemetry.ingest_attacker_event(
            attacker_event,
            tick=self._world.tick,
            world=self._world,
        )
        noise_alerts = self._telemetry.maybe_emit_background_noise(
            tick=self._world.tick
        )

        if alerts_from_attack:
            self._step_events.append(
                f"{len(alerts_from_attack)} attacker-correlated alert(s) generated"
            )
        if noise_alerts:
            self._step_events.append(
                f"{len(noise_alerts)} background anomaly alert(s) generated"
            )

        forensics_updates = self._telemetry.collect_ready_forensics(
            tick=self._world.tick,
            world=self._world,
        )
        if forensics_updates:
            self._step_events.append(
                f"{len(forensics_updates)} forensics job(s) completed"
            )

        done = (
            self._state.step_count >= self._horizon
            or self._world.exfiltration_succeeded
        )
        if done:
            self._done = True
            self._terminal_reason = (
                "adversary_exfiltration"
                if self._world.exfiltration_succeeded
                else "horizon_reached"
            )

        reward, breakdown = self._reward_model.step_reward(
            world=self._world,
            signals=signals,
            done=done,
        )
        self._last_reward_breakdown = breakdown

        for event in self._step_events:
            self._world.add_activity(event)

        info = self._build_info(
            reward_breakdown=breakdown,
            done=done,
            workflow_updates=workflow_updates,
            forensics_updates=forensics_updates,
            attacker_event=attacker_event,
        )

        return self._build_observation(reward=reward, done=done, info=info)

    @property
    def state(self) -> State:
        scenario_id = self._world.scenario.scenario_id if self._world else None
        risk_score = self._world.enterprise_risk_score() if self._world else 0.0
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            done=self._done,
            scenario_id=scenario_id,
            horizon=self._horizon,
            enterprise_risk_score=risk_score,
            terminal_reason=self._terminal_reason,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="cybersec-long-horizon-enterprise-defense",
            description=(
                "Long-horizon, partially observable enterprise incident response"
                " environment with adaptive attacker dynamics and workflow-gated"
                " defender controls."
            ),
            version="1.0.0",
            author="cybersec_env",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
        )

    def _apply_defender_action(
        self, *, action: CybersecAction, signals: StepSignals
    ) -> None:
        assert self._world is not None
        assert self._telemetry is not None
        assert self._workflow is not None

        action_type = action.action_type

        if action_type == "MONITOR":
            signals.efficiency_cost += 0.01
            self._step_events.append("SOC remained in monitor mode")
            return

        if action_type == "QUERY_LOGS":
            source = action.parameter or ""
            gain = self._apply_log_query(source=source)
            signals.detection_gain += gain
            signals.efficiency_cost += 0.03
            self._step_events.append(
                f"Queried {source} logs and improved evidence discovery"
            )
            return

        if action_type == "TRIAGE_ALERT":
            alert_id = action.target or ""
            ok, message = self._telemetry.triage_alert(alert_id, world=self._world)
            if not ok:
                self._set_invalid(message, signals=signals, penalty=0.08)
                return
            alert = self._telemetry.get_alert(alert_id)
            if alert and alert.true_positive:
                signals.detection_gain += 0.12 + 0.10 * alert.confidence
            elif alert:
                signals.false_positive_penalty += 0.04
            signals.efficiency_cost += 0.02
            self._step_events.append(message)
            return

        if action_type == "REQUEST_FORENSICS":
            target = action.target or ""
            target_type = self._world.target_type(target)
            if target_type is None:
                self._set_invalid(
                    f"Unknown forensics target '{target}'",
                    signals=signals,
                    penalty=0.10,
                )
                return
            job = self._telemetry.create_forensics_job(
                target=target,
                target_type=target_type,
                tick=self._world.tick,
                world=self._world,
            )
            signals.efficiency_cost += 0.06
            self._step_events.append(
                f"Forensics requested on {target}; job {job.job_id} ready at step {job.ready_step}"
            )
            return

        if action_type == "OPEN_TICKET":
            target = action.target or ""
            requested_action = action.parameter or ""
            if requested_action not in {
                "ISOLATE_ASSET",
                "REVOKE_IDENTITY",
                "ROTATE_SECRET",
                "BLOCK_EGRESS",
                "PATCH_ASSET",
            }:
                self._set_invalid(
                    "OPEN_TICKET parameter must be a control action",
                    signals=signals,
                    penalty=0.10,
                )
                return
            if self._world.target_type(target) is None:
                self._set_invalid(
                    f"Ticket target '{target}' not found",
                    signals=signals,
                    penalty=0.08,
                )
                return
            evidence = self._world.evidence_for(target)
            ticket = self._workflow.open_ticket(
                tick=self._world.tick,
                requested_action=requested_action,
                target=target,
                urgency=action.urgency,
                target_evidence=evidence,
            )
            signals.governance_cost += 0.03
            signals.efficiency_cost += 0.04
            self._step_events.append(
                f"Opened ticket {ticket.ticket_id} for {requested_action} on {target}"
            )
            return

        if action_type == "EXECUTE_TICKET":
            ticket_id = action.target or ""
            ticket = self._workflow.get_ticket(ticket_id)
            if ticket is None:
                self._set_invalid(
                    f"Ticket '{ticket_id}' not found",
                    signals=signals,
                    penalty=0.10,
                )
                return
            if ticket.status != "ready":
                self._set_invalid(
                    f"Ticket {ticket_id} not executable in status '{ticket.status}'",
                    signals=signals,
                    penalty=0.08,
                )
                return
            result = self._world.apply_control(
                action_type=ticket.requested_action,
                target=ticket.target,
                tick=self._world.tick,
                via_ticket=True,
            )
            self._workflow.mark_executed(ticket_id, note=result.message)
            self._apply_control_signals(result=result, signals=signals)
            self._step_events.append(
                f"Executed ticket {ticket.ticket_id}: {result.message}"
            )
            return

        # Direct controls (governance penalty if executed outside workflow)
        target = action.target or ""
        result = self._world.apply_control(
            action_type=action_type,  # type: ignore[arg-type]
            target=target,
            tick=self._world.tick,
            via_ticket=False,
        )
        if not result.success:
            self._set_invalid(result.message, signals=signals, penalty=0.08)
            return

        self._apply_control_signals(result=result, signals=signals)
        signals.governance_cost += 0.08
        self._step_events.append(
            f"Executed direct control {action_type} on {target} (bypassed workflow)"
        )

    def _apply_log_query(self, *, source: LogSource) -> float:
        assert self._world is not None
        source_weights = {
            "identity": 0.12,
            "endpoint": 0.10,
            "network": 0.13,
            "code": 0.11,
            "cloud": 0.14,
            "ticketing": 0.07,
        }
        weight = source_weights.get(source, 0.08)

        gain = 0.0
        for asset in self._world.assets.values():
            if asset.compromised and not self._world.is_asset_contained(asset.asset_id):
                delta = weight * 0.45
                self._world.adjust_evidence(asset.asset_id, delta)
                gain += delta
        for identity in self._world.identities.values():
            if identity.compromised and not self._world.is_identity_contained(
                identity.identity_id
            ):
                delta = weight * 0.40
                self._world.adjust_evidence(identity.identity_id, delta)
                gain += delta
        return min(0.30, gain)

    def _apply_control_signals(
        self, *, result: ControlApplication, signals: StepSignals
    ) -> None:
        signals.containment_gain += result.containment_gain
        signals.disruption_cost += result.disruption_cost
        if result.target_was_compromised:
            self._world.register_discovery(result.target)
            signals.detection_gain += 0.10
        if result.via_ticket:
            signals.governance_cost += 0.01

    def _handle_attacker_event(
        self, *, attacker_event: AttackerEvent, signals: StepSignals
    ) -> None:
        assert self._world is not None
        if attacker_event.attempted:
            signals.adversary_progress_delta += attacker_event.progress_delta
            self._step_events.append(attacker_event.description)
            if attacker_event.blocked:
                signals.containment_gain += 0.08
        else:
            self._step_events.append(attacker_event.description)

    def _set_invalid(
        self, message: str, *, signals: StepSignals, penalty: float
    ) -> None:
        self._invalid_action = message
        signals.invalid_action_penalty += penalty
        self._step_events.append(message)

    def _build_info(
        self,
        *,
        reward_breakdown: Dict[str, float],
        done: bool,
        workflow_updates: List[Any],
        forensics_updates: List[ForensicsJob],
        attacker_event: AttackerEvent,
    ) -> Dict[str, Any]:
        assert self._world is not None
        assert self._reward_model is not None

        info: Dict[str, Any] = {
            "phase": "terminal" if done else "running",
            "scenario_id": self._world.scenario.scenario_id,
            "scenario_title": self._world.scenario.title,
            "available_actions": list(self._ACTIONS) if not done else [],
            "reward_breakdown": reward_breakdown,
            "enterprise_risk_score": self._world.enterprise_risk_score(),
            "attacker_summary": self._attacker_summary,
            "attacker_event": {
                "attempted": attacker_event.attempted,
                "success": attacker_event.success,
                "blocked": attacker_event.blocked,
                "path_id": attacker_event.path_id,
                "step_id": attacker_event.step_id,
                "stage": attacker_event.stage,
                "target": attacker_event.target,
                "source": attacker_event.source,
                "exfiltration_succeeded": attacker_event.exfiltration_succeeded,
            },
            "workflow_updates": [asdict(update) for update in workflow_updates],
            "forensics_updates": [
                {
                    "job_id": job.job_id,
                    "target": job.target,
                    "target_type": job.target_type,
                    "status": "completed",
                    "finding": job.finding,
                    "confidence": job.confidence,
                }
                for job in forensics_updates
            ],
            "events": self._world.get_recent_activity(limit=12),
            "step_contract": "observation_reward_done_info",
        }

        if self._invalid_action is not None:
            info["invalid_action"] = self._invalid_action
            info["last_action_error"] = self._invalid_action

        if done:
            terminal_score = self._reward_model.terminal_score(self._world)
            info["terminal_reason"] = self._terminal_reason
            info["grader_score"] = terminal_score
            info["grader_success"] = self._reward_model.grader_success(self._world)

        return info

    def _build_observation(
        self,
        *,
        reward: Optional[float],
        done: bool,
        info: Dict[str, Any],
    ) -> CybersecObservation:
        assert self._world is not None
        assert self._telemetry is not None
        assert self._workflow is not None

        alerts = [
            SecurityAlert(
                alert_id=alert.alert_id,
                source=alert.source,
                severity=alert.severity,
                confidence=alert.confidence,
                entity=alert.entity,
                title=alert.title,
                triage_status=alert.triage_status,
            )
            for alert in self._telemetry.list_alerts()
        ]
        open_tickets = [
            self._ticket_to_model(ticket) for ticket in self._workflow.visible_tickets()
        ]

        forensics_updates = [
            ForensicsUpdate(
                job_id=entry["job_id"],
                target=entry["target"],
                target_type=entry["target_type"],
                status=entry["status"],
                finding=entry["finding"],
                confidence=entry["confidence"],
            )
            for entry in info.get("forensics_updates", [])
        ]

        reward_breakdown = CybersecRewardBreakdown(
            detection=self._last_reward_breakdown.get("detection", 0.0),
            containment=self._last_reward_breakdown.get("containment", 0.0),
            business_continuity=self._last_reward_breakdown.get(
                "business_continuity", 0.0
            ),
            governance=self._last_reward_breakdown.get("governance", 0.0),
            efficiency=self._last_reward_breakdown.get("efficiency", 0.0),
            disruption_penalty=self._last_reward_breakdown.get(
                "disruption_penalty", 0.0
            ),
            governance_penalty=self._last_reward_breakdown.get(
                "governance_penalty", 0.0
            ),
            efficiency_penalty=self._last_reward_breakdown.get(
                "efficiency_penalty", 0.0
            ),
            adversary_pressure=self._last_reward_breakdown.get(
                "adversary_pressure", 0.0
            ),
            terminal_outcome=self._last_reward_breakdown.get("terminal_outcome", 0.0),
            false_positive_penalty=self._last_reward_breakdown.get(
                "false_positive_penalty", 0.0
            ),
            invalid_action_penalty=self._last_reward_breakdown.get(
                "invalid_action_penalty", 0.0
            ),
            total=self._last_reward_breakdown.get("total", 0.0),
        )

        info = {
            **info,
            "reward_breakdown": reward_breakdown.model_dump(),
        }

        return CybersecObservation(
            scenario_id=self._world.scenario.scenario_id,
            scenario_title=self._world.scenario.title,
            scenario_objective=self._world.scenario.objective,
            tick=self._state.step_count,
            horizon=self._horizon,
            enterprise_risk_score=self._world.enterprise_risk_score(),
            alerts=alerts,
            open_tickets=open_tickets,
            ticket_updates=[
                update.get("message", "") for update in info.get("workflow_updates", [])
            ],
            forensics_updates=forensics_updates,
            known_compromised_assets=sorted(self._world.known_compromised_assets),
            known_compromised_identities=sorted(
                self._world.known_compromised_identities
            ),
            recent_activity=self._world.get_recent_activity(limit=12),
            available_actions=list(self._ACTIONS) if not done else [],
            valid_targets=self._build_valid_targets(done=done),
            reward=reward,
            done=done,
            info=info,
        )

    def _build_valid_targets(self, *, done: bool) -> Dict[str, List[str]]:
        if done or self._world is None or self._workflow is None:
            return {}
        targets = self._world.target_catalog()
        targets["alert_ids"] = (
            [alert.alert_id for alert in self._telemetry.list_alerts()]
            if self._telemetry
            else []
        )
        targets["ticket_ids"] = [
            ticket.ticket_id for ticket in self._workflow.visible_tickets()
        ]
        targets["ready_ticket_ids"] = list(self._workflow.ready_ticket_ids())
        targets["query_log_sources"] = [
            "identity",
            "endpoint",
            "network",
            "code",
            "cloud",
            "ticketing",
        ]
        targets["ticketable_actions"] = [
            "ISOLATE_ASSET",
            "REVOKE_IDENTITY",
            "ROTATE_SECRET",
            "BLOCK_EGRESS",
            "PATCH_ASSET",
        ]
        return targets

    @staticmethod
    def _ticket_to_model(ticket: TicketRecord) -> WorkflowTicket:
        return WorkflowTicket(
            ticket_id=ticket.ticket_id,
            requested_action=ticket.requested_action,
            target=ticket.target,
            status=ticket.status,
            urgency=ticket.urgency,
            created_step=ticket.created_step,
            review_due_step=ticket.review_due_step,
            execute_ready_step=ticket.execute_ready_step,
        )
