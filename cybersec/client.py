# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cybersec Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    CybersecAction,
    CybersecObservation,
    ForensicsUpdate,
    SecurityAlert,
    WorkflowTicket,
)


class CybersecEnv(EnvClient[CybersecAction, CybersecObservation, State]):
    """Client for long-horizon enterprise cyber defense environment."""

    def _step_payload(self, action: CybersecAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
            "urgency": action.urgency,
        }
        if action.target is not None:
            payload["target"] = action.target
        if action.parameter is not None:
            payload["parameter"] = action.parameter
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CybersecObservation]:
        obs_data = payload.get("observation", {}) or {}

        alerts: List[SecurityAlert] = []
        for alert in obs_data.get("alerts", []) or []:
            alerts.append(
                SecurityAlert(
                    alert_id=alert.get("alert_id", ""),
                    source=alert.get("source", "endpoint"),
                    severity=alert.get("severity", "low"),
                    confidence=float(alert.get("confidence", 0.0)),
                    entity=alert.get("entity", ""),
                    title=alert.get("title", ""),
                    triage_status=alert.get("triage_status", "pending"),
                )
            )

        open_tickets: List[WorkflowTicket] = []
        for ticket in obs_data.get("open_tickets", []) or []:
            open_tickets.append(
                WorkflowTicket(
                    ticket_id=ticket.get("ticket_id", ""),
                    requested_action=ticket.get("requested_action", "ISOLATE_ASSET"),
                    target=ticket.get("target", ""),
                    status=ticket.get("status", "pending_review"),
                    urgency=ticket.get("urgency", "normal"),
                    created_step=int(ticket.get("created_step", 0)),
                    review_due_step=int(ticket.get("review_due_step", 0)),
                    execute_ready_step=ticket.get("execute_ready_step"),
                )
            )

        forensics_updates: List[ForensicsUpdate] = []
        for update in obs_data.get("forensics_updates", []) or []:
            forensics_updates.append(
                ForensicsUpdate(
                    job_id=update.get("job_id", ""),
                    target=update.get("target", ""),
                    target_type=update.get("target_type", "asset"),
                    status=update.get("status", "queued"),
                    finding=update.get("finding", ""),
                    confidence=float(update.get("confidence", 0.0)),
                )
            )

        observation = CybersecObservation(
            scenario_id=obs_data.get("scenario_id", ""),
            scenario_title=obs_data.get("scenario_title", ""),
            scenario_objective=obs_data.get("scenario_objective", ""),
            tick=int(obs_data.get("tick", 0)),
            horizon=int(obs_data.get("horizon", 1)),
            enterprise_risk_score=float(obs_data.get("enterprise_risk_score", 0.0)),
            alerts=alerts,
            open_tickets=open_tickets,
            ticket_updates=list(obs_data.get("ticket_updates", []) or []),
            forensics_updates=forensics_updates,
            known_compromised_assets=list(
                obs_data.get("known_compromised_assets", []) or []
            ),
            known_compromised_identities=list(
                obs_data.get("known_compromised_identities", []) or []
            ),
            recent_activity=list(obs_data.get("recent_activity", []) or []),
            available_actions=list(obs_data.get("available_actions", []) or []),
            valid_targets=dict(obs_data.get("valid_targets", {}) or {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            info=dict(obs_data.get("info", {}) or {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
