# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for CybersecEnv enterprise campaign simulation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator

ActionType = Literal[
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

TicketActionType = Literal[
    "ISOLATE_ASSET",
    "REVOKE_IDENTITY",
    "ROTATE_SECRET",
    "BLOCK_EGRESS",
    "PATCH_ASSET",
]

LogSourceType = Literal[
    "identity",
    "endpoint",
    "network",
    "code",
    "cloud",
    "ticketing",
]

UrgencyType = Literal["low", "normal", "high"]


class SecurityAlert(BaseModel):
    """Agent-facing alert representation generated from noisy telemetry."""

    alert_id: str
    source: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    entity: str
    title: str
    triage_status: Literal["pending", "triaged"]


class WorkflowTicket(BaseModel):
    """Visible incident-response workflow ticket."""

    ticket_id: str
    requested_action: TicketActionType
    target: str
    status: Literal[
        "pending_review",
        "approved_waiting",
        "ready",
        "rejected",
        "executed",
    ]
    urgency: UrgencyType
    created_step: int = Field(..., ge=0)
    review_due_step: int = Field(..., ge=0)
    execute_ready_step: Optional[int] = Field(default=None, ge=0)


class ForensicsUpdate(BaseModel):
    """Visible update from asynchronous forensics jobs."""

    job_id: str
    target: str
    target_type: Literal["asset", "identity"]
    status: Literal["queued", "completed"]
    finding: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class CybersecRewardBreakdown(BaseModel):
    """Reward decomposition for diagnostics and judging explainability."""

    detection: float
    containment: float
    business_continuity: float
    governance: float
    efficiency: float
    disruption_penalty: float
    governance_penalty: float
    efficiency_penalty: float
    adversary_pressure: float
    terminal_outcome: float
    false_positive_penalty: float
    invalid_action_penalty: float
    total: float


class CybersecAction(Action):
    """Structured action for long-horizon enterprise incident handling."""

    action_type: ActionType = Field(..., description="Defender operation to execute")
    target: Optional[str] = Field(
        default=None,
        description="Entity target (asset, identity, alert_id, ticket_id)",
    )
    parameter: Optional[str] = Field(
        default=None,
        description=(
            "Auxiliary parameter. Used as log source for QUERY_LOGS, or"
            " requested control action for OPEN_TICKET."
        ),
    )
    urgency: UrgencyType = Field(
        default="normal",
        description="Workflow urgency for OPEN_TICKET requests",
    )

    @model_validator(mode="after")
    def validate_action_contract(self) -> "CybersecAction":
        no_target_actions = {"MONITOR"}
        target_only_actions = {
            "TRIAGE_ALERT",
            "REQUEST_FORENSICS",
            "EXECUTE_TICKET",
            "ISOLATE_ASSET",
            "REVOKE_IDENTITY",
            "ROTATE_SECRET",
            "BLOCK_EGRESS",
            "PATCH_ASSET",
        }

        if self.action_type in no_target_actions:
            if self.target is not None or self.parameter is not None:
                raise ValueError(
                    f"{self.action_type} does not accept target or parameter"
                )
            return self

        if self.action_type == "QUERY_LOGS":
            if self.target is not None:
                raise ValueError("QUERY_LOGS does not accept target")
            if self.parameter is None:
                raise ValueError("QUERY_LOGS requires parameter=log_source")
            if self.parameter not in {
                "identity",
                "endpoint",
                "network",
                "code",
                "cloud",
                "ticketing",
            }:
                raise ValueError("QUERY_LOGS parameter must be a supported log source")
            return self

        if self.action_type == "OPEN_TICKET":
            if self.target is None:
                raise ValueError("OPEN_TICKET requires target")
            if self.parameter not in {
                "ISOLATE_ASSET",
                "REVOKE_IDENTITY",
                "ROTATE_SECRET",
                "BLOCK_EGRESS",
                "PATCH_ASSET",
            }:
                raise ValueError("OPEN_TICKET requires parameter=ticketable action")
            return self

        if self.action_type in target_only_actions:
            if self.target is None:
                raise ValueError(f"{self.action_type} requires target")
            if self.parameter is not None:
                raise ValueError(f"{self.action_type} does not use parameter")
            return self

        return self


class CybersecObservation(Observation):
    """Observation for partially observable enterprise defense campaigns."""

    scenario_id: str
    scenario_title: str
    scenario_objective: str
    tick: int = Field(..., ge=0)
    horizon: int = Field(..., ge=1)
    enterprise_risk_score: float = Field(..., ge=0.0, le=1.0)
    alerts: List[SecurityAlert] = Field(default_factory=list)
    open_tickets: List[WorkflowTicket] = Field(default_factory=list)
    ticket_updates: List[str] = Field(default_factory=list)
    forensics_updates: List[ForensicsUpdate] = Field(default_factory=list)
    known_compromised_assets: List[str] = Field(default_factory=list)
    known_compromised_identities: List[str] = Field(default_factory=list)
    recent_activity: List[str] = Field(default_factory=list)
    available_actions: List[ActionType] = Field(default_factory=list)
    valid_targets: Dict[str, List[str]] = Field(default_factory=dict)
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-readable info channel for diagnostics and evaluation",
    )
