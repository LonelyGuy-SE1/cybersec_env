"""Pydantic data models for the Cybersec environment.

This module defines the public contract between the environment, agents,
and any downstream training/evaluation code. All field names and types are
stable; downstream tooling (the LLM prompt formatter, the reward shaping in
training scripts, the heuristic baseline) reads these classes directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """The six defender actions exposed to the policy.

    The action surface is intentionally small so a 1.5B-parameter LLM can
    reliably emit valid JSON and reason about target selection.
    """

    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    ISOLATE_ASSET = "ISOLATE_ASSET"
    REVOKE_IDENTITY = "REVOKE_IDENTITY"
    BLOCK_EGRESS = "BLOCK_EGRESS"
    PATCH_ASSET = "PATCH_ASSET"


class AttackerPersonality(str, Enum):
    """Three scripted attacker archetypes sampled per episode."""

    STEALTHY = "stealthy"
    AGGRESSIVE = "aggressive"
    OPPORTUNISTIC = "opportunistic"


class AlertSignal(str, Enum):
    """Coarse signal type attached to every alert.

    Real SOCs surface dozens of signal kinds; we keep five buckets so the
    policy can learn distinct response patterns without combinatorial blow-up.
    """

    AUTH_ANOMALY = "auth_anomaly"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_STAGING = "data_staging"
    EGRESS_ANOMALY = "egress_anomaly"
    BACKGROUND_NOISE = "background_noise"


# ---------------------------------------------------------------------------
# Public data records (used inside Observation)
# ---------------------------------------------------------------------------


class AlertEvent(BaseModel):
    """One alert visible to the defender at a given tick.

    Alerts are produced by the telemetry engine. Real adversary actions
    surface as alerts only after a stochastic detection delay; benign
    background noise also produces alerts. Severity is the only signal the
    defender has to discriminate the two.
    """

    model_config = ConfigDict(extra="forbid")

    tick: int = Field(..., description="Tick at which the alert became visible")
    signal: AlertSignal = Field(..., description="Coarse signal category")
    asset: Optional[str] = Field(default=None, description="Asset implicated, if any")
    identity: Optional[str] = Field(default=None, description="Identity implicated, if any")
    severity: float = Field(..., ge=0.0, le=1.0, description="0..1 confidence proxy")
    description: str = Field(default="", description="Short human-readable hint")


class ForensicResult(BaseModel):
    """Outcome of an INVESTIGATE action.

    Forensics return a noisy ground-truth signal: confidence rises with the
    asset/identity actually being on the active attack path, but it is never
    exactly 1.0 to keep the defender from blindly trusting one query.
    """

    model_config = ConfigDict(extra="forbid")

    tick: int = Field(..., description="Tick the investigation completed")
    target: str = Field(..., description="Asset or identity that was investigated")
    target_kind: str = Field(..., description="'asset' or 'identity'")
    is_compromised: bool = Field(..., description="Investigator's verdict")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the verdict")


class RewardBreakdown(BaseModel):
    """Per-step reward decomposition exposed for analysis and reward shaping."""

    model_config = ConfigDict(extra="forbid")

    detection: float = 0.0
    containment: float = 0.0
    evidence_bonus: float = 0.0
    false_positive_penalty: float = 0.0
    disruption_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    terminal_score: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class CybersecAction(Action):
    """Defender action.

    All six action types take at most a single string ``target``; this keeps
    the JSON the policy must emit minimal:

        {"action_type": "ISOLATE_ASSET", "target": "asset-web-01"}

    Validation rules (enforced server-side):
      * MONITOR forbids a target.
      * INVESTIGATE requires a target (asset or identity).
      * ISOLATE_ASSET / BLOCK_EGRESS / PATCH_ASSET require an asset target.
      * REVOKE_IDENTITY requires an identity target.

    Cross-checking the target against the live ``valid_targets`` dictionary
    in the observation is the environment's responsibility, not Pydantic's.
    """

    action_type: ActionType = Field(..., description="One of the six action verbs")
    target: Optional[str] = Field(default=None, description="Asset or identity id")

    @model_validator(mode="after")
    def _check_target_presence(self) -> "CybersecAction":
        requires_target = {
            ActionType.INVESTIGATE,
            ActionType.ISOLATE_ASSET,
            ActionType.REVOKE_IDENTITY,
            ActionType.BLOCK_EGRESS,
            ActionType.PATCH_ASSET,
        }
        if self.action_type in requires_target and not self.target:
            raise ValueError(f"{self.action_type.value} requires a non-empty target")
        if self.action_type is ActionType.MONITOR and self.target:
            raise ValueError("MONITOR must not be given a target")
        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class CybersecObservation(Observation):
    """Defender's view of the world at a single tick.

    Fields are partitioned into:

      * Episode meta: tick / horizon / scenario_id / attacker_personality.
      * Telemetry: ``alerts`` (recent, lag-delayed) and ``forensics`` (responses
        to past INVESTIGATE actions).
      * Defender controls: which assets/identities are isolated, revoked,
        blocked, patched, or have been confirmed compromised.
      * Action grounding: ``available_actions`` and ``valid_targets`` so the
        policy can be hard-constrained to the legal action space.
      * ``info``: free-form dict carrying ``reward_breakdown`` every step and
        ``terminal`` info on the final tick.
    """

    tick: int = Field(..., description="Current tick (0-indexed)")
    horizon: int = Field(..., description="Maximum number of ticks in the episode")

    scenario_id: str = Field(..., description="Active scenario identifier")
    attacker_personality: AttackerPersonality = Field(
        ..., description="Sampled attacker archetype for this episode"
    )

    alerts: List[AlertEvent] = Field(
        default_factory=list,
        description="Alerts visible to the defender (most recent last)",
    )
    forensics: List[ForensicResult] = Field(
        default_factory=list,
        description="Investigation results received so far",
    )

    isolated_assets: List[str] = Field(default_factory=list)
    revoked_identities: List[str] = Field(default_factory=list)
    blocked_egress_assets: List[str] = Field(default_factory=list)
    patched_assets: List[str] = Field(default_factory=list)
    confirmed_compromised: List[str] = Field(
        default_factory=list,
        description="Targets the defender has positively identified as compromised",
    )

    valid_targets: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="{'assets': [...], 'identities': [...]} - canonical legal targets",
    )
    available_actions: List[ActionType] = Field(
        default_factory=list,
        description="Action verbs currently legal (filtered by remaining cooldowns/etc.)",
    )

    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-step diagnostics: reward_breakdown, terminal stats, debug",
    )


# ---------------------------------------------------------------------------
# State (server-side only; never returned to clients verbatim)
# ---------------------------------------------------------------------------


class CybersecState(State):
    """Coarse state snapshot for debugging / GET /state.

    Intentionally lightweight: the full ground-truth WorldState lives in
    ``cybersec.server.cybersec_environment`` and is kept private. Only fields
    safe to expose on the OpenEnv state endpoint live here.
    """

    scenario_id: Optional[str] = None
    attacker_personality: Optional[AttackerPersonality] = None
    tick: int = 0
    horizon: int = 0
    completed_attack_stages: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


__all__ = [
    "ActionType",
    "AttackerPersonality",
    "AlertSignal",
    "AlertEvent",
    "ForensicResult",
    "RewardBreakdown",
    "CybersecAction",
    "CybersecObservation",
    "CybersecState",
]
