"""CybersecEnvironment - the OpenEnv ``Environment`` subclass.

Per OpenEnv contract this class implements ``reset(seed, episode_id, **kw)``,
``step(action)``, and the ``state`` property. It owns:

  * the active :class:`Scenario` (assets, identities, stages, horizon)
  * an :class:`AttackerPolicy` carrying the per-episode personality
  * a :class:`TelemetryEngine` for noise alerts and INVESTIGATE oracles
  * a :class:`RewardModel` and the running cumulative reward
  * the defender's control state: isolated assets, revoked identities,
    blocked egress targets, patched assets, confirmed compromised targets

Reset accepts two keyword overrides:

  ``scenario_id``         - one of :func:`scenarios.list_scenarios`. If
                             omitted, pick ``list_scenarios()[seed % n]`` when
                             ``seed`` is set; if ``seed`` is also omitted, a
                             random seed is drawn so each reset can vary (HTTP
                             clients that omit both are not stuck on the first
                             scenario).
  ``attacker_personality``- a :class:`AttackerPersonality` value. Sampled by
                             RNG if omitted.

Step contract:

  * Every action consumes one tick.
  * Invalid actions still consume the tick; defender pays an
    ``invalid_action_penalty`` and no defender state is mutated.
  * The episode terminates when ``tick >= horizon``, the attacker exfiltrates,
    or the attack DAG has nothing left to advance.

This module lives at ``cybersec.server.cybersec_environment`` per the
official ``openenv init`` template layout.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server.interfaces import Environment

# Two-mode import. When the package is pip-installed (local dev, tests,
# notebooks) we are reachable as ``cybersec.server.cybersec_environment`` and
# the ``..`` imports resolve to ``cybersec.attacker`` etc. On Hugging Face
# Spaces the env is launched with ``uvicorn server.app:app`` from ``/app/env``,
# which makes ``server`` a *top-level* package and ``..`` reaches above it; we
# fall back to bare absolute imports that pick the modules up from CWD.
try:
    from ..attacker import (
        AttackerEvent,
        DefenderView,
        ScriptedAttacker,
        StageStatus,
    )
    from ..models import (
        ActionType,
        AlertEvent,
        AttackerPersonality,
        CybersecAction,
        CybersecObservation,
        CybersecState,
        ForensicResult,
    )
    from ..reward import RewardModel, RewardWeights, StepSignals
    from ..scenarios import Scenario, get_scenario, list_scenarios
    from ..telemetry import TelemetryEngine
except ImportError:  # pragma: no cover - HF Spaces / docker runtime path
    from attacker import (  # type: ignore[no-redef]
        AttackerEvent,
        DefenderView,
        ScriptedAttacker,
        StageStatus,
    )
    from models import (  # type: ignore[no-redef]
        ActionType,
        AlertEvent,
        AttackerPersonality,
        CybersecAction,
        CybersecObservation,
        CybersecState,
        ForensicResult,
    )
    from reward import RewardModel, RewardWeights, StepSignals  # type: ignore[no-redef]
    from scenarios import Scenario, get_scenario, list_scenarios  # type: ignore[no-redef]
    from telemetry import TelemetryEngine  # type: ignore[no-redef]


_ALERT_BUFFER = 12
_FORENSIC_BUFFER = 8


# ---------------------------------------------------------------------------
# Internal mutable world state (kept private; not serialized to clients)
# ---------------------------------------------------------------------------


@dataclass
class _WorldState:
    scenario: Scenario
    personality: AttackerPersonality
    tick: int = 0
    isolated_assets: Set[str] = field(default_factory=set)
    revoked_identities: Set[str] = field(default_factory=set)
    blocked_egress_assets: Set[str] = field(default_factory=set)
    patched_assets: Set[str] = field(default_factory=set)
    confirmed_compromised: Set[str] = field(default_factory=set)
    alerts: List[AlertEvent] = field(default_factory=list)
    forensics: List[ForensicResult] = field(default_factory=list)
    cumulative_reward: float = 0.0
    defender_acted_at_least_once: bool = False
    done: bool = False
    exfil_completed: bool = False
    last_terminal_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CybersecEnvironment(Environment[CybersecAction, CybersecObservation, CybersecState]):
    """Long-horizon, partially observable enterprise defender environment.

    The OpenEnv server wraps this class in HTTP/WebSocket endpoints; the
    in-package :class:`cybersec.client.CybersecEnv` client is the canonical
    way to drive it from Python.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        default_scenario_id: Optional[str] = None,
        default_personality: Optional[AttackerPersonality] = None,
        reward_weights: Optional[RewardWeights] = None,
    ) -> None:
        super().__init__()
        self._default_scenario_id = default_scenario_id
        self._default_personality = default_personality
        self._reward_model = RewardModel(reward_weights)
        self._world: Optional[_WorldState] = None
        self._attacker: Optional[ScriptedAttacker] = None
        self._telemetry: Optional[TelemetryEngine] = None
        self._rng: Optional[random.Random] = None
        self._attack_path_assets: Set[str] = set()
        self._attack_path_identities: Set[str] = set()
        self._compromisable_assets: Set[str] = set()
        self._compromisable_identities: Set[str] = set()

    # -------------------------------------------------------------- reset

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CybersecObservation:
        # OpenEnv HTTP reset often omits seed; rng_seed=0 would force scenario
        # list_scenarios()[0] every time (supply_chain_token_drift).
        if seed is None:
            rng_seed = secrets.randbits(31)
        else:
            rng_seed = int(seed)
        self._rng = random.Random(rng_seed)

        scenario_id = kwargs.get("scenario_id") or self._default_scenario_id
        if scenario_id is None:
            ids = list_scenarios()
            scenario_id = ids[rng_seed % len(ids)]
        scenario = get_scenario(scenario_id)

        personality_arg = kwargs.get("attacker_personality") or self._default_personality
        if personality_arg is None:
            personality = self._rng.choice(list(AttackerPersonality))
        elif isinstance(personality_arg, AttackerPersonality):
            personality = personality_arg
        else:
            personality = AttackerPersonality(personality_arg)

        self._attacker = ScriptedAttacker(scenario, personality, self._rng)
        self._telemetry = TelemetryEngine(scenario, self._rng)
        self._world = _WorldState(scenario=scenario, personality=personality)

        self._attack_path_assets = {
            stage.target_asset for stage in scenario.stages if stage.target_asset
        }
        self._attack_path_identities = {
            stage.target_identity for stage in scenario.stages if stage.target_identity
        }
        self._compromisable_assets = {
            stage.target_asset
            for stage in scenario.stages
            if stage.compromises_asset and stage.target_asset
        }
        self._compromisable_identities = {
            stage.target_identity
            for stage in scenario.stages
            if stage.compromises_identity and stage.target_identity
        }

        return self._build_observation(reward_breakdown=None, terminal_info=None)

    # -------------------------------------------------------------- step

    def step(
        self,
        action: CybersecAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CybersecObservation:
        if self._world is None or self._attacker is None or self._telemetry is None:
            raise RuntimeError("step() called before reset()")
        world = self._world
        if world.done:
            raise RuntimeError("step() called on a terminated episode; call reset()")

        world.tick += 1
        signals = StepSignals(total_stage_count=len(world.scenario.stages))

        invalid, validation_msg = self._validate_action(action)
        if invalid:
            signals.invalid_action_count = 1
            applied_forensic: Optional[ForensicResult] = None
        else:
            applied_forensic, fp_count, was_containment = self._apply_defender_action(action, signals)
            signals.false_positive_count += fp_count
            if was_containment:
                signals.containment_action_count = 1
                # Evidence bonus: containing a target already confirmed compromised
                if action.target and action.target in world.confirmed_compromised:
                    signals.containment_on_confirmed += 1
            if action.action_type is not ActionType.MONITOR:
                world.defender_acted_at_least_once = True

        if applied_forensic is not None:
            world.forensics.append(applied_forensic)
            world.forensics = world.forensics[-_FORENSIC_BUFFER:]
            if (
                applied_forensic.is_compromised
                and applied_forensic.confidence >= 0.6
                and applied_forensic.target not in world.confirmed_compromised
                and self._target_actually_compromised(
                    applied_forensic.target, applied_forensic.target_kind
                )
            ):
                world.confirmed_compromised.add(applied_forensic.target)
                signals.new_confirmed_compromised.add(applied_forensic.target)

        defender_view = DefenderView(
            isolated_assets=set(world.isolated_assets),
            revoked_identities=set(world.revoked_identities),
            blocked_egress_assets=set(world.blocked_egress_assets),
            patched_assets=set(world.patched_assets),
            defender_acted_this_tick=action.action_type is not ActionType.MONITOR
            and not invalid,
        )
        ev: AttackerEvent = self._attacker.step(world.tick, defender_view)

        for alert in ev.surfaced_alerts:
            world.alerts.append(alert)
        world.alerts.extend(self._telemetry.background_alerts(world.tick))
        world.alerts = world.alerts[-_ALERT_BUFFER:]

        signals.contained_active_stage_ids.update(ev.blocked_active)
        signals.contained_preemptive_stage_ids.update(ev.blocked_preemptive)
        if ev.exfil_completed:
            signals.exfil_completed = True

        active_count, weighted = self._disruption_load()
        signals.active_control_count = active_count
        signals.weighted_disruption = weighted

        terminate_reason: Optional[str] = None
        if ev.exfil_completed:
            terminate_reason = "exfil_completed"
        elif world.tick >= world.scenario.horizon:
            terminate_reason = "horizon_reached"
        elif self._attacker.is_done():
            terminate_reason = "attacker_done"
        signals.is_terminal = terminate_reason is not None
        signals.succeeded_stage_count = len(self._attacker.succeeded_stage_ids())
        signals.defender_acted_at_least_once = world.defender_acted_at_least_once

        reward = self._reward_model.step_reward(signals)
        world.cumulative_reward += reward.total

        terminal_info: Optional[Dict[str, Any]] = None
        if terminate_reason is not None:
            world.done = True
            world.exfil_completed = ev.exfil_completed
            world.last_terminal_reason = terminate_reason
            terminal_info = self._build_terminal_info(terminate_reason)
            terminal_info["validation_msg"] = validation_msg

        return self._build_observation(reward_breakdown=reward, terminal_info=terminal_info,
                                       validation_msg=validation_msg)

    # -------------------------------------------------------------- state

    @property
    def state(self) -> CybersecState:
        if self._world is None:
            return CybersecState()
        w = self._world
        return CybersecState(
            scenario_id=w.scenario.scenario_id,
            attacker_personality=w.personality,
            tick=w.tick,
            horizon=w.scenario.horizon,
            completed_attack_stages=self._attacker.succeeded_stage_ids() if self._attacker else [],
            cumulative_reward=round(w.cumulative_reward, 4),
            done=w.done,
        )

    # ====================================================================== helpers

    def _validate_action(self, action: CybersecAction) -> tuple[bool, str]:
        """Return ``(is_invalid, message)``. Pydantic guarantees structural
        fields; this layer enforces the *live* valid_targets gate.
        """

        if self._world is None:
            return True, "world not initialized"
        scenario = self._world.scenario
        asset_ids = {a.asset_id for a in scenario.assets}
        identity_ids = {i.identity_id for i in scenario.identities}

        atype = action.action_type
        target = action.target
        if atype is ActionType.MONITOR:
            return False, "ok"
        if atype is ActionType.INVESTIGATE:
            if target in asset_ids or target in identity_ids:
                return False, "ok"
            return True, f"investigate target {target!r} is not a known asset or identity"
        if atype in (ActionType.ISOLATE_ASSET, ActionType.BLOCK_EGRESS, ActionType.PATCH_ASSET):
            if target in asset_ids:
                return False, "ok"
            return True, f"{atype.value} requires a known asset target, got {target!r}"
        if atype is ActionType.REVOKE_IDENTITY:
            if target in identity_ids:
                return False, "ok"
            return True, f"REVOKE_IDENTITY requires a known identity target, got {target!r}"
        return True, f"unsupported action_type {atype!r}"

    def _apply_defender_action(
        self, action: CybersecAction, signals: StepSignals
    ) -> tuple[Optional[ForensicResult], int, bool]:
        """Mutate world state for a validated action.

        Returns ``(forensic_result_if_any, false_positive_count_added,
        is_containment_action)``. The boolean is True iff the action is one of
        the four containment verbs (ISOLATE / REVOKE / BLOCK / PATCH); the
        reward model uses it to apply a small per-action containment cost so
        that spamming containment isn't free.
        """

        assert self._world is not None and self._attacker is not None
        assert self._telemetry is not None
        world = self._world
        atype = action.action_type
        target = action.target

        if atype is ActionType.MONITOR:
            return None, 0, False

        if atype is ActionType.INVESTIGATE:
            kind = "asset" if target in {a.asset_id for a in world.scenario.assets} else "identity"
            forensic = self._telemetry.investigate(
                tick=world.tick,
                target=target,
                target_kind=kind,
                truly_compromised_assets=self._attacker.compromised_assets,
                truly_compromised_identities=self._attacker.compromised_identities,
            )
            return forensic, 0, False

        is_attack_path = target in self._attack_path_assets or target in self._attack_path_identities
        false_positive = 0
        if not is_attack_path:
            false_positive = 1

        if atype is ActionType.ISOLATE_ASSET:
            world.isolated_assets.add(target)
        elif atype is ActionType.REVOKE_IDENTITY:
            world.revoked_identities.add(target)
        elif atype is ActionType.BLOCK_EGRESS:
            world.blocked_egress_assets.add(target)
        elif atype is ActionType.PATCH_ASSET:
            world.patched_assets.add(target)

        return None, false_positive, True

    def _target_actually_compromised(self, target: str, kind: str) -> bool:
        if self._attacker is None:
            return False
        if kind == "asset":
            return target in self._attacker.compromised_assets
        if kind == "identity":
            return target in self._attacker.compromised_identities
        return False

    def _disruption_load(self) -> tuple[int, float]:
        """Return ``(active_count, criticality-weighted disruption score)``."""

        assert self._world is not None
        w = self._world
        scenario = w.scenario
        asset_lookup = {a.asset_id: a for a in scenario.assets}
        identity_lookup = {i.identity_id: i for i in scenario.identities}
        weighted = 0.0
        count = 0
        for asset_id in w.isolated_assets:
            count += 1
            asset = asset_lookup.get(asset_id)
            weighted += asset.criticality if asset else 0.5
        for asset_id in w.blocked_egress_assets:
            count += 1
            asset = asset_lookup.get(asset_id)
            weighted += 0.4 * (asset.criticality if asset else 0.5)
        for identity_id in w.revoked_identities:
            count += 1
            identity = identity_lookup.get(identity_id)
            weighted += 0.3 * (identity.privilege if identity else 0.5)
        return count, weighted

    def _build_terminal_info(self, reason: str) -> Dict[str, Any]:
        assert self._world is not None and self._attacker is not None
        w = self._world
        return {
            "terminal_reason": reason,
            "succeeded_stage_ids": self._attacker.succeeded_stage_ids(),
            "stages_succeeded": len(self._attacker.succeeded_stage_ids()),
            "stages_total": len(w.scenario.stages),
            "exfil_completed": w.exfil_completed,
            "cumulative_reward": round(w.cumulative_reward, 4),
            "scenario_id": w.scenario.scenario_id,
            "attacker_personality": w.personality.value,
            "tick": w.tick,
            "horizon": w.scenario.horizon,
        }

    # -------------------------------------------------------------- observation

    def _build_observation(
        self,
        reward_breakdown,
        terminal_info: Optional[Dict[str, Any]] = None,
        validation_msg: str = "ok",
    ) -> CybersecObservation:
        assert self._world is not None
        w = self._world

        valid_targets = {
            "assets": [a.asset_id for a in w.scenario.assets],
            "identities": [i.identity_id for i in w.scenario.identities],
        }
        info: Dict[str, Any] = {"validation_msg": validation_msg}
        if reward_breakdown is not None:
            info["reward_breakdown"] = reward_breakdown.model_dump()
            info["cumulative_reward"] = round(w.cumulative_reward, 4)
        if terminal_info is not None:
            info["terminal"] = terminal_info

        obs = CybersecObservation(
            tick=w.tick,
            horizon=w.scenario.horizon,
            scenario_id=w.scenario.scenario_id,
            attacker_personality=w.personality,
            alerts=list(w.alerts),
            forensics=list(w.forensics),
            isolated_assets=sorted(w.isolated_assets),
            revoked_identities=sorted(w.revoked_identities),
            blocked_egress_assets=sorted(w.blocked_egress_assets),
            patched_assets=sorted(w.patched_assets),
            confirmed_compromised=sorted(w.confirmed_compromised),
            valid_targets=valid_targets,
            available_actions=list(ActionType),
            info=info,
            done=w.done,
            reward=reward_breakdown.total if reward_breakdown is not None else None,
        )
        return obs


__all__ = ["CybersecEnvironment"]
