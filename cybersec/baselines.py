"""Reference defender policies + a minimal in-process episode runner.

Two policies are shipped:

  * :class:`RandomPolicy`    - uniform random over legal (action, target) pairs.
  * :class:`HeuristicPolicy` - rule-based defender driven by alert severity,
    confirmed-compromise lists, and a small set of thresholds. Encodes the
    behavior we expect a competent SOC analyst to exhibit; serves as the
    primary baseline a trained LLM defender must beat.

The episode runner :func:`run_episode` drives a single :class:`CybersecEnvironment`
instance with a chosen policy and returns a structured :class:`EpisodeResult`.
This is intentionally separate from the OpenEnv HTTP/WebSocket server so it
can be used inside training notebooks without a network hop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from .server.cybersec_environment import CybersecEnvironment
from .models import (
    ActionType,
    AlertEvent,
    AlertSignal,
    AttackerPersonality,
    CybersecAction,
    CybersecObservation,
)


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class DefenderPolicy(Protocol):
    """Stateless or stateful per-episode policy. ``reset`` is optional."""

    name: str

    def act(self, observation: CybersecObservation) -> CybersecAction: ...

    def reset(self) -> None: ...  # no-op default; concrete classes may override


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Uniform random policy constrained to currently-valid (action, target)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def reset(self) -> None:  # pragma: no cover - intentionally trivial
        pass

    def act(self, observation: CybersecObservation) -> CybersecAction:
        rng = self._rng
        atype = rng.choice(observation.available_actions)
        if atype is ActionType.MONITOR:
            return CybersecAction(action_type=atype)

        assets = observation.valid_targets.get("assets", [])
        identities = observation.valid_targets.get("identities", [])
        if atype is ActionType.INVESTIGATE:
            pool = assets + identities
        elif atype in (ActionType.ISOLATE_ASSET, ActionType.BLOCK_EGRESS, ActionType.PATCH_ASSET):
            pool = assets
        else:  # REVOKE_IDENTITY
            pool = identities

        if not pool:
            return CybersecAction(action_type=ActionType.MONITOR)
        return CybersecAction(action_type=atype, target=rng.choice(pool))


# ---------------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------------


@dataclass
class HeuristicConfig:
    investigate_severity_threshold: float = 0.30
    isolate_severity_threshold: float = 0.55
    revoke_severity_threshold: float = 0.55
    confirmed_isolate_min_confidence: float = 0.65
    monitor_when_quiet: bool = True


class HeuristicPolicy:
    """Decision tree over alerts + forensics.

    Priority order each tick:

      1. Containment of any *confirmed-compromised* asset/identity that is
         not yet isolated/revoked.
      2. Escalation to ISOLATE/REVOKE on the highest-severity alert if it
         crosses the isolation threshold.
      3. INVESTIGATE the most suspicious-looking unconfirmed target.
      4. PATCH a never-touched, on-attack-path asset (proactive hardening).
      5. MONITOR.
    """

    name = "heuristic"

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.cfg = config or HeuristicConfig()
        self._investigated: set[str] = set()

    def reset(self) -> None:
        self._investigated = set()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _alert_target(alert: AlertEvent) -> Optional[str]:
        return alert.asset or alert.identity

    @staticmethod
    def _strongest_alert(observation: CybersecObservation) -> Optional[AlertEvent]:
        best: Optional[AlertEvent] = None
        for a in observation.alerts:
            if a.signal is AlertSignal.BACKGROUND_NOISE:
                continue
            if best is None or a.severity > best.severity:
                best = a
        return best

    def _is_asset(self, target: str, obs: CybersecObservation) -> bool:
        return target in obs.valid_targets.get("assets", [])

    def _is_identity(self, target: str, obs: CybersecObservation) -> bool:
        return target in obs.valid_targets.get("identities", [])

    # ------------------------------------------------------------------ act

    def act(self, observation: CybersecObservation) -> CybersecAction:
        cfg = self.cfg

        # 1. Containment of confirmed-compromised targets we haven't sealed.
        for target in observation.confirmed_compromised:
            if self._is_asset(target, observation):
                if target not in observation.isolated_assets:
                    return CybersecAction(action_type=ActionType.ISOLATE_ASSET, target=target)
            elif self._is_identity(target, observation):
                if target not in observation.revoked_identities:
                    return CybersecAction(action_type=ActionType.REVOKE_IDENTITY, target=target)

        strongest = self._strongest_alert(observation)

        # 2. Escalate on a high-severity alert with a clear target.
        if strongest is not None:
            target = self._alert_target(strongest)
            if target is not None:
                if (
                    strongest.severity >= cfg.isolate_severity_threshold
                    and self._is_asset(target, observation)
                    and target not in observation.isolated_assets
                ):
                    if strongest.signal is AlertSignal.EGRESS_ANOMALY and target not in observation.blocked_egress_assets:
                        return CybersecAction(action_type=ActionType.BLOCK_EGRESS, target=target)
                    return CybersecAction(action_type=ActionType.ISOLATE_ASSET, target=target)
                if (
                    strongest.severity >= cfg.revoke_severity_threshold
                    and self._is_identity(target, observation)
                    and target not in observation.revoked_identities
                ):
                    return CybersecAction(action_type=ActionType.REVOKE_IDENTITY, target=target)

        # 3. Investigate a moderately-suspicious untouched target.
        if strongest is not None and strongest.severity >= cfg.investigate_severity_threshold:
            target = self._alert_target(strongest)
            if target is not None and target not in self._investigated:
                self._investigated.add(target)
                return CybersecAction(action_type=ActionType.INVESTIGATE, target=target)

        # 4. Proactively patch the first never-patched asset that already had
        #    *any* prior alert (cheap proxy for "the policy thought about it").
        seen_assets = {self._alert_target(a) for a in observation.alerts}
        seen_assets.discard(None)
        for asset in observation.valid_targets.get("assets", []):
            if asset in seen_assets and asset not in observation.patched_assets:
                return CybersecAction(action_type=ActionType.PATCH_ASSET, target=asset)

        # 5. Monitor.
        return CybersecAction(action_type=ActionType.MONITOR)


# ---------------------------------------------------------------------------
# Episode runner + result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    policy_name: str
    scenario_id: str
    attacker_personality: str
    seed: int
    steps: int
    cumulative_reward: float
    succeeded_stage_count: int
    total_stage_count: int
    exfil_completed: bool
    terminal_reason: Optional[str]
    confirmed_compromised: List[str]
    invalid_action_count: int
    false_positive_count: int
    # Iter-4: honest validity reporting.
    #
    # ``invalid_action_count`` only counts steps the *server* rejected as
    # invalid (target missing from valid_targets, etc). LLM policies that
    # silently fall back to ``MONITOR`` on parse failure trivially score
    # 0 on that metric while still being garbage. ``monitor_fallback_count``
    # is set by the policy itself via ``policy.last_act_was_fallback``
    # before each action is returned, and bubbles up through ``run_episode``.
    monitor_fallback_count: int = 0
    reward_curve: List[float] = field(default_factory=list)


def run_episode(
    env: CybersecEnvironment,
    policy: DefenderPolicy,
    seed: int,
    scenario_id: Optional[str] = None,
    attacker_personality: Optional[AttackerPersonality] = None,
) -> EpisodeResult:
    """Drive ``env`` with ``policy`` for a full episode, return diagnostics."""

    if hasattr(policy, "reset"):
        policy.reset()
    reset_kwargs: Dict[str, object] = {}
    if scenario_id is not None:
        reset_kwargs["scenario_id"] = scenario_id
    if attacker_personality is not None:
        reset_kwargs["attacker_personality"] = attacker_personality

    obs = env.reset(seed=seed, **reset_kwargs)
    invalid = 0
    fp = 0
    fallback = 0
    reward_curve: List[float] = []

    while not obs.done:
        action = policy.act(obs)
        # Policies that emit text and parse it (LLM policies in particular)
        # set this attribute *inside* `act()` so we can distinguish a
        # genuine MONITOR decision from a parse-failure fallback.
        if getattr(policy, "last_act_was_fallback", False):
            fallback += 1
        obs = env.step(action)
        rb = obs.info.get("reward_breakdown", {}) if isinstance(obs.info, dict) else {}
        if rb.get("invalid_action_penalty", 0.0):
            invalid += 1
        if rb.get("false_positive_penalty", 0.0):
            fp += 1
        reward_curve.append(float(obs.reward or 0.0))

    terminal = obs.info.get("terminal", {}) if isinstance(obs.info, dict) else {}
    return EpisodeResult(
        policy_name=policy.name,
        scenario_id=obs.scenario_id,
        attacker_personality=obs.attacker_personality.value,
        seed=seed,
        steps=obs.tick,
        cumulative_reward=float(terminal.get("cumulative_reward", sum(reward_curve))),
        succeeded_stage_count=int(terminal.get("stages_succeeded", 0)),
        total_stage_count=int(terminal.get("stages_total", 0)),
        exfil_completed=bool(terminal.get("exfil_completed", False)),
        terminal_reason=terminal.get("terminal_reason"),
        confirmed_compromised=list(obs.confirmed_compromised),
        invalid_action_count=invalid,
        false_positive_count=fp,
        monitor_fallback_count=fallback,
        reward_curve=reward_curve,
    )


def aggregate_results(results: List[EpisodeResult]) -> Dict[str, float]:
    """Mean over a batch of episode results, suitable for paper-style tables."""

    if not results:
        return {}
    n = len(results)
    total_steps = sum(r.steps for r in results) or 1
    return {
        "n_episodes": n,
        "mean_return": round(sum(r.cumulative_reward for r in results) / n, 3),
        "mean_steps": round(sum(r.steps for r in results) / n, 2),
        "exfil_rate": round(sum(1 for r in results if r.exfil_completed) / n, 3),
        "mean_stages_succeeded": round(
            sum(r.succeeded_stage_count for r in results) / n, 3
        ),
        "mean_invalid_actions": round(sum(r.invalid_action_count for r in results) / n, 3),
        "mean_false_positives": round(sum(r.false_positive_count for r in results) / n, 3),
        "monitor_fallback_rate": round(
            sum(r.monitor_fallback_count for r in results) / total_steps, 3
        ),
        "containment_rate": round(
            sum(
                1
                for r in results
                if r.total_stage_count
                and (r.total_stage_count - r.succeeded_stage_count) >= max(1, r.total_stage_count // 2)
            )
            / n,
            3,
        ),
    }


__all__ = [
    "DefenderPolicy",
    "RandomPolicy",
    "HeuristicConfig",
    "HeuristicPolicy",
    "EpisodeResult",
    "run_episode",
    "aggregate_results",
]
