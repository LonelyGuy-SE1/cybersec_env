"""TRL-compatible reward functions for GRPO training of a defender LLM.

This module is the single source of truth for every reward signal used to
train the Cybersec OpenEnv defender. The runtime environment (in
``cybersec.server``) does NOT import from here -- training rewards are a
separate layer that operates on LLM completions, not on env transitions.

Why six task-rewards plus two anti-collapse rewards?

    The env itself returns a single scalar ``obs.reward`` per step (the
    six-channel reward model in ``cybersec.server.reward_model``). That
    scalar is plumbed back to the trainer via :func:`reward_step_total`,
    which clones a snapshotted env and applies the candidate action. That
    one function IS the env reward.

    The other reward functions exist because GRPO scores text completions,
    not env transitions. Things like "is the JSON parseable?" or "is the
    target in ``valid_targets``?" are properties of the LLM's output, not
    of the env's MDP. They give GRPO a dense, immediately-available signal
    that the env reward cannot.

    The two diversity rewards (:func:`reward_action_diversity` and
    :func:`reward_observation_aware`) were added in iter-2 specifically to
    counter the mode-collapse failure observed in iter-1: the trained
    policy memorised a single deterministic action sequence that froze the
    attacker on 2/3 scenarios with std=0 across 50 different seeds. The
    diversity rewards make that behaviour costly during training without
    changing the env.

All reward functions match TRL's ``GRPOTrainer`` signature::

    fn(prompts, completions, **kw_from_dataset_columns) -> List[float]

Per-row dataset columns (``valid_assets``, ``isolated_assets``,
``env_snapshot``, ...) are forwarded to the function as keyword arguments
of the same name. Missing kwargs default to lists of ``None`` / ``[]`` of
the right length so reward funcs are robust to dataset shape changes.
"""

from __future__ import annotations

import base64
import json
import pickle
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..models import ActionType, CybersecAction

try:  # local import, but make tests importable even without server deps
    from ..server.cybersec_environment import CybersecEnvironment  # noqa: F401
except Exception:  # pragma: no cover - server-side imports are optional here
    CybersecEnvironment = None  # type: ignore[assignment]


SYSTEM_PROMPT = (
    "You are an SRE-grade cyber-defender driving an OpenEnv environment.\n"
    "Reply with exactly one JSON object on one line of the form\n"
    '{"action_type": "...", "target": "..."}.\n'
    "action_type must be one of MONITOR, INVESTIGATE, ISOLATE_ASSET, "
    "REVOKE_IDENTITY, BLOCK_EGRESS, PATCH_ASSET.\n"
    "target must be omitted (or null) for MONITOR; otherwise it must come "
    "from valid_targets."
)


# ---------------------------------------------------------------------------
# Parsing helpers (used by every reward function and by inference policies)
# ---------------------------------------------------------------------------


def parse_first_json_object(text: str) -> Optional[dict]:
    """Return the first balanced ``{...}`` JSON object in ``text`` or ``None``.

    Lenient on purpose: small LLMs love to wrap their JSON in chatter
    ("Sure! Here's the action: { ... }. Hope that helps."). We chop out the
    first balanced object and try to parse it.
    """

    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parsed_action(completion: str) -> Optional[CybersecAction]:
    """Best-effort: completion -> :class:`CybersecAction` or ``None``."""

    payload = parse_first_json_object(completion)
    if not payload:
        return None
    try:
        return CybersecAction(**payload)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Observation rendering + env snapshotting
# ---------------------------------------------------------------------------


def render_observation(obs) -> str:
    """Render an observation as the user-message text the LLM sees.

    Kept stable across training and inference so the same prompt is used
    in dataset build, training, baseline eval, and trained-policy eval.
    Includes a short telemetry timeline (last 6 alerts, last 4 forensics)
    so a long-horizon policy has access to *change*, not just current state.
    """

    lines = [
        f"tick={obs.tick}/{obs.horizon}  scenario={obs.scenario_id}  "
        f"attacker={obs.attacker_personality.value}",
        f"isolated_assets={list(obs.isolated_assets)}",
        f"revoked_identities={list(obs.revoked_identities)}",
        f"blocked_egress={list(obs.blocked_egress_assets)}",
        f"patched={list(obs.patched_assets)}",
        f"confirmed_compromised={list(obs.confirmed_compromised)}",
        f"valid_targets={dict(obs.valid_targets)}",
        "recent_alerts:",
    ]
    for a in obs.alerts[-6:]:
        lines.append(
            f"  t={a.tick} {a.signal.value} sev={a.severity} asset={a.asset} "
            f"id={a.identity} :: {a.description}"
        )
    lines.append("recent_forensics:")
    for f in obs.forensics[-4:]:
        lines.append(
            f"  t={f.tick} {f.target_kind}={f.target} "
            f"compromised={f.is_compromised} conf={f.confidence}"
        )
    return "\n".join(lines)


def snapshot_env(env) -> str:
    """Pickle ``env`` to a base64 blob.

    Used at dataset-build time to attach a "rollback point" to every
    training row, so :func:`reward_step_total` can clone the env and apply
    a candidate action without touching the live environment used for
    bookkeeping.
    """

    return base64.b64encode(pickle.dumps(env)).decode("ascii")


def restore_env(blob: str):
    """Inverse of :func:`snapshot_env`."""

    return pickle.loads(base64.b64decode(blob.encode("ascii")))


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _coerce_list(value: Optional[Sequence], n: int, default: Any) -> List[Any]:
    if value is None:
        return [default] * n
    return list(value)


def reward_json_valid(prompts, completions, **kw) -> List[float]:
    """1.0 if the completion contains a parseable JSON object, else 0.0.

    Necessary because Qwen 1.5B in 4-bit happily produces text-prefixed JSON
    or unbalanced braces on a cold start. Without this signal the model
    spends most of GRPO learning to close brackets.
    """

    return [1.0 if parse_first_json_object(c) is not None else 0.0 for c in completions]


def reward_schema_valid(prompts, completions, **kw) -> List[float]:
    """1.0 if the JSON also validates as a :class:`CybersecAction`."""

    return [1.0 if parsed_action(c) is not None else 0.0 for c in completions]


def reward_target_in_valid_targets(
    prompts,
    completions,
    valid_assets: Optional[Sequence] = None,
    valid_identities: Optional[Sequence] = None,
    **kw,
) -> List[float]:
    """1.0 if the action's target is legal in this state, else 0.0.

    The env itself penalises invalid actions via
    ``invalid_action_penalty``, but only mildly. This explicit signal makes
    the gradient much denser at the start of training.
    """

    n = len(completions)
    valid_assets = _coerce_list(valid_assets, n, [])
    valid_identities = _coerce_list(valid_identities, n, [])
    out: List[float] = []
    for c, va, vi in zip(completions, valid_assets, valid_identities):
        action = parsed_action(c)
        if action is None:
            out.append(0.0)
            continue
        if action.action_type is ActionType.MONITOR:
            out.append(1.0)
            continue
        if action.action_type is ActionType.REVOKE_IDENTITY:
            out.append(1.0 if action.target in (vi or []) else 0.0)
        elif action.action_type is ActionType.INVESTIGATE:
            out.append(1.0 if action.target in ((va or []) + (vi or [])) else 0.0)
        else:
            out.append(1.0 if action.target in (va or []) else 0.0)
    return out


def reward_no_redundant_containment(
    prompts,
    completions,
    isolated_assets: Optional[Sequence] = None,
    revoked_identities: Optional[Sequence] = None,
    blocked_egress: Optional[Sequence] = None,
    patched: Optional[Sequence] = None,
    **kw,
) -> List[float]:
    """Penalise re-applying a containment action to an already-contained target.

    ``MONITOR`` gets 0.5 (it is always at least vaguely safe), other
    actions get 1.0 if their target is fresh, 0.0 if it is already on the
    contained list. The env gives no negative reward for redundant
    containments, so without this signal the policy can ride a
    "re-isolate-everything" loop forever.
    """

    n = len(completions)
    isolated_assets = _coerce_list(isolated_assets, n, [])
    revoked_identities = _coerce_list(revoked_identities, n, [])
    blocked_egress = _coerce_list(blocked_egress, n, [])
    patched = _coerce_list(patched, n, [])
    out: List[float] = []
    for c, iso, rev, blk, pat in zip(
        completions, isolated_assets, revoked_identities, blocked_egress, patched
    ):
        action = parsed_action(c)
        if action is None:
            out.append(0.0)
            continue
        if action.action_type is ActionType.MONITOR:
            out.append(0.5)
            continue
        already = {
            ActionType.ISOLATE_ASSET: iso,
            ActionType.REVOKE_IDENTITY: rev,
            ActionType.BLOCK_EGRESS: blk,
            ActionType.PATCH_ASSET: pat,
        }.get(action.action_type, [])
        out.append(0.0 if action.target in (already or []) else 1.0)
    return out


def reward_step_total(
    prompts,
    completions,
    env_snapshot: Optional[Sequence[str]] = None,
    **kw,
) -> List[float]:
    """The actual env reward, obtained by cloning a snapshotted env and stepping.

    This is the bridge from the training script back to the env's MDP.
    Every other reward function in this module is a training-time shaping
    signal; this one IS the env reward. Clamped to ``[-2, 2]`` so a
    single bad step cannot dominate the sum of the seven other channels.
    """

    n = len(completions)
    snapshots = _coerce_list(env_snapshot, n, None)
    out: List[float] = []
    for c, blob in zip(completions, snapshots):
        action = parsed_action(c)
        if action is None or blob is None:
            out.append(-1.0)
            continue
        try:
            ep_env = restore_env(blob)
            obs = ep_env.step(action)
            r = float(obs.reward or 0.0)
        except Exception:
            r = -1.0
        out.append(max(-2.0, min(2.0, r)))
    return out


def reward_avoids_exfil_path(
    prompts,
    completions,
    valid_assets: Optional[Sequence] = None,
    alert_count: Optional[Sequence[int]] = None,
    **kw,
) -> List[float]:
    """Small bonus for active containment when alerts are firing.

    Pure shaping prior: if the observation has any alerts, mild reward
    for picking ISOLATE_ASSET / BLOCK_EGRESS / REVOKE_IDENTITY, smaller
    reward for INVESTIGATE, zero otherwise. Helps GRPO discover that
    containment exists in the action space; the env reward then takes
    over and tells the policy *which* asset to pick.
    """

    n = len(completions)
    alert_count = _coerce_list(alert_count, n, 0)
    containment = {
        ActionType.ISOLATE_ASSET,
        ActionType.BLOCK_EGRESS,
        ActionType.REVOKE_IDENTITY,
    }
    out: List[float] = []
    for c, n_alerts in zip(completions, alert_count):
        action = parsed_action(c)
        if action is None:
            out.append(0.0)
            continue
        if (n_alerts or 0) > 0 and action.action_type in containment:
            out.append(0.5)
        elif (n_alerts or 0) > 0 and action.action_type is ActionType.INVESTIGATE:
            out.append(0.3)
        else:
            out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# Anti-collapse rewards (added in iter-2)
# ---------------------------------------------------------------------------


def reward_action_diversity(prompts, completions, **kw) -> List[float]:
    """Reward each completion by the inverse of how many siblings copy it.

    GRPO calls reward funcs with the full batch of ``B*K`` completions per
    step. Within each prompt's group of ``K`` candidates, this reward
    returns ``1 / count_of_same_action`` for each completion. If all four
    candidates output the same ``ISOLATE_ASSET secrets-vault``, each one
    gets ``0.25``; if every candidate is unique, each gets ``1.0``.

    Why this directly counters the iter-1 reward hack:

        Iter-1 trained policy collapsed to a single deterministic action
        sequence (``std_return = 0.0`` on 2/3 scenarios across 50 seeds).
        Mode-collapse like that is rewarded in expectation by the six
        task-rewards because the canned plan is genuinely high-EV. Adding
        this term means the *gradient* prefers candidates that diverge
        from their group siblings, breaking the symmetry that lets the
        policy converge to a single text output.

    Unparseable completions get ``0.0`` so this term cannot rescue
    garbage output.
    """

    n = len(completions)
    if n == 0:
        return []
    by_prompt: dict[Any, List[int]] = {}
    for i, p in enumerate(prompts):
        key = p if isinstance(p, str) else json.dumps(p, sort_keys=True, default=str)
        by_prompt.setdefault(key, []).append(i)
    actions: List[Optional[tuple]] = []
    for c in completions:
        a = parsed_action(c)
        if a is None:
            actions.append(None)
        else:
            actions.append((a.action_type.value, getattr(a, "target", None) or ""))
    out = [0.0] * n
    for indices in by_prompt.values():
        group_actions = [actions[i] for i in indices]
        for idx in indices:
            a = actions[idx]
            if a is None:
                out[idx] = 0.0
            else:
                same = sum(1 for x in group_actions if x == a)
                out[idx] = 1.0 / max(1, same)
    return out


def reward_observation_aware(
    prompts,
    completions,
    valid_assets: Optional[Sequence] = None,
    alert_count: Optional[Sequence[int]] = None,
    **kw,
) -> List[float]:
    """Reward state-conditioned behaviour: do something when alerts fire.

    Penalises the trivial degenerate policy "always MONITOR regardless of
    state" and the iter-1 hack "always emit the same containment regardless
    of state". Concretely:

      * alerts present, action != MONITOR, target in valid_assets   -> 1.0
      * alerts present, action == MONITOR                           -> 0.0
      * no alerts present, action == MONITOR                        -> 0.7
      * no alerts present, action != MONITOR, target in valid_assets-> 0.4
      * unparseable / invalid                                       -> 0.0

    The asymmetry between the "alerts present" rows is what makes the
    policy condition on the observation: it cannot collect this reward
    without reading ``alert_count``.
    """

    n = len(completions)
    valid_assets = _coerce_list(valid_assets, n, [])
    alert_count = _coerce_list(alert_count, n, 0)
    out: List[float] = []
    for c, va, n_alerts in zip(completions, valid_assets, alert_count):
        action = parsed_action(c)
        if action is None:
            out.append(0.0)
            continue
        target_ok = action.action_type is ActionType.MONITOR or (
            action.target in (va or [])
        )
        if not target_ok:
            out.append(0.0)
            continue
        alerts = int(n_alerts or 0)
        if alerts > 0:
            out.append(0.0 if action.action_type is ActionType.MONITOR else 1.0)
        else:
            out.append(0.7 if action.action_type is ActionType.MONITOR else 0.4)
    return out


# ---------------------------------------------------------------------------
# Canonical bundle
# ---------------------------------------------------------------------------


def default_reward_funcs() -> List[Callable[..., List[float]]]:
    """Canonical reward function list used by the unified GRPO notebook.

    Order is stable so ``training_log.json`` columns line up across runs.
    The first six entries are the original iter-1 task rewards; the last
    two are iter-2 anti-collapse additions and can be removed for a pure
    iter-1 reproduction.
    """

    return [
        reward_json_valid,
        reward_schema_valid,
        reward_target_in_valid_targets,
        reward_no_redundant_containment,
        reward_step_total,
        reward_avoids_exfil_path,
        reward_action_diversity,
        reward_observation_aware,
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "default_reward_funcs",
    "parse_first_json_object",
    "parsed_action",
    "render_observation",
    "restore_env",
    "reward_action_diversity",
    "reward_avoids_exfil_path",
    "reward_json_valid",
    "reward_no_redundant_containment",
    "reward_observation_aware",
    "reward_schema_valid",
    "reward_step_total",
    "reward_target_in_valid_targets",
    "snapshot_env",
]
