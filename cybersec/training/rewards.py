"""TRL-compatible reward functions for GRPO training of a defender LLM.

This module defines reward signals used with Hugging Face TRL's
``GRPOTrainer``. The live environment in ``cybersec.server`` does not import
this module; training-time rewards are computed from LLM completions and
optional dataset columns (snapshots, validity lists, etc.).

**Env-aligned signal:** :func:`reward_step_total` replays each candidate
action against a pickled in-process environment and returns the scalar step
reward from the server reward model.

**Completion-only signals:** JSON validity, schema checks, target
membership, and similar terms provide dense feedback that does not require
a full env step.

**Dispersion signals:** :func:`reward_action_diversity`,
:func:`reward_observation_aware`, and :func:`reward_batch_action_entropy`
encourage varied, state-conditioned outputs and mitigate mode collapse in
group-relative GRPO updates.

All reward functions follow TRL's expected signature::

    fn(prompts, completions, **kw_from_dataset_columns) -> List[float]

Dataset columns (``valid_assets``, ``env_snapshot``, ...) are passed
through as keyword arguments. Missing keys are padded with ``None`` or
empty lists per row.
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


def collect_grpo_rows_from_rollouts(
    scenario_ids: Sequence[str],
    seeds: Sequence[int],
    policy: Any,
    *,
    max_rows: int,
    shuffle_seed: int = 0,
) -> List[dict]:
    """Build GRPO training rows by rolling ``policy`` through :class:`CybersecEnvironment`.

    Schema matches the heuristic dataset builder: ``prompt``, ``system``,
    ``valid_*`` metadata columns, and ``env_snapshot`` for
    :func:`reward_step_total`. Only includes states **before** each ``step``
    (snapshot taken, then policy acts, then env steps).

    ``policy`` must implement ``act(obs) -> CybersecAction`` and optional
    ``reset()``.
    """

    from ..server.cybersec_environment import CybersecEnvironment

    rows: List[dict] = []
    for sid in scenario_ids:
        for seed in seeds:
            ep_env = CybersecEnvironment()
            if hasattr(policy, "reset"):
                policy.reset()
            obs = ep_env.reset(seed=seed, scenario_id=sid)
            while not obs.done:
                blob = snapshot_env(ep_env)
                prompt = render_observation(obs)
                rows.append({
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "valid_assets": obs.valid_targets["assets"],
                    "valid_identities": obs.valid_targets["identities"],
                    "isolated_assets": list(obs.isolated_assets),
                    "revoked_identities": list(obs.revoked_identities),
                    "blocked_egress": list(obs.blocked_egress_assets),
                    "patched": list(obs.patched_assets),
                    "alert_count": len(obs.alerts),
                    "env_snapshot": blob,
                })
                act = policy.act(obs)
                obs = ep_env.step(act)
    rng = __import__("random").Random(shuffle_seed)
    rng.shuffle(rows)
    return rows[:max_rows]


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
        out.append(max(-5.0, min(5.0, r)))
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
# Dispersion and state-conditioning rewards
# ---------------------------------------------------------------------------


def reward_action_diversity(prompts, completions, **kw) -> List[float]:
    """Reward each completion by the inverse of how many siblings copy it.

    GRPO calls reward funcs with the full batch of ``B*K`` completions per
    step. Within each prompt's group of ``K`` candidates, this reward
    returns ``1 / count_of_same_action`` for each completion. If all four
    candidates output the same ``ISOLATE_ASSET secrets-vault``, each one
    gets ``0.25``; if every candidate is unique, each gets ``1.0``.

    Identical actions within a prompt's candidate group share reward mass
    (each receives ``1 / count``), so GRPO's relative comparison favours
    breaking ties between siblings. Unparseable completions receive ``0.0``.
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

    Penalises policies that ignore ``alert_count`` (e.g. always ``MONITOR``
    under active alerts, or always isolating when quiet). Concretely:

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


def reward_batch_action_entropy(prompts, completions, **kw) -> List[float]:
    """Per-completion bonus from action rarity over the full training batch.

    Within-prompt diversity (:func:`reward_action_diversity`) is flat when
    all candidates in a group match. This term pools actions across the
    entire batch and assigns ``log(N / count_of_action)``, normalised to
    ``[0, 1]``, then scaled by a constant so its magnitude is comparable to
    other dense rewards. Values lie in ``[0, 2]`` with the default scale.
    """

    import math

    n = len(completions)
    if n == 0:
        return []
    actions: List[Optional[tuple]] = []
    for c in completions:
        a = parsed_action(c)
        if a is None:
            actions.append(None)
        else:
            actions.append((a.action_type.value, getattr(a, "target", None) or ""))
    counts: dict[tuple, int] = {}
    for a in actions:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    out: List[float] = []
    for a in actions:
        if a is None:
            out.append(0.0)
            continue
        c = counts[a]
        # log(n / c): n when c=1 (max bonus), 0 when c=n (full collapse)
        v = math.log(max(1, n) / c) / math.log(max(2, n))
        out.append(max(0.0, min(1.0, v)))
    # Scale so GRPO gradient from this term is comparable to dense format rewards.
    scale = 2.0
    return [x * scale for x in out]


# ---------------------------------------------------------------------------
# Canonical bundle
# ---------------------------------------------------------------------------


def default_reward_funcs() -> List[Callable[..., List[float]]]:
    """Ordered list of reward functions for GRPO training.

    Order is stable so exported training logs remain comparable across runs.
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
    "collect_grpo_rows_from_rollouts",
    "default_reward_funcs",
    "parse_first_json_object",
    "parsed_action",
    "render_observation",
    "restore_env",
    "reward_action_diversity",
    "reward_avoids_exfil_path",
    "reward_batch_action_entropy",
    "reward_json_valid",
    "reward_no_redundant_containment",
    "reward_observation_aware",
    "reward_schema_valid",
    "reward_step_total",
    "reward_target_in_valid_targets",
    "snapshot_env",
]
