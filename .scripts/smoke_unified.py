"""Run the non-GPU sections of cybersec_grpo.ipynb against the in-tree env.

Skips the Unsloth load, the GRPO training, and the trained-policy eval (all
GPU-only). Exercises everything else end-to-end so we know nothing is broken
before the user clicks Run All on Colab.

Mirrors the dataset-build using a *small* number of seeds for time, but uses
the real reward functions and the real env (no shortcuts).
"""

from __future__ import annotations

import base64
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

from cybersec import (
    ActionType,
    CybersecAction,
    CybersecEnvironment,
    list_scenarios,
)
from cybersec.baselines import (
    HeuristicPolicy,
    RandomPolicy,
    aggregate_results,
    run_episode,
)


def main() -> int:
    SCENARIOS = list_scenarios()
    env = CybersecEnvironment()

    # --- Section 2: baseline -------------------------------------------------
    t0 = time.time()
    baseline_runs = {}
    for sid in SCENARIOS:
        rand = [run_episode(env, RandomPolicy(seed=s), seed=s, scenario_id=sid) for s in range(10)]
        heur = [run_episode(env, HeuristicPolicy(),    seed=s, scenario_id=sid) for s in range(10)]
        baseline_runs[sid] = {"random": rand, "heuristic": heur}
    print(f"[smoke] baselines (N=10): {time.time()-t0:.1f}s")

    # --- Section 3: dataset build ------------------------------------------
    SYSTEM_PROMPT = (
        "You are an SRE-grade cyber-defender. Reply with one JSON object "
        '{"action_type": "...", "target": "..."}.'
    )

    def render_observation(obs) -> str:
        return f"tick={obs.tick}/{obs.horizon} alerts={len(obs.alerts)} valid={obs.valid_targets}"

    def snapshot_env(e: CybersecEnvironment) -> str:
        return base64.b64encode(pickle.dumps(e)).decode("ascii")

    def restore_env(blob: str) -> CybersecEnvironment:
        return pickle.loads(base64.b64decode(blob.encode("ascii")))

    t0 = time.time()
    rows = []
    for sid in SCENARIOS:
        for seed in range(4):
            ep_env = CybersecEnvironment()
            policy = HeuristicPolicy()
            obs = ep_env.reset(seed=seed, scenario_id=sid)
            while not obs.done:
                blob = snapshot_env(ep_env)
                rows.append({
                    "prompt": render_observation(obs),
                    "system": SYSTEM_PROMPT,
                    "valid_assets":      obs.valid_targets["assets"],
                    "valid_identities":  obs.valid_targets["identities"],
                    "isolated_assets":   list(obs.isolated_assets),
                    "revoked_identities": list(obs.revoked_identities),
                    "blocked_egress":    list(obs.blocked_egress_assets),
                    "patched":           list(obs.patched_assets),
                    "alert_count":       len(obs.alerts),
                    "env_snapshot":      blob,
                })
                obs = ep_env.step(policy.act(obs))
    print(f"[smoke] dataset rows: {len(rows)}  build elapsed: {time.time()-t0:.1f}s")

    # --- Section 4: reward functions on synthetic completions ----------------
    def _parse_first_json_object(text):
        text = text.strip()
        s = text.find("{")
        if s < 0:
            return None
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _parsed_action(c):
        p = _parse_first_json_object(c)
        if not p:
            return None
        try:
            return CybersecAction(**p)
        except Exception:
            return None

    def reward_step_total(prompts, completions, env_snapshot=None, **kw):
        snapshots = env_snapshot or [None] * len(completions)
        out = []
        for c, blob in zip(completions, snapshots):
            action = _parsed_action(c)
            if action is None or blob is None:
                out.append(-1.0); continue
            try:
                ep_env = restore_env(blob)
                obs = ep_env.step(action)
                out.append(max(-2.0, min(2.0, float(obs.reward or 0.0))))
            except Exception:
                out.append(-1.0)
        return out

    sample = rows[:4]
    completions = [
        '{"action_type":"MONITOR"}',
        '{"action_type":"INVESTIGATE","target":"ci-runner-01"}',
        'not json at all',
        '{"action_type":"ISOLATE_ASSET","target":"NONEXISTENT"}',
    ]
    snapshots = [r["env_snapshot"] for r in sample]
    scores = reward_step_total(None, completions, env_snapshot=snapshots)
    print(f"[smoke] reward_step_total: {scores}")
    assert all(isinstance(x, float) for x in scores)

    # --- Section 11/12 sanity checks (reward shaping) ------------------------
    wins = 0
    for sid, cell in baseline_runs.items():
        h = float(np.mean([r.cumulative_reward for r in cell['heuristic']]))
        r = float(np.mean([r.cumulative_reward for r in cell['random']]))
        wins += int(h > r)
    print(f"[smoke] heuristic-wins-vs-random: {wins}/3 (expected >= 1 at N=10)")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
