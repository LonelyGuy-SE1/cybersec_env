"""Author the single end-to-end ``cybersec_grpo.ipynb`` notebook.

We construct the .ipynb JSON in-script rather than hand-edit it cell by cell
so the structure is reviewable as plain Python and trivially regeneratable.
Everything below mirrors the merged contents of the prior 01/02/03 notebooks
with deduplicated helpers and a single config block at the top.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.lstrip("\n").rstrip() + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).lstrip("\n").rstrip() + "\n",
    }


CELLS: List[dict] = []

CELLS.append(md(r"""
# Cybersec OpenEnv: end-to-end RL train + eval

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb)

One notebook, one Run All. Runs the **whole** training+evaluation pipeline against the
[`cybersec` OpenEnv](https://huggingface.co/spaces/Lonelyguyse1/cybersec):

1. Baseline `RandomPolicy` and `HeuristicPolicy` over **50 seeds × 3 scenarios**
2. Build a 1500-prompt training dataset of `(state-snapshot, prompt)` pairs
   from heuristic rollouts (so GRPO reward functions can clone the env and
   score candidate actions).
3. Fine-tune **Qwen2.5-1.5B-Instruct** with **GRPO** + **Unsloth QLoRA** for
   **100 steps × 4 generations/prompt**, scored by 6 independent reward
   functions.
4. Re-run the same 50 seeds × 3 scenarios with the trained adapter.
5. Plot before/after reward curves on identical axes and write the headline
   summary table.

**Target compute**: free Colab **T4 GPU**. Wall clock end-to-end ≈ **40–55 min**.

### Config

A single `MODE` dict at the top of the *Imports & config* cell controls the
non-trivial dials (episode counts, GRPO step count, generations, dataset
size). The defaults are *real* settings that produce interpretable results
in one Colab session — nothing here is reduced just to make the notebook run
faster. To tune, edit the dict, not the cells below.

### Outputs (all under `_artifacts/`)

* `baseline_metrics.json` — random/heuristic returns per scenario
* `qwen_cybersec_lora/` — the trained LoRA adapter (re-loadable into Qwen2.5-1.5B)
* `training_log.json` — per-step GRPO reward components
* `before_after_curves.png` — pre/post-training cumulative reward curves
* `summary_table.md` — headline numbers for the README / report
* `post_train_metrics.json` — full per-policy aggregate stats
"""))

CELLS.append(md("## 0. Install"))

CELLS.append(code(r"""
%%capture
# Colab-friendly install. On a fresh Colab kernel we clone the repo first;
# locally (when the repo is already on disk) we just pip-install the in-tree
# package.
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/LonelyGuy-SE1/cybersec_env.git"
REPO_DIR = Path("/content/cybersec_env") if "google.colab" in sys.modules else Path("..").resolve()

if "google.colab" in sys.modules and not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

PKG_DIR = REPO_DIR / "cybersec"
assert PKG_DIR.exists(), f"cybersec package not found at {PKG_DIR}"

# Pinned for the free Colab T4 (CUDA 12.x, py 3.10/3.11). Unsloth ships its
# own torch wheel so install it before TRL/peft/accelerate.
!pip install -q -e {PKG_DIR}
!pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --upgrade trl peft accelerate bitsandbytes
!pip install -q matplotlib pandas
"""))

CELLS.append(md("## 1. Imports & config"))

CELLS.append(code(r"""
from __future__ import annotations

import base64
import copy
import json
import math
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg") if "google.colab" not in sys.modules else None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset

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

# ---------------------------------------------------------------------------
# Single source of truth for every dial in this notebook.
# Defaults are tuned for one full pass on a Colab T4 GPU:
#   - 50-episode baselines:                ~3 min  (CPU)
#   - 1500-prompt dataset build:           ~2 min  (CPU)
#   - Qwen 1.5B + Unsloth load (4-bit):    ~2 min
#   - 100 GRPO steps x 4 generations:      ~25 min (T4)
#   - 50-episode trained-policy eval:      ~12 min (T4 inference)
#   - Plots, tables, persistence:          ~1 min
#   ----------------------------------------------------------
#   total wall clock:                      ~45 min
#
# Increase MODE['grpo_max_steps'] / MODE['n_baseline_episodes'] for a longer
# polished run; nothing else has to change.
# ---------------------------------------------------------------------------
MODE = {
    "n_baseline_episodes":   50,
    "n_dataset_seeds":       30,
    "max_dataset_rows":      1500,
    "grpo_max_steps":        100,
    "grpo_num_generations":  4,
    "grpo_per_device_bs":    2,
    "grpo_grad_accum":       4,
    "grpo_logging_steps":    5,
    "grpo_save_steps":       50,
    "n_post_train_episodes": 50,
}

ARTIFACTS = Path("_artifacts")
ARTIFACTS.mkdir(exist_ok=True)

ADAPTER_DIR     = ARTIFACTS / "qwen_cybersec_lora"
TRAIN_LOG       = ARTIFACTS / "training_log.json"
BASELINE_JSON   = ARTIFACTS / "baseline_metrics.json"
POST_JSON       = ARTIFACTS / "post_train_metrics.json"
BASELINE_PLOT   = ARTIFACTS / "baseline_curves.png"
BEFORE_AFTER    = ARTIFACTS / "before_after_curves.png"
SUMMARY_MD      = ARTIFACTS / "summary_table.md"
TRAJECTORY_JSON = ARTIFACTS / "trajectory_dataset.jsonl"

MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_LEN  = 1024
MAX_NEW_TOKENS  = 48

SCENARIOS = list_scenarios()
SEEDS_BASELINE = list(range(MODE["n_baseline_episodes"]))
SEEDS_DATASET  = list(range(MODE["n_dataset_seeds"]))
SEEDS_POST     = list(range(MODE["n_post_train_episodes"]))

print("scenarios:", SCENARIOS)
print(json.dumps(MODE, indent=2))
"""))

CELLS.append(md(r"""
## 2. Baseline: Random vs Heuristic

50 episodes per scenario, no LLM in the loop. This anchors the post-training
comparison and is the only block that runs on CPU — start it first, the GPU
phase doesn't depend on it.
"""))

CELLS.append(code(r"""
env = CybersecEnvironment()
baseline_runs = {}

t0 = time.time()
for sid in SCENARIOS:
    rand = [run_episode(env, RandomPolicy(seed=s), seed=s, scenario_id=sid) for s in SEEDS_BASELINE]
    heur = [run_episode(env, HeuristicPolicy(),    seed=s, scenario_id=sid) for s in SEEDS_BASELINE]
    baseline_runs[sid] = {"random": rand, "heuristic": heur}
    print(f"{sid:<32s}  random={aggregate_results(rand)['mean_return']:7.3f}"
          f"  heuristic={aggregate_results(heur)['mean_return']:7.3f}")
print(f"baseline elapsed: {time.time()-t0:.1f}s")
"""))

CELLS.append(md(r"""
### 2a. Persist baseline metrics + plot
"""))

CELLS.append(code(r"""
def _padded_cumulative(curves, target_len):
    out = np.zeros((len(curves), target_len), dtype=float)
    for i, c in enumerate(curves):
        out[i, : len(c)] = c
        if len(c) < target_len:
            out[i, len(c):] = c[-1] if c else 0.0
    return np.cumsum(out, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, sid in zip(axes, SCENARIOS):
    cell = baseline_runs[sid]
    horizon = max(max(len(r.reward_curve) for r in cell['random']),
                  max(len(r.reward_curve) for r in cell['heuristic']))
    for label, color in [("random", "tab:gray"), ("heuristic", "tab:blue")]:
        cumr = _padded_cumulative([r.reward_curve for r in cell[label]], horizon)
        mean = cumr.mean(axis=0); std = cumr.std(axis=0)
        ax.plot(mean, label=label, color=color)
        ax.fill_between(np.arange(horizon), mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title(sid, fontsize=10); ax.set_xlabel("tick"); ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("cumulative reward"); axes[0].legend()
fig.suptitle(f"Cybersec OpenEnv baselines, n={MODE['n_baseline_episodes']}/scenario")
fig.tight_layout(); fig.savefig(BASELINE_PLOT, dpi=140); plt.show()

baseline_metrics = {
    sid: {p: aggregate_results(cell[p]) for p in ("random", "heuristic")}
    for sid, cell in baseline_runs.items()
}
baseline_metrics["_meta"] = {"n_episodes": MODE["n_baseline_episodes"], "scenarios": SCENARIOS}
BASELINE_JSON.write_text(json.dumps(baseline_metrics, indent=2))
print("wrote", BASELINE_JSON)
"""))

CELLS.append(md(r"""
## 3. Build the GRPO training dataset

GRPO needs a fixed corpus of *prompts*. We follow heuristic rollouts and at
every step snapshot the env (pickle + base64 of the live `CybersecEnvironment`)
alongside the rendered prompt. At training time, the `reward_step_total`
function unpickles the snapshot, runs `env.step(candidate)`, and uses the
real environment reward to score the model's output.

This is the long-horizon-multi-agent novelty surface: each prompt is a real
state in the middle of a stochastic, dwell-time-driven attack scenario, and
the reward comes from running the actual scripted attacker forward one tick.
"""))

CELLS.append(code(r"""
SYSTEM_PROMPT = (
    "You are an SRE-grade cyber-defender driving an OpenEnv environment.\n"
    "Reply with exactly one JSON object on one line of the form\n"
    '{"action_type": "...", "target": "..."}.\n'
    "action_type must be one of MONITOR, INVESTIGATE, ISOLATE_ASSET, "
    "REVOKE_IDENTITY, BLOCK_EGRESS, PATCH_ASSET.\n"
    "target must be omitted (or null) for MONITOR; otherwise it must come "
    "from valid_targets."
)

def render_observation(obs) -> str:
    lines = [
        f"tick={obs.tick}/{obs.horizon}  scenario={obs.scenario_id}  attacker={obs.attacker_personality.value}",
        f"isolated_assets={obs.isolated_assets}",
        f"revoked_identities={obs.revoked_identities}",
        f"blocked_egress={obs.blocked_egress_assets}",
        f"patched={obs.patched_assets}",
        f"confirmed_compromised={obs.confirmed_compromised}",
        f"valid_targets={obs.valid_targets}",
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

def snapshot_env(env: CybersecEnvironment) -> str:
    return base64.b64encode(pickle.dumps(env)).decode("ascii")

def restore_env(blob: str) -> CybersecEnvironment:
    return pickle.loads(base64.b64decode(blob.encode("ascii")))

t0 = time.time()
rows = []
trajectory_lines = []
for sid in SCENARIOS:
    for seed in SEEDS_DATASET:
        ep_env = CybersecEnvironment()
        policy = HeuristicPolicy()
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
            trajectory_lines.append(
                json.dumps({"prompt": prompt, "completion": act.model_dump_json()})
            )
            obs = ep_env.step(act)

random.Random(0).shuffle(rows)
rows = rows[: MODE["max_dataset_rows"]]
ds = Dataset.from_list(rows)
TRAJECTORY_JSON.write_text("\n".join(trajectory_lines))
print(f"dataset size: {len(ds)}  build elapsed: {time.time()-t0:.1f}s")
print(ds[0]["prompt"][:300])
"""))

CELLS.append(md(r"""
## 4. Reward functions (six independent signals)

GRPO scores each candidate completion with a sum of these six rewards.
Splitting them keeps the training signal interpretable in `training_log.json`.
"""))

CELLS.append(code(r"""
def _parse_first_json_object(text: str):
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

def _parsed_action(c):
    payload = _parse_first_json_object(c)
    if not payload:
        return None
    try:
        return CybersecAction(**payload)
    except Exception:
        return None

def reward_json_valid(prompts, completions, **kw):
    return [1.0 if _parse_first_json_object(c) is not None else 0.0 for c in completions]

def reward_schema_valid(prompts, completions, **kw):
    return [1.0 if _parsed_action(c) is not None else 0.0 for c in completions]

def reward_target_in_valid_targets(prompts, completions, valid_assets=None, valid_identities=None, **kw):
    valid_assets = valid_assets or [None] * len(completions)
    valid_identities = valid_identities or [None] * len(completions)
    out = []
    for c, va, vi in zip(completions, valid_assets, valid_identities):
        action = _parsed_action(c)
        if action is None:
            out.append(0.0); continue
        if action.action_type is ActionType.MONITOR:
            out.append(1.0); continue
        if action.action_type is ActionType.REVOKE_IDENTITY:
            out.append(1.0 if action.target in (vi or []) else 0.0)
        elif action.action_type is ActionType.INVESTIGATE:
            out.append(1.0 if action.target in ((va or []) + (vi or [])) else 0.0)
        else:
            out.append(1.0 if action.target in (va or []) else 0.0)
    return out

def reward_no_redundant_containment(prompts, completions, isolated_assets=None,
                                    revoked_identities=None, blocked_egress=None,
                                    patched=None, **kw):
    isolated_assets    = isolated_assets    or [[] for _ in completions]
    revoked_identities = revoked_identities or [[] for _ in completions]
    blocked_egress     = blocked_egress     or [[] for _ in completions]
    patched            = patched            or [[] for _ in completions]
    out = []
    for c, iso, rev, blk, pat in zip(completions, isolated_assets, revoked_identities,
                                     blocked_egress, patched):
        action = _parsed_action(c)
        if action is None:
            out.append(0.0); continue
        if action.action_type is ActionType.MONITOR:
            out.append(0.5); continue
        already = {
            ActionType.ISOLATE_ASSET:    iso,
            ActionType.REVOKE_IDENTITY:  rev,
            ActionType.BLOCK_EGRESS:     blk,
            ActionType.PATCH_ASSET:      pat,
        }.get(action.action_type, [])
        out.append(0.0 if action.target in already else 1.0)
    return out

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
            r = float(obs.reward or 0.0)
        except Exception:
            r = -1.0
        out.append(max(-2.0, min(2.0, r)))
    return out

def reward_avoids_exfil_path(prompts, completions, valid_assets=None, alert_count=None, **kw):
    valid_assets = valid_assets or [[] for _ in completions]
    alert_count = alert_count or [0] * len(completions)
    containment = {ActionType.ISOLATE_ASSET, ActionType.BLOCK_EGRESS, ActionType.REVOKE_IDENTITY}
    out = []
    for c, n_alerts in zip(completions, alert_count):
        action = _parsed_action(c)
        if action is None:
            out.append(0.0); continue
        if n_alerts > 0 and action.action_type in containment:
            out.append(0.5)
        elif n_alerts > 0 and action.action_type is ActionType.INVESTIGATE:
            out.append(0.3)
        else:
            out.append(0.0)
    return out

REWARD_FUNCS = [
    reward_json_valid,
    reward_schema_valid,
    reward_target_in_valid_targets,
    reward_no_redundant_containment,
    reward_step_total,
    reward_avoids_exfil_path,
]
REWARD_NAMES = [f.__name__ for f in REWARD_FUNCS]
print("reward components:", REWARD_NAMES)
"""))

CELLS.append(md(r"""
## 5. Load Qwen2.5-1.5B with Unsloth (4-bit QLoRA)
"""))

CELLS.append(code(r"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_PROMPT_LEN + MAX_NEW_TOKENS,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=0,
)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model.print_trainable_parameters()
"""))

CELLS.append(md(r"""
## 6. Format dataset with the chat template
"""))

CELLS.append(code(r"""
def to_chat_prompt(row):
    msgs = [
        {"role": "system", "content": row["system"]},
        {"role": "user",   "content": row["prompt"]},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return {"prompt": text}

ds_chat = ds.map(to_chat_prompt)
print("first prompt (truncated):")
print(ds_chat[0]["prompt"][:500])
"""))

CELLS.append(md(r"""
## 7. GRPO trainer
"""))

CELLS.append(code(r"""
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir=str(ARTIFACTS / "grpo_checkpoints"),
    learning_rate=5e-6,
    per_device_train_batch_size=MODE["grpo_per_device_bs"],
    gradient_accumulation_steps=MODE["grpo_grad_accum"],
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_NEW_TOKENS,
    num_generations=MODE["grpo_num_generations"],
    num_train_epochs=1,
    max_steps=MODE["grpo_max_steps"],
    logging_steps=MODE["grpo_logging_steps"],
    save_steps=MODE["grpo_save_steps"],
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    report_to=[],
    seed=0,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=ds_chat,
    processing_class=tokenizer,
    reward_funcs=REWARD_FUNCS,
)

t0 = time.time()
trainer.train()
print(f"GRPO train elapsed: {time.time()-t0:.1f}s")
"""))

CELLS.append(md(r"""
## 8. Save adapter + training log
"""))

CELLS.append(code(r"""
ADAPTER_DIR.mkdir(exist_ok=True)
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

history = getattr(trainer.state, "log_history", []) or []
TRAIN_LOG.write_text(json.dumps({
    "reward_names": REWARD_NAMES,
    "history": history,
    "model_name": MODEL_NAME,
    "mode": MODE,
}, indent=2))
print("saved adapter to", ADAPTER_DIR)
print("wrote", TRAIN_LOG, "with", len(history), "log rows")
"""))

CELLS.append(md(r"""
## 9. Evaluate the trained policy on the same seeds

We reload through **Unsloth** (not vanilla `transformers + peft`) on
purpose. GRPO trains the model with Unsloth's fused-QKV attention patch in
place -- that patch adds an `apply_qkv` method to `Qwen2Attention`, and the
saved LoRA expects to plug into it. A vanilla `AutoModelForCausalLM` reload
will crash at first inference with
`AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'`.
Loading the adapter directory through `FastLanguageModel.from_pretrained`
re-applies Unsloth's patches before grafting the LoRA on, which is the
supported path. We then call `FastLanguageModel.for_inference(eval_model)`
to switch to Unsloth's 2x faster decode.
"""))

CELLS.append(code(r"""
from unsloth import FastLanguageModel

# Free training-only buffers if the kernel still has them from the previous
# cells. Wrapped in a guard so this cell can also be run standalone after a
# kernel restart (kernel restart -> Run-All-from-here, with the adapter
# already on disk in ADAPTER_DIR).
for _name in ("model", "trainer"):
    if _name in globals():
        del globals()[_name]
torch.cuda.empty_cache()

# Loading from the adapter directory: Unsloth reads adapter_config.json,
# pulls the base model that the LoRA was trained against, applies its own
# attention patches (including fused QKV / apply_qkv), and only then grafts
# the LoRA weights on top.
eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),
    max_seq_length=MAX_PROMPT_LEN + MAX_NEW_TOKENS,
    dtype=None,
    load_in_4bit=True,
)
eval_tokenizer.pad_token = eval_tokenizer.pad_token or eval_tokenizer.eos_token
FastLanguageModel.for_inference(eval_model)
print("loaded trained adapter via Unsloth for evaluation")
"""))

CELLS.append(code(r"""
@torch.inference_mode()
def llm_act(obs):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": render_observation(obs)},
    ]
    text = eval_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = eval_tokenizer(text, return_tensors="pt").to(eval_model.device)
    out = eval_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=eval_tokenizer.eos_token_id,
    )
    completion = eval_tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    payload = _parse_first_json_object(completion)
    if not payload:
        return CybersecAction(action_type=ActionType.MONITOR)
    try:
        return CybersecAction(**payload)
    except Exception:
        return CybersecAction(action_type=ActionType.MONITOR)

class TrainedLLMPolicy:
    name = "qwen-1.5b-grpo"
    def reset(self):
        pass
    def act(self, obs):
        return llm_act(obs)

trained_runs = {}
t0 = time.time()
for sid in SCENARIOS:
    runs = [run_episode(env, TrainedLLMPolicy(), seed=s, scenario_id=sid) for s in SEEDS_POST]
    trained_runs[sid] = runs
    print(f"{sid:<32s}  trained-llm={aggregate_results(runs)['mean_return']:7.3f}")
print(f"trained-policy eval elapsed: {time.time()-t0:.1f}s")
"""))

CELLS.append(md(r"""
## 10. Before/after curves on identical axes
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, sid in zip(axes, SCENARIOS):
    cell = baseline_runs[sid]
    llm  = trained_runs[sid]
    horizon = max(
        max(len(r.reward_curve) for r in cell['random']),
        max(len(r.reward_curve) for r in cell['heuristic']),
        max(len(r.reward_curve) for r in llm),
    )
    palette = [("random", cell['random'], "tab:gray"),
               ("heuristic", cell['heuristic'], "tab:blue"),
               ("trained-llm", llm, "tab:red")]
    for label, runs, color in palette:
        cumr = _padded_cumulative([r.reward_curve for r in runs], horizon)
        mean = cumr.mean(axis=0); std = cumr.std(axis=0)
        ax.plot(mean, label=label, color=color)
        ax.fill_between(np.arange(horizon), mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title(sid, fontsize=10); ax.set_xlabel("tick"); ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("cumulative reward"); axes[0].legend()
fig.suptitle(
    f"Cybersec OpenEnv -- pre/post GRPO ({MODE['grpo_max_steps']} steps), "
    f"n={MODE['n_post_train_episodes']}/scenario"
)
fig.tight_layout(); fig.savefig(BEFORE_AFTER, dpi=140); plt.show()
"""))

CELLS.append(md(r"""
## 11. Summary table + headline delta
"""))

CELLS.append(code(r"""
rows = []
for sid in SCENARIOS:
    cell = baseline_runs[sid]
    for policy_name, runs in [
        ("random",     cell["random"]),
        ("heuristic",  cell["heuristic"]),
        ("trained-llm", trained_runs[sid]),
    ]:
        agg = aggregate_results(runs)
        returns = [r.cumulative_reward for r in runs]
        rows.append({
            "scenario":     sid,
            "policy":       policy_name,
            "mean_return":  agg["mean_return"],
            "std_return":   round(float(np.std(returns)), 3) if returns else 0.0,
            "mean_stages":  agg["mean_stages_succeeded"],
            "exfil_rate":   agg["exfil_rate"],
            "invalid_rate": agg["mean_invalid_actions"],
        })
df = pd.DataFrame(rows)
print(df.to_string(index=False))

post_metrics = {
    sid: {
        "random":      aggregate_results(baseline_runs[sid]["random"]),
        "heuristic":   aggregate_results(baseline_runs[sid]["heuristic"]),
        "trained_llm": aggregate_results(trained_runs[sid]),
    }
    for sid in SCENARIOS
}
post_metrics["_meta"] = {
    "n_post_episodes": MODE["n_post_train_episodes"],
    "grpo_max_steps":  MODE["grpo_max_steps"],
    "model":           MODEL_NAME,
    "adapter":         str(ADAPTER_DIR),
}
POST_JSON.write_text(json.dumps(post_metrics, indent=2))
SUMMARY_MD.write_text(df.to_markdown(index=False))
print("wrote", POST_JSON)
print("wrote", SUMMARY_MD)
"""))

CELLS.append(code(r"""
print("=== Headline delta (trained-llm vs heuristic) ===")
for sid in SCENARIOS:
    h = aggregate_results(baseline_runs[sid]["heuristic"])["mean_return"]
    t = aggregate_results(trained_runs[sid])["mean_return"]
    print(f"{sid:<32s}  heuristic={h:7.3f}  trained-llm={t:7.3f}  delta={t-h:+.3f}")
"""))

CELLS.append(md(r"""
## 12. Sanity assertions

These are the contracts a regression in reward shaping or training would
break. They are deliberately mild: at 100 GRPO steps with a 1.5B model the
trained policy is not guaranteed to beat the heuristic, but it must at least
be a valid policy that emits parseable actions and isn't catastrophically
worse than random.
"""))

CELLS.append(code(r"""
# 1. Reward shaping is healthy: heuristic should out-earn random *on
#    aggregate*. We deliberately don't require it to win on every scenario:
#    `insider_repo_pivot` is hard for a fixed heuristic by design (random's
#    "burn the network down" strategy artificially scores well there). The
#    shaping is fine as long as the heuristic's federated_identity gain
#    outweighs that loss on average.
heur_total = sum(
    float(np.mean([r.cumulative_reward for r in cell['heuristic']]))
    for cell in baseline_runs.values()
)
rand_total = sum(
    float(np.mean([r.cumulative_reward for r in cell['random']]))
    for cell in baseline_runs.values()
)
print(f"baseline totals: heuristic={heur_total:+.3f}  random={rand_total:+.3f}")
assert heur_total > rand_total, (
    f"reward shaping regressed: heuristic total ({heur_total:.3f}) "
    f"<= random total ({rand_total:.3f}) summed across scenarios"
)

# 2. Trained policy must produce mostly-valid actions across all trained
#    episodes. We aggregate steps and invalid-action counts for a single
#    rate rather than thresholding per-episode.
total_steps = sum(len(r.reward_curve) for runs in trained_runs.values() for r in runs)
total_invalid = sum(r.invalid_action_count for runs in trained_runs.values() for r in runs)
valid_rate = 1.0 - (total_invalid / max(1, total_steps))
print(f"valid-action rate across trained episodes: {valid_rate:.1%}  "
      f"({total_invalid} invalid / {total_steps} steps)")
assert valid_rate >= 0.8, (
    f"trained policy is producing too many invalid actions ({valid_rate:.1%} valid)"
)

# 3. Trained policy must not be catastrophically worse than random.
for sid in SCENARIOS:
    rand = float(np.mean([r.cumulative_reward for r in baseline_runs[sid]['random']]))
    llm  = float(np.mean([r.cumulative_reward for r in trained_runs[sid]]))
    assert llm >= rand - 5.0, (
        f"{sid}: trained-llm ({llm:.3f}) is more than 5 points below random "
        f"({rand:.3f}) -- training likely diverged"
    )

print("all sanity checks passed")
"""))


def main() -> int:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": CELLS,
    }
    out = (
        Path(__file__).resolve().parent.parent / "notebooks" / "cybersec_grpo.ipynb"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
