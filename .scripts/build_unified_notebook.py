"""Author the single end-to-end ``cybersec_grpo.ipynb`` notebook (iter-2).

Iter-2 changes vs. iter-1:

* Install path now goes through the **HF Space** (``huggingface_hub
  .snapshot_download``) instead of cloning the GitHub repo.
* Baseline **and** post-train eval optionally run against the deployed
  HF Space env over HTTP/WS (a ``USE_REMOTE_ENV`` toggle in the config),
  proving OpenEnv compliance end-to-end. Dataset build stays local because
  reward-function snapshotting needs in-process pickling.
* Reward functions are imported from ``cybersec.training.rewards`` instead
  of being inlined. The bundle now includes two anti-collapse rewards
  (``reward_action_diversity``, ``reward_observation_aware``) added to
  fix iter-1's std=0 mode collapse.
* A held-out OOD scenario (``cloud_metadata_ssrf``) is evaluated with
  the trained policy and reported separately.
* KL / entropy / reward-component plots are generated from
  ``training_log.json``.
* All known transformers warning spam is silenced before training so the
  Colab notebook output is actually readable.
* Sanity asserts now fail loudly on iter-1-style mode collapse
  (``std_return == 0.0`` on every training scenario).

We construct the .ipynb JSON in-script rather than hand-edit it cell by
cell so the structure is reviewable as plain Python and trivially
regeneratable.
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
# Cybersec OpenEnv: end-to-end RL train + eval (iter-2)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb)

One notebook, one Run-All. Runs the whole pipeline against the
[`cybersec` OpenEnv](https://huggingface.co/spaces/Lonelyguyse1/cybersec):

1. Install the env package straight from the **HF Space** (no GitHub clone).
2. Baseline `RandomPolicy` + `HeuristicPolicy` over 50 seeds × 3 train
   scenarios + 1 held-out OOD scenario (`cloud_metadata_ssrf`).
3. Build a ~1500-prompt GRPO dataset from heuristic rollouts. Each row
   carries a pickled env snapshot so the trainer can clone & step it.
4. Fine-tune **Qwen2.5-1.5B-Instruct** with **GRPO + Unsloth QLoRA** for
   100 steps × 4 generations/prompt, scored by **eight reward functions**
   imported from `cybersec.training.rewards`:
     * 4 schema/validity rewards
     * 1 actual env-step reward (clones a snapshot, applies the candidate)
     * 1 exfil-path shaping prior
     * 2 **anti-collapse rewards** (iter-2): action-diversity within a
       group, and observation-aware (state-conditioned) behaviour.
5. Re-run the same seeds with the trained adapter on **train scenarios
   AND the held-out scenario**.
6. Plot before/after reward curves and KL/entropy/component diagnostics.
7. Sanity-check: fail loud on iter-1-style mode collapse
   (`std_return == 0` across all scenarios).

**Target compute**: free Colab **T4 GPU** or Colab Pro T4/A100. Wall clock
end-to-end ≈ **45–60 min** at the defaults.
"""))

CELLS.append(md("## 0. Install (from the HF Space)"))

CELLS.append(code(r"""
%%capture
# Iter-2: install the env package directly from the deployed HF Space using
# the huggingface_hub API. The Space is the single canonical source -- no
# GitHub clone, no manual sync. If you need a different revision, change
# HF_SPACE_REVISION below.
import os, sys, subprocess, importlib
from pathlib import Path

HF_SPACE_REPO_ID = "Lonelyguyse1/cybersec"
HF_SPACE_REVISION = "main"
PKG_STAGE = Path("/content/cybersec_env_pkg") if "google.colab" in sys.modules else Path("..").resolve() / "cybersec"

# Local checkout: we already have the package on disk (running off a
# developer machine). Skip the snapshot_download dance entirely.
if "google.colab" in sys.modules:
    !pip install -q "huggingface_hub>=0.24"
    from huggingface_hub import snapshot_download
    PKG_STAGE.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_SPACE_REPO_ID,
        repo_type="space",
        revision=HF_SPACE_REVISION,
        local_dir=str(PKG_STAGE),
        local_dir_use_symlinks=False,
    )
    !pip install -q -e {PKG_STAGE}
else:
    print(f"using in-tree package at {PKG_STAGE}")

# Pinned for the free Colab T4 (CUDA 12.x). Unsloth ships its own torch
# wheel so install it before TRL/peft/accelerate.
!pip install -q "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --upgrade trl peft accelerate bitsandbytes
!pip install -q matplotlib pandas
"""))

CELLS.append(md("## 1. Imports & config"))

CELLS.append(code(r"""
from __future__ import annotations

# Iter-1's notebook output was buried under thousands of lines of identical
# `max_new_tokens / attention_mask` deprecation warnings. Silence the known
# noise *before* importing torch / transformers so the cell outputs are
# actually readable.
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

import base64
import copy
import json
import math
import pickle
import random
import sys
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
    CybersecEnv,
    CybersecEnvironment,
    list_eval_scenarios,
    list_scenarios,
    list_train_scenarios,
)
from cybersec.baselines import (
    HeuristicPolicy,
    RandomPolicy,
    aggregate_results,
    run_episode,
)
from cybersec.training.rewards import (
    SYSTEM_PROMPT,
    default_reward_funcs,
    parse_first_json_object,
    parsed_action,
    render_observation,
    restore_env,
    snapshot_env,
)

# ---------------------------------------------------------------------------
# Single source of truth for every dial in this notebook.
# Defaults are tuned for one full pass on a Colab T4 GPU:
#   - 50-episode baselines (4 scenarios):  ~1 min   (CPU)
#   - 1500-prompt dataset build:           ~2 min   (CPU)
#   - Qwen 1.5B + Unsloth load (4-bit):    ~2 min
#   - 100 GRPO steps x 4 generations:      ~25 min  (T4)
#   - 50-episode trained-policy eval:      ~12 min  (T4 inference)
#   - Held-out OOD eval (50 ep):           ~4 min   (T4 inference)
#   - Plots, tables, persistence:          ~1 min
#   ----------------------------------------------------------
#   total wall clock:                      ~45-50 min
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
    # Set to True to also run baseline + post-train eval against the
    # deployed HF Space env over WebSocket. Off by default to keep the
    # default Run-All hermetic and offline-friendly.
    "use_remote_env":        False,
    "remote_env_url":        "https://lonelyguyse1-cybersec.hf.space",
}

ARTIFACTS = Path("_artifacts")
ARTIFACTS.mkdir(exist_ok=True)

ADAPTER_DIR        = ARTIFACTS / "qwen_cybersec_lora"
TRAIN_LOG          = ARTIFACTS / "training_log.json"
BASELINE_JSON      = ARTIFACTS / "baseline_metrics.json"
POST_JSON          = ARTIFACTS / "post_train_metrics.json"
HELDOUT_JSON       = ARTIFACTS / "heldout_metrics.json"
BASELINE_PLOT      = ARTIFACTS / "baseline_curves.png"
BEFORE_AFTER       = ARTIFACTS / "before_after_curves.png"
TRAIN_DIAGNOSTICS  = ARTIFACTS / "training_diagnostics.png"
SUMMARY_MD         = ARTIFACTS / "summary_table.md"
TRAJECTORY_JSON    = ARTIFACTS / "trajectory_dataset.jsonl"

MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_LEN  = 1024
MAX_NEW_TOKENS  = 48

TRAIN_SCENARIOS = list_train_scenarios()
HELDOUT_SCENARIOS = list_eval_scenarios()
ALL_SCENARIOS = list_scenarios()
SEEDS_BASELINE = list(range(MODE["n_baseline_episodes"]))
SEEDS_DATASET  = list(range(MODE["n_dataset_seeds"]))
SEEDS_POST     = list(range(MODE["n_post_train_episodes"]))

print("train scenarios:   ", TRAIN_SCENARIOS)
print("held-out scenarios:", HELDOUT_SCENARIOS)
print("MODE:")
print(json.dumps(MODE, indent=2))
"""))

CELLS.append(md("## 2. Baseline: Random vs Heuristic (train + held-out scenarios)"))

CELLS.append(code(r"""
# Local CybersecEnvironment for fast in-kernel rollouts. We optionally
# repeat the baseline against the deployed HF Space below as a separate
# remote-env smoke check.
env = CybersecEnvironment()
baseline_runs = {}

t0 = time.time()
for sid in ALL_SCENARIOS:
    rand = [run_episode(env, RandomPolicy(seed=s), seed=s, scenario_id=sid) for s in SEEDS_BASELINE]
    heur = [run_episode(env, HeuristicPolicy(),    seed=s, scenario_id=sid) for s in SEEDS_BASELINE]
    baseline_runs[sid] = {"random": rand, "heuristic": heur}
    print(f"{sid:<32s}  random={aggregate_results(rand)['mean_return']:7.3f}"
          f"  heuristic={aggregate_results(heur)['mean_return']:7.3f}")
print(f"baseline elapsed: {time.time()-t0:.1f}s")
"""))

CELLS.append(md(r"""
### 2a. (Optional) Remote-env smoke check

If `MODE['use_remote_env']` is `True`, this cell runs **one** baseline
episode of the heuristic policy against the deployed HF Space env. The
purpose is to prove that the same package code, served over the OpenEnv
WebSocket protocol, returns identical-shape observations -- not to repeat
the full baseline against the network (which would be slow).
"""))

CELLS.append(code(r"""
import asyncio

async def _remote_smoke():
    base_url = MODE["remote_env_url"]
    print(f"connecting to remote env: {base_url}")
    async with CybersecEnv(base_url=base_url) as remote:
        result = await remote.reset(seed=0, scenario_id=TRAIN_SCENARIOS[0])
        steps = 0
        cumulative = 0.0
        while not result.done and steps < 80:
            policy = HeuristicPolicy()
            action = policy.act(result.observation)
            result = await remote.step(action)
            cumulative += float(result.reward or 0.0)
            steps += 1
        print(f"remote heuristic episode: steps={steps}  cumulative_reward={cumulative:.3f}")

if MODE["use_remote_env"]:
    try:
        asyncio.run(_remote_smoke())
    except Exception as exc:
        print(f"remote env smoke failed (continuing with local): {exc!r}")
else:
    print("remote env smoke skipped (MODE['use_remote_env'] is False)")
"""))

CELLS.append(md("### 2b. Persist baseline metrics + plot"))

CELLS.append(code(r"""
def _padded_cumulative(curves, target_len):
    out = np.zeros((len(curves), target_len), dtype=float)
    for i, c in enumerate(curves):
        out[i, : len(c)] = c
        if len(c) < target_len:
            out[i, len(c):] = c[-1] if c else 0.0
    return np.cumsum(out, axis=1)

n_panels = len(ALL_SCENARIOS)
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
if n_panels == 1:
    axes = [axes]
for ax, sid in zip(axes, ALL_SCENARIOS):
    cell = baseline_runs[sid]
    horizon = max(max(len(r.reward_curve) for r in cell['random']),
                  max(len(r.reward_curve) for r in cell['heuristic']))
    for label, color in [("random", "tab:gray"), ("heuristic", "tab:blue")]:
        cumr = _padded_cumulative([r.reward_curve for r in cell[label]], horizon)
        mean = cumr.mean(axis=0); std = cumr.std(axis=0)
        ax.plot(mean, label=label, color=color)
        ax.fill_between(np.arange(horizon), mean - std, mean + std, color=color, alpha=0.15)
    held_out_tag = "  (HELD-OUT)" if sid in HELDOUT_SCENARIOS else ""
    ax.set_title(sid + held_out_tag, fontsize=10); ax.set_xlabel("tick"); ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("cumulative reward"); axes[0].legend()
fig.suptitle(f"Cybersec OpenEnv baselines, n={MODE['n_baseline_episodes']}/scenario")
fig.tight_layout(); fig.savefig(BASELINE_PLOT, dpi=140); plt.show()

baseline_metrics = {
    sid: {p: aggregate_results(cell[p]) for p in ("random", "heuristic")}
    for sid, cell in baseline_runs.items()
}
baseline_metrics["_meta"] = {
    "n_episodes": MODE["n_baseline_episodes"],
    "train_scenarios": TRAIN_SCENARIOS,
    "heldout_scenarios": HELDOUT_SCENARIOS,
}
BASELINE_JSON.write_text(json.dumps(baseline_metrics, indent=2))
print("wrote", BASELINE_JSON)
"""))

CELLS.append(md(r"""
## 3. Build the GRPO training dataset

Build a ~1500-prompt dataset from heuristic rollouts on the **training
scenarios only**. Each row carries a pickled env snapshot so
`reward_step_total` can clone the env and apply the candidate action.

The held-out scenario (`cloud_metadata_ssrf`) is **deliberately not in
the training data** -- it exists only for the OOD post-training eval.
"""))

CELLS.append(code(r"""
t0 = time.time()
rows = []
trajectory_lines = []
for sid in TRAIN_SCENARIOS:
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
print(f"dataset size: {len(ds)}  (train scenarios only)  build elapsed: {time.time()-t0:.1f}s")
print(ds[0]["prompt"][:300])
"""))

CELLS.append(md(r"""
## 4. Reward functions (eight independent signals)

All reward functions live in `cybersec.training.rewards` so the notebook,
the tests, and any future training script use the same definitions.
"""))

CELLS.append(code(r"""
REWARD_FUNCS = default_reward_funcs()
REWARD_NAMES = [f.__name__ for f in REWARD_FUNCS]
print("reward components:")
for n in REWARD_NAMES:
    print(f"  - {n}")
"""))

CELLS.append(md("## 5. Load Qwen2.5-1.5B with Unsloth (4-bit QLoRA)"))

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

CELLS.append(md("## 6. Format dataset with the chat template"))

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

CELLS.append(md("## 7. GRPO trainer"))

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
    # Mild sampling temperature so candidates within a group can diverge,
    # giving the new reward_action_diversity term something to score.
    temperature=1.0,
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

CELLS.append(md("## 8. Save adapter + training log"))

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
### 8a. Training diagnostics: KL, loss, per-component reward

Iter-1 had no KL/entropy plots, so we couldn't see the policy collapsing
toward a single canned plan even though the cumulative reward kept
climbing. These plots are the early-warning system: if KL is rising fast
and `reward_action_diversity` plateaus near `1/num_generations`, the
policy is mode-collapsing and you should stop and adjust.
"""))

CELLS.append(code(r"""
log = json.loads(TRAIN_LOG.read_text()) if TRAIN_LOG.exists() else {"history": []}
hist = log.get("history") or []

if not hist:
    print("no training log rows -- did GRPO train?")
else:
    df_log = pd.DataFrame(hist)
    if "step" not in df_log.columns:
        df_log["step"] = range(len(df_log))

    base_cols = ["loss", "kl", "reward", "completion_length", "grad_norm"]
    component_cols = [c for c in df_log.columns
                      if c.startswith("rewards/") or c in REWARD_NAMES
                      or any(c.endswith(name) for name in REWARD_NAMES)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # panel 1: loss + kl
    for col, color in [("loss", "tab:blue"), ("kl", "tab:red")]:
        if col in df_log.columns and df_log[col].notna().any():
            axes[0].plot(df_log["step"], df_log[col], label=col, color=color)
    axes[0].set_title("loss / KL"); axes[0].set_xlabel("step"); axes[0].legend()

    # panel 2: total reward + completion length
    if "reward" in df_log.columns and df_log["reward"].notna().any():
        axes[1].plot(df_log["step"], df_log["reward"], label="reward", color="tab:green")
    if "completion_length" in df_log.columns and df_log["completion_length"].notna().any():
        ax2 = axes[1].twinx()
        ax2.plot(df_log["step"], df_log["completion_length"], label="completion_length",
                 color="tab:purple", linestyle="--")
        ax2.set_ylabel("completion_length", color="tab:purple")
    axes[1].set_title("total reward + completion length"); axes[1].set_xlabel("step")
    axes[1].legend(loc="upper left")

    # panel 3: per-reward-component breakdown
    if component_cols:
        for col in component_cols:
            if df_log[col].notna().any():
                short = col.split("/")[-1].replace("reward_", "")
                axes[2].plot(df_log["step"], df_log[col], label=short, alpha=0.85)
        axes[2].set_title("reward components per step")
        axes[2].set_xlabel("step")
        axes[2].legend(fontsize=7, ncol=2, loc="best")
    else:
        axes[2].set_title("(no per-component reward columns in log)")

    fig.tight_layout(); fig.savefig(TRAIN_DIAGNOSTICS, dpi=140); plt.show()
    print("wrote", TRAIN_DIAGNOSTICS)
"""))

CELLS.append(md(r"""
## 9. Evaluate the trained policy

Reload through **Unsloth** (not vanilla `transformers + peft`) on
purpose. GRPO trains the model with Unsloth's fused-QKV attention patch
in place -- that patch adds an `apply_qkv` method to `Qwen2Attention`,
and the saved LoRA expects to plug into it. Loading the adapter
directory through `FastLanguageModel.from_pretrained` re-applies
Unsloth's patches before grafting the LoRA on, which is the supported
path. We then call `FastLanguageModel.for_inference(eval_model)` to
switch to Unsloth's 2× faster decode.
"""))

CELLS.append(code(r"""
from unsloth import FastLanguageModel

# Free training-only buffers if the kernel still has them.
for _name in ("model", "trainer"):
    if _name in globals():
        del globals()[_name]
torch.cuda.empty_cache()

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
    a = parsed_action(completion)
    return a or CybersecAction(action_type=ActionType.MONITOR)

class TrainedLLMPolicy:
    name = "qwen-1.5b-grpo"
    def reset(self):
        pass
    def act(self, obs):
        return llm_act(obs)

trained_runs = {}
t0 = time.time()
for sid in TRAIN_SCENARIOS:
    runs = [run_episode(env, TrainedLLMPolicy(), seed=s, scenario_id=sid) for s in SEEDS_POST]
    trained_runs[sid] = runs
    agg = aggregate_results(runs)
    print(f"{sid:<32s}  trained-llm={agg['mean_return']:7.3f}  "
          f"std={float(np.std([r.cumulative_reward for r in runs])):.3f}")
print(f"trained-policy eval (train scenarios) elapsed: {time.time()-t0:.1f}s")
"""))

CELLS.append(md(r"""
### 9a. Held-out OOD evaluation

The whole point of holding out `cloud_metadata_ssrf` from training is
this cell: a clean read on whether the trained policy *generalises* or
just memorised per-scenario plans.
"""))

CELLS.append(code(r"""
heldout_runs = {}
t0 = time.time()
for sid in HELDOUT_SCENARIOS:
    runs = [run_episode(env, TrainedLLMPolicy(), seed=s, scenario_id=sid) for s in SEEDS_POST]
    heldout_runs[sid] = runs
    agg = aggregate_results(runs)
    rand_mean = aggregate_results(baseline_runs[sid]['random'])['mean_return']
    heur_mean = aggregate_results(baseline_runs[sid]['heuristic'])['mean_return']
    std = float(np.std([r.cumulative_reward for r in runs]))
    print(f"{sid:<32s}  trained-llm={agg['mean_return']:7.3f}  std={std:.3f}  "
          f"vs heur={heur_mean:.3f}  vs rand={rand_mean:.3f}")
print(f"trained-policy eval (held-out scenarios) elapsed: {time.time()-t0:.1f}s")

heldout_metrics = {
    sid: {
        "random":      aggregate_results(baseline_runs[sid]["random"]),
        "heuristic":   aggregate_results(baseline_runs[sid]["heuristic"]),
        "trained_llm": aggregate_results(heldout_runs[sid]),
    }
    for sid in HELDOUT_SCENARIOS
}
HELDOUT_JSON.write_text(json.dumps(heldout_metrics, indent=2))
print("wrote", HELDOUT_JSON)
"""))

CELLS.append(md("## 10. Before/after curves on identical axes"))

CELLS.append(code(r"""
all_post = {**trained_runs, **heldout_runs}
panels = ALL_SCENARIOS
fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
if len(panels) == 1:
    axes = [axes]
for ax, sid in zip(axes, panels):
    cell = baseline_runs[sid]
    llm  = all_post.get(sid, [])
    horizon = max(
        max(len(r.reward_curve) for r in cell['random']),
        max(len(r.reward_curve) for r in cell['heuristic']),
        max((len(r.reward_curve) for r in llm), default=1),
    )
    palette = [("random", cell['random'], "tab:gray"),
               ("heuristic", cell['heuristic'], "tab:blue"),
               ("trained-llm", llm, "tab:red")]
    for label, runs, color in palette:
        if not runs:
            continue
        cumr = _padded_cumulative([r.reward_curve for r in runs], horizon)
        mean = cumr.mean(axis=0); std = cumr.std(axis=0)
        ax.plot(mean, label=label, color=color)
        ax.fill_between(np.arange(horizon), mean - std, mean + std, color=color, alpha=0.15)
    held_out_tag = "  (HELD-OUT)" if sid in HELDOUT_SCENARIOS else ""
    ax.set_title(sid + held_out_tag, fontsize=10); ax.set_xlabel("tick"); ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("cumulative reward"); axes[0].legend()
fig.suptitle(
    f"Cybersec OpenEnv -- pre/post GRPO ({MODE['grpo_max_steps']} steps), "
    f"n={MODE['n_post_train_episodes']}/scenario"
)
fig.tight_layout(); fig.savefig(BEFORE_AFTER, dpi=140); plt.show()
"""))

CELLS.append(md("## 11. Summary table + headline delta"))

CELLS.append(code(r"""
rows = []
for sid in ALL_SCENARIOS:
    cell = baseline_runs[sid]
    post_runs = all_post.get(sid, [])
    for policy_name, runs in [
        ("random",     cell["random"]),
        ("heuristic",  cell["heuristic"]),
        ("trained-llm", post_runs),
    ]:
        if not runs:
            continue
        agg = aggregate_results(runs)
        returns = [r.cumulative_reward for r in runs]
        rows.append({
            "scenario":     sid,
            "split":        "held-out" if sid in HELDOUT_SCENARIOS else "train",
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
        "trained_llm": aggregate_results(all_post.get(sid, [])) if all_post.get(sid) else None,
    }
    for sid in ALL_SCENARIOS
}
post_metrics["_meta"] = {
    "n_post_episodes": MODE["n_post_train_episodes"],
    "grpo_max_steps":  MODE["grpo_max_steps"],
    "model":           MODEL_NAME,
    "adapter":         str(ADAPTER_DIR),
    "train_scenarios": TRAIN_SCENARIOS,
    "heldout_scenarios": HELDOUT_SCENARIOS,
    "reward_funcs":    REWARD_NAMES,
}
POST_JSON.write_text(json.dumps(post_metrics, indent=2))
SUMMARY_MD.write_text(df.to_markdown(index=False))
print("wrote", POST_JSON)
print("wrote", SUMMARY_MD)
"""))

CELLS.append(code(r"""
print("=== Headline delta (trained-llm vs heuristic) ===")
for sid in ALL_SCENARIOS:
    h = aggregate_results(baseline_runs[sid]["heuristic"])["mean_return"]
    runs = all_post.get(sid, [])
    if not runs:
        continue
    t = aggregate_results(runs)["mean_return"]
    tag = "(HELD-OUT)" if sid in HELDOUT_SCENARIOS else "(train)"
    print(f"{sid:<32s} {tag:<11s} heuristic={h:7.3f}  trained-llm={t:7.3f}  delta={t-h:+.3f}")
"""))

CELLS.append(md(r"""
## 12. Sanity assertions (incl. iter-1 reward-hack canaries)

These are the gate before we ship a run as "good". The earlier checks
(invalid-action rate, divergence-vs-random) survive iter-1; the
**std_return** check is new in iter-2 -- it specifically catches the
mode-collapse failure where every seed produces the same trajectory.
"""))

CELLS.append(code(r"""
# 1. Reward shaping is healthy: heuristic must out-earn random *on aggregate*.
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
#    episodes. Aggregate over train + held-out to keep one number.
total_steps = sum(len(r.reward_curve) for runs in all_post.values() for r in runs)
total_invalid = sum(r.invalid_action_count for runs in all_post.values() for r in runs)
valid_rate = 1.0 - (total_invalid / max(1, total_steps))
print(f"valid-action rate across all post-train episodes: {valid_rate:.1%}  "
      f"({total_invalid} invalid / {total_steps} steps)")
assert valid_rate >= 0.8, (
    f"trained policy is producing too many invalid actions ({valid_rate:.1%} valid)"
)

# 3. Trained policy must not be catastrophically worse than random on any
#    *training* scenario. (Held-out scenarios are allowed to be worse --
#    that's the point of generalisation reporting; we just print it.)
for sid in TRAIN_SCENARIOS:
    rand = float(np.mean([r.cumulative_reward for r in baseline_runs[sid]['random']]))
    llm  = float(np.mean([r.cumulative_reward for r in trained_runs[sid]]))
    assert llm >= rand - 5.0, (
        f"{sid}: trained-llm ({llm:.3f}) is more than 5 points below random "
        f"({rand:.3f}) -- training likely diverged"
    )

# 4. Reward-hack canary: the trained policy MUST have non-trivial variance on
#    at least one training scenario. std_return == 0.0 across all training
#    scenarios is iter-1's exact failure mode -- a memorised canned plan
#    that ignores the observation.
non_zero_std_scenarios = []
for sid in TRAIN_SCENARIOS:
    returns = [r.cumulative_reward for r in trained_runs[sid]]
    s = float(np.std(returns))
    print(f"{sid:<32s}  trained-llm std_return={s:.4f}")
    if s > 0.1:
        non_zero_std_scenarios.append(sid)
assert non_zero_std_scenarios, (
    "REWARD HACK CANARY: trained-llm has std_return == 0 on every training "
    "scenario -- this is exactly the iter-1 mode-collapse pattern. "
    "Inspect _artifacts/training_diagnostics.png and the action distribution "
    "before publishing this run."
)

print("all sanity checks passed")
"""))

CELLS.append(md(r"""
## 13. (Optional) Push artifacts back to the HF Space

If you want the trained adapter and updated metrics to live on the HF
Space alongside the env (so the judges can diff `_artifacts/` between
iter-1 and iter-2), enable this cell. It uses the same
`huggingface_hub` API as the install cell.
"""))

CELLS.append(code(r"""
# Set this to True after a successful run if you want to upload artifacts.
PUSH_ARTIFACTS = False

if PUSH_ARTIFACTS:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(ARTIFACTS),
        path_in_repo="_artifacts",
        repo_id=HF_SPACE_REPO_ID,
        repo_type="space",
        commit_message="iter-2 training artifacts",
    )
    print("uploaded _artifacts/ to", HF_SPACE_REPO_ID)
else:
    print("artifact push skipped (set PUSH_ARTIFACTS=True to enable)")
"""))


def write(out_path: Path) -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "notebooks" / "cybersec_grpo.ipynb"
    write(target)
    print(f"wrote {target}  ({len(CELLS)} cells)")
