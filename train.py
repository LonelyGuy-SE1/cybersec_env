"""End-to-end GRPO training + eval superscript for the Cybersec OpenEnv.

This is the single canonical entry point. It mirrors
``notebooks/cybersec_grpo.ipynb`` cell-for-cell so reviewers can pick
either UX:

  * Notebook  -- click "Run all" in Colab / Kaggle.
  * This script -- ``python train.py`` from a fresh GPU box.

What it does, in order:

  0.  Detect runtime (Colab / Kaggle / local) and pick a writable workdir.
  1.  Probe the live HF Space at MODE["remote_env_url"] to make sure it's
      healthy and on the right action surface.
  2.  Run baseline RandomPolicy + HeuristicPolicy episodes against the
      live Space (or local env if --local) over all 4 scenarios.
  3.  Build a GRPO dataset locally (the reward functions clone a
      pickled env for each candidate completion; the network client
      isn't picklable, so this step has to be local).
  4.  Fine-tune Qwen2.5-1.5B-Instruct with Unsloth QLoRA + TRL GRPO
      using the 9 reward functions from ``cybersec.training.rewards``.
  5.  Reload the trained adapter through Unsloth, run it against the
      live Space on train + held-out scenarios.
  6.  Plot before/after curves + training diagnostics, persist metrics,
      and run the iter-4 sanity / reward-hack canaries.

All hyperparameters live in the ``MODE`` dict below; pass ``--mode local``
to develop offline against an in-process env.

The script is intentionally batteries-included: every artifact (curves,
JSON metrics, summary table, training log, trained adapter) lands in
``WORKDIR / "_artifacts"``. Re-run safe.

Usage
-----

    # Default (remote env, GPU)
    python train.py

    # Local env (offline dev), still trains GRPO
    python train.py --mode local

    # Only do the baseline pass and exit (great for smoke-testing the
    # live Space without burning GPU time)
    python train.py --baseline-only

    # Override any MODE value from the command line
    python train.py --grpo-max-steps 50 --n-baseline-episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 0. Logging hygiene + runtime detection
# ---------------------------------------------------------------------------


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)


def _detect_runtime() -> tuple[str, Path]:
    if "google.colab" in sys.modules:
        return "colab", Path("/content/cybersec_workdir")
    if Path("/kaggle/working").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle", Path("/kaggle/working/cybersec_workdir")
    return "local", Path.cwd() / "_artifacts"


RUNTIME, WORKDIR = _detect_runtime()
WORKDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Default MODE (everything is overridable from the CLI)
# ---------------------------------------------------------------------------


MODE: Dict[str, Any] = {
    "remote_env_url":         "https://lonelyguyse1-cybersec.hf.space",
    "n_baseline_episodes":    30,
    "n_dataset_seeds":        30,
    "max_dataset_rows":       1500,
    "grpo_max_steps":         120,
    # Iter-4: 6 candidates per prompt, temp=1.2, KL beta=0.04, lr=3e-6.
    # See cybersec/README.md "Iteration history" for why these defaults
    # changed from iter-3.
    "grpo_num_generations":   6,
    "grpo_per_device_bs":     2,
    "grpo_grad_accum":        4,
    "grpo_logging_steps":     5,
    "grpo_save_steps":        50,
    "grpo_temperature":       1.2,
    "grpo_beta":              0.04,
    "grpo_learning_rate":     3e-6,
    "n_post_train_episodes":  30,
    # "remote" hits the deployed HF Space (the way judges will). "local"
    # uses an in-process CybersecEnvironment.
    "eval_target":            "remote",
    "model_name":             "Qwen/Qwen2.5-1.5B-Instruct",
    "max_prompt_len":         1024,
    "max_new_tokens":         48,
}


# ---------------------------------------------------------------------------
# Optional: install dependencies on first run inside Colab/Kaggle. Local
# users are expected to have run `pip install -e cybersec/` already.
# ---------------------------------------------------------------------------


def _ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    if RUNTIME == "local":
        return  # local users manage their own env
    try:
        import cybersec  # noqa: F401
    except ImportError:
        # On Colab/Kaggle, pull the package straight from the HF Space
        # (the canonical source) so the script is fully self-contained.
        print("[deps] cybersec not installed; pulling from HF Space")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "huggingface_hub>=0.24", "openenv-core>=0.2.2"],
            check=True,
        )
        from huggingface_hub import snapshot_download
        stage = WORKDIR / "cybersec_env_pkg"
        stage.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Lonelyguyse1/cybersec",
            repo_type="space",
            revision="main",
            local_dir=str(stage),
            local_dir_use_symlinks=False,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(stage)],
            check=True,
        )

    # Training stack. Unsloth pulls torch + bitsandbytes; install it first
    # so its pinned wheels win.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"],
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "trl", "peft", "accelerate", "bitsandbytes"],
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "matplotlib", "pandas"],
        check=False,
    )


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------


def probe_space(base_url: str) -> dict:
    """GET /health and /schema on the live Space and assert the action surface."""

    out: dict = {}
    with urllib.request.urlopen(base_url + "/health", timeout=20) as r:
        out["health"] = json.loads(r.read())
    with urllib.request.urlopen(base_url + "/schema", timeout=20) as r:
        out["schema"] = json.loads(r.read())
    action_enum = (
        out["schema"].get("action", {}).get("$defs", {})
        .get("ActionType", {}).get("enum", [])
    )
    print(f"[probe] space health: {out['health']}")
    print(f"[probe] action enum: {action_enum}")
    assert "MONITOR" in action_enum and "ISOLATE_ASSET" in action_enum, (
        f"remote action surface doesn't match this script: {action_enum}"
    )
    return out


def _run_episode_remote(
    client_factory: Callable[[], Any],
    policy: Any,
    seed: int,
    scenario_id: str,
    EpisodeResult,
):
    """One episode over the OpenEnv WebSocket protocol."""

    if hasattr(policy, "reset"):
        policy.reset()

    invalid = 0
    fp = 0
    fallback = 0
    reward_curve: List[float] = []

    client = client_factory()
    with client:
        res = client.reset(seed=seed, scenario_id=scenario_id)
        obs = res.observation
        while not (res.done or obs.done):
            action = policy.act(obs)
            if getattr(policy, "last_act_was_fallback", False):
                fallback += 1
            res = client.step(action)
            obs = res.observation
            rb = obs.info.get("reward_breakdown", {}) if isinstance(obs.info, dict) else {}
            if rb.get("invalid_action_penalty", 0.0):
                invalid += 1
            if rb.get("false_positive_penalty", 0.0):
                fp += 1
            step_reward = res.reward if res.reward is not None else obs.reward
            reward_curve.append(float(0.0 if step_reward is None else step_reward))
        terminal = obs.info.get("terminal", {}) if isinstance(obs.info, dict) else {}

    return EpisodeResult(
        policy_name=policy.name,
        scenario_id=obs.scenario_id,
        attacker_personality=obs.attacker_personality.value
            if hasattr(obs.attacker_personality, "value")
            else str(obs.attacker_personality),
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


def make_eval_runner(eval_target: str, remote_env_url: str):
    """Return a `(policy, seed, scenario_id) -> EpisodeResult` function."""

    from cybersec import CybersecEnv, CybersecEnvironment
    from cybersec.baselines import EpisodeResult, run_episode

    if eval_target == "remote":
        def remote_factory():
            return CybersecEnv(base_url=remote_env_url).sync()
        return lambda policy, seed, scenario_id: _run_episode_remote(
            remote_factory, policy, seed, scenario_id, EpisodeResult
        )
    local_env = CybersecEnvironment()
    return lambda policy, seed, scenario_id: run_episode(
        local_env, policy, seed=seed, scenario_id=scenario_id
    )


def run_baselines(eval_runner, scenarios: List[str], n_episodes: int) -> dict:
    """RandomPolicy + HeuristicPolicy across every scenario × n seeds."""

    from cybersec.baselines import HeuristicPolicy, RandomPolicy, aggregate_results

    seeds = list(range(n_episodes))
    out: Dict[str, dict] = {}
    t0 = time.time()
    for sid in scenarios:
        rand = [eval_runner(RandomPolicy(seed=s), seed=s, scenario_id=sid) for s in seeds]
        heur = [eval_runner(HeuristicPolicy(),    seed=s, scenario_id=sid) for s in seeds]
        out[sid] = {"random": rand, "heuristic": heur}
        print(
            f"[baseline] {sid:<32s} "
            f"random={aggregate_results(rand)['mean_return']:7.3f}  "
            f"heuristic={aggregate_results(heur)['mean_return']:7.3f}  "
            f"({n_episodes} ep each)"
        )
    print(f"[baseline] elapsed {time.time()-t0:.1f}s")
    return out


def build_dataset(train_scenarios: List[str], n_seeds: int, max_rows: int):
    """Build the local GRPO dataset (snapshots can't go over WebSocket)."""

    from cybersec import CybersecEnvironment
    from cybersec.baselines import HeuristicPolicy
    from cybersec.training.rewards import (
        SYSTEM_PROMPT, render_observation, snapshot_env,
    )

    rows = []
    t0 = time.time()
    for sid in train_scenarios:
        for seed in range(n_seeds):
            ep_env = CybersecEnvironment()
            policy = HeuristicPolicy()
            obs = ep_env.reset(seed=seed, scenario_id=sid)
            while not obs.done:
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
                    "env_snapshot":      snapshot_env(ep_env),
                })
                obs = ep_env.step(policy.act(obs))
    random.Random(0).shuffle(rows)
    rows = rows[:max_rows]
    print(f"[dataset] {len(rows)} rows  built in {time.time()-t0:.1f}s")
    return rows


def train_grpo(rows: list, mode: dict, adapter_dir: Path, train_log_path: Path):
    """Fine-tune Qwen with GRPO + Unsloth QLoRA. Returns the saved log."""

    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from cybersec.training.rewards import default_reward_funcs

    reward_funcs = default_reward_funcs()
    reward_names = [f.__name__ for f in reward_funcs]
    print("[train] reward components:", reward_names)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mode["model_name"],
        max_seq_length=mode["max_prompt_len"] + mode["max_new_tokens"],
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

    def _to_chat_prompt(row):
        msgs = [
            {"role": "system", "content": row["system"]},
            {"role": "user",   "content": row["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        }

    ds_chat = Dataset.from_list(rows).map(_to_chat_prompt)

    config = GRPOConfig(
        output_dir=str(adapter_dir.parent / "grpo_checkpoints"),
        learning_rate=mode["grpo_learning_rate"],
        per_device_train_batch_size=mode["grpo_per_device_bs"],
        gradient_accumulation_steps=mode["grpo_grad_accum"],
        max_prompt_length=mode["max_prompt_len"],
        max_completion_length=mode["max_new_tokens"],
        num_generations=mode["grpo_num_generations"],
        num_train_epochs=1,
        max_steps=mode["grpo_max_steps"],
        logging_steps=mode["grpo_logging_steps"],
        save_steps=mode["grpo_save_steps"],
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=0,
        temperature=mode["grpo_temperature"],
        beta=mode["grpo_beta"],
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds_chat,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    t0 = time.time()
    trainer.train()
    print(f"[train] GRPO elapsed {time.time()-t0:.1f}s")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    history = getattr(trainer.state, "log_history", []) or []
    train_log_path.write_text(json.dumps({
        "reward_names": reward_names,
        "history": history,
        "model_name": mode["model_name"],
        "mode": mode,
    }, indent=2))
    print(f"[train] saved adapter -> {adapter_dir}")
    return reward_names


def plot_training_diagnostics(
    train_log_path: Path, out_png: Path, reward_names: List[str]
) -> None:
    """KL / loss / per-component reward plot, mirrors notebook section 9a."""

    if not train_log_path.exists():
        print(f"[diag] {train_log_path} missing -- skipping diagnostics plot")
        return

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = json.loads(train_log_path.read_text())
    hist = log.get("history") or []
    if not hist:
        print("[diag] empty training_log.history -- skipping diagnostics plot")
        return

    df = pd.DataFrame(hist)
    if "step" not in df.columns:
        df["step"] = range(len(df))

    component_cols = [
        c for c in df.columns
        if c.startswith("rewards/") or c in reward_names
        or any(c.endswith(name) for name in reward_names)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for col, color in [("loss", "tab:blue"), ("kl", "tab:red")]:
        if col in df.columns and df[col].notna().any():
            axes[0].plot(df["step"], df[col], label=col, color=color)
    axes[0].set_title("loss / KL"); axes[0].set_xlabel("step"); axes[0].legend()

    if "reward" in df.columns and df["reward"].notna().any():
        axes[1].plot(df["step"], df["reward"], label="reward", color="tab:green")
    if "completion_length" in df.columns and df["completion_length"].notna().any():
        ax2 = axes[1].twinx()
        ax2.plot(df["step"], df["completion_length"], label="completion_length",
                 color="tab:purple", linestyle="--")
        ax2.set_ylabel("completion_length", color="tab:purple")
    axes[1].set_title("total reward + completion length"); axes[1].set_xlabel("step")
    axes[1].legend(loc="upper left")

    if component_cols:
        for col in component_cols:
            if df[col].notna().any():
                short = col.split("/")[-1].replace("reward_", "")
                axes[2].plot(df["step"], df[col], label=short, alpha=0.85)
        axes[2].set_title("reward components per step")
        axes[2].set_xlabel("step")
        axes[2].legend(fontsize=7, ncol=2, loc="best")
    else:
        axes[2].set_title("(no per-component reward columns in log)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[diag] wrote {out_png}")


def evaluate_trained(
    adapter_dir: Path,
    eval_runner,
    train_scenarios: List[str],
    heldout_scenarios: List[str],
    mode: dict,
):
    """Reload through Unsloth and roll the trained policy out."""

    import torch
    import numpy as np
    from unsloth import FastLanguageModel

    from cybersec import ActionType, CybersecAction
    from cybersec.baselines import aggregate_results
    from cybersec.training.rewards import (
        SYSTEM_PROMPT, parsed_action, render_observation,
    )

    eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=mode["max_prompt_len"] + mode["max_new_tokens"],
        dtype=None,
        load_in_4bit=True,
    )
    eval_tokenizer.pad_token = eval_tokenizer.pad_token or eval_tokenizer.eos_token
    FastLanguageModel.for_inference(eval_model)

    @torch.inference_mode()
    def _llm_act(obs):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": render_observation(obs)},
        ]
        text = eval_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = eval_tokenizer(text, return_tensors="pt").to(eval_model.device)
        out = eval_model.generate(
            **inputs,
            max_new_tokens=mode["max_new_tokens"],
            do_sample=False,
            pad_token_id=eval_tokenizer.eos_token_id,
        )
        completion = eval_tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        a = parsed_action(completion)
        _llm_act.last_was_fallback = a is None
        return a or CybersecAction(action_type=ActionType.MONITOR)

    class _TrainedLLMPolicy:
        name = "qwen-1.5b-grpo"
        last_act_was_fallback = False

        def reset(self):
            self.last_act_was_fallback = False

        def act(self, obs):
            a = _llm_act(obs)
            self.last_act_was_fallback = bool(
                getattr(_llm_act, "last_was_fallback", False)
            )
            return a

    seeds = list(range(mode["n_post_train_episodes"]))
    trained_runs: Dict[str, list] = {}
    for sid in train_scenarios:
        runs = [
            eval_runner(_TrainedLLMPolicy(), seed=s, scenario_id=sid)
            for s in seeds
        ]
        trained_runs[sid] = runs
        agg = aggregate_results(runs)
        std = float(np.std([r.cumulative_reward for r in runs]))
        print(
            f"[eval-train]  {sid:<32s} mean_return={agg['mean_return']:7.3f}  "
            f"std={std:.3f}  monitor_fallback_rate={agg['monitor_fallback_rate']:.3f}"
        )

    heldout_runs: Dict[str, list] = {}
    for sid in heldout_scenarios:
        runs = [
            eval_runner(_TrainedLLMPolicy(), seed=s, scenario_id=sid)
            for s in seeds
        ]
        heldout_runs[sid] = runs
        agg = aggregate_results(runs)
        std = float(np.std([r.cumulative_reward for r in runs]))
        print(
            f"[eval-OOD]    {sid:<32s} mean_return={agg['mean_return']:7.3f}  "
            f"std={std:.3f}  monitor_fallback_rate={agg['monitor_fallback_rate']:.3f}"
        )

    return trained_runs, heldout_runs


def assert_canaries(
    baseline_runs: dict,
    trained_runs: dict,
    heldout_runs: dict,
    train_scenarios: List[str],
):
    """Iter-4 sanity checks; raises AssertionError on regression."""

    import numpy as np

    heur_total = sum(
        float(np.mean([r.cumulative_reward for r in cell["heuristic"]]))
        for cell in baseline_runs.values()
    )
    rand_total = sum(
        float(np.mean([r.cumulative_reward for r in cell["random"]]))
        for cell in baseline_runs.values()
    )
    print(f"[canary] baseline totals  heuristic={heur_total:+.3f}  random={rand_total:+.3f}")
    assert heur_total > rand_total, (
        f"reward shaping regressed: heuristic ({heur_total:.3f}) "
        f"<= random ({rand_total:.3f})"
    )

    all_post = {**trained_runs, **heldout_runs}
    total_steps = sum(len(r.reward_curve) for runs in all_post.values() for r in runs) or 1
    total_invalid = sum(r.invalid_action_count for runs in all_post.values() for r in runs)
    valid_rate = 1.0 - total_invalid / total_steps
    print(f"[canary] valid-action rate {valid_rate:.1%}  ({total_invalid}/{total_steps})")
    assert valid_rate >= 0.8, f"too many invalid actions ({valid_rate:.1%})"

    for sid in train_scenarios:
        rand = float(np.mean([r.cumulative_reward for r in baseline_runs[sid]["random"]]))
        llm = float(np.mean([r.cumulative_reward for r in trained_runs[sid]]))
        assert llm >= rand - 5.0, (
            f"{sid}: trained-llm ({llm:.3f}) >5pts below random ({rand:.3f})"
        )

    non_zero = []
    for sid in train_scenarios:
        s = float(np.std([r.cumulative_reward for r in trained_runs[sid]]))
        print(f"[canary] {sid:<32s} std_return={s:.4f}")
        if s > 0.1:
            non_zero.append(sid)
    required = max(2, len(train_scenarios) - 1)
    assert len(non_zero) >= required, (
        f"REWARD HACK CANARY: {len(train_scenarios) - len(non_zero)} of "
        f"{len(train_scenarios)} train scenarios have std_return < 0.1 "
        f"(non-zero: {non_zero}). This is iter-3-style mode collapse; "
        "raise grpo_num_generations / grpo_temperature / grpo_beta or "
        "down-weight reward_step_total before publishing."
    )

    fallback_rate = sum(
        r.monitor_fallback_count for runs in all_post.values() for r in runs
    ) / total_steps
    print(f"[canary] monitor_fallback_rate {fallback_rate:.1%}")
    assert fallback_rate <= 0.5, (
        f"trained-llm is falling back to MONITOR on {fallback_rate:.1%} "
        "of steps -- model is mostly emitting unparseable text"
    )

    print("[canary] all sanity checks passed")


def persist_artifacts(
    workdir: Path,
    baseline_runs: dict,
    trained_runs: dict,
    heldout_runs: dict,
    train_scenarios: List[str],
    heldout_scenarios: List[str],
    mode: dict,
):
    """Persist before/after curves + JSON metrics + summary table."""

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cybersec.baselines import aggregate_results

    all_scenarios = list(baseline_runs.keys())
    all_post = {**trained_runs, **heldout_runs}

    def _padded_cumulative(curves, target_len):
        out = np.zeros((len(curves), target_len), dtype=float)
        for i, c in enumerate(curves):
            out[i, : len(c)] = c
            if len(c) < target_len:
                out[i, len(c):] = c[-1] if c else 0.0
        return np.cumsum(out, axis=1)

    fig, axes = plt.subplots(1, len(all_scenarios),
                             figsize=(5 * len(all_scenarios), 4), sharey=True)
    if len(all_scenarios) == 1:
        axes = [axes]
    for ax, sid in zip(axes, all_scenarios):
        cell = baseline_runs[sid]
        llm = all_post.get(sid, [])
        horizon = max(
            max(len(r.reward_curve) for r in cell["random"]),
            max(len(r.reward_curve) for r in cell["heuristic"]),
            max((len(r.reward_curve) for r in llm), default=1),
        )
        for label, runs, color in [
            ("random",    cell["random"],    "tab:gray"),
            ("heuristic", cell["heuristic"], "tab:blue"),
            ("trained-llm", llm,            "tab:red"),
        ]:
            if not runs:
                continue
            cumr = _padded_cumulative([r.reward_curve for r in runs], horizon)
            mean = cumr.mean(axis=0); std = cumr.std(axis=0)
            ax.plot(mean, label=label, color=color)
            ax.fill_between(np.arange(horizon),
                            mean - std, mean + std, color=color, alpha=0.15)
        held = "  (HELD-OUT)" if sid in heldout_scenarios else ""
        ax.set_title(sid + held, fontsize=10)
        ax.set_xlabel("tick"); ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("cumulative reward"); axes[0].legend()
    fig.suptitle(
        f"Cybersec OpenEnv -- pre/post GRPO ({mode['grpo_max_steps']} steps), "
        f"n={mode['n_post_train_episodes']}/scenario, env={mode['eval_target']}"
    )
    fig.tight_layout()
    fig.savefig(workdir / "before_after_curves.png", dpi=140)
    plt.close(fig)

    rows = []
    for sid in all_scenarios:
        cell = baseline_runs[sid]
        post_runs = all_post.get(sid, [])
        for policy_name, runs in [
            ("random",      cell["random"]),
            ("heuristic",   cell["heuristic"]),
            ("trained-llm", post_runs),
        ]:
            if not runs:
                continue
            agg = aggregate_results(runs)
            returns = [r.cumulative_reward for r in runs]
            rows.append({
                "scenario":               sid,
                "split":                  "held-out" if sid in heldout_scenarios else "train",
                "policy":                 policy_name,
                "mean_return":            agg["mean_return"],
                "std_return":             round(float(np.std(returns)), 3) if returns else 0.0,
                "mean_stages":            agg["mean_stages_succeeded"],
                "exfil_rate":             agg["exfil_rate"],
                "invalid_rate":           agg["mean_invalid_actions"],
                "monitor_fallback_rate":  agg.get("monitor_fallback_rate", 0.0),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    (workdir / "summary_table.md").write_text(df.to_markdown(index=False))
    (workdir / "post_train_metrics.json").write_text(json.dumps({
        sid: {
            "random":     aggregate_results(baseline_runs[sid]["random"]),
            "heuristic":  aggregate_results(baseline_runs[sid]["heuristic"]),
            "trained_llm": aggregate_results(all_post.get(sid, []))
                if all_post.get(sid) else None,
        }
        for sid in all_scenarios
    } | {"_meta": {
        "n_post_episodes": mode["n_post_train_episodes"],
        "grpo_max_steps":  mode["grpo_max_steps"],
        "eval_target":     mode["eval_target"],
        "model":           mode["model_name"],
        "train_scenarios": train_scenarios,
        "heldout_scenarios": heldout_scenarios,
    }}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["remote", "local"], default=None,
                    help="Override MODE['eval_target'].")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Run only the baseline pass and exit (no GRPO).")
    ap.add_argument("--skip-install", action="store_true",
                    help="Skip pip install bootstrap on Colab/Kaggle.")
    ap.add_argument("--remote-env-url", default=None)
    ap.add_argument("--n-baseline-episodes", type=int, default=None)
    ap.add_argument("--n-post-train-episodes", type=int, default=None)
    ap.add_argument("--grpo-max-steps", type=int, default=None)
    ap.add_argument("--grpo-num-generations", type=int, default=None)
    ap.add_argument("--grpo-temperature", type=float, default=None)
    ap.add_argument("--grpo-beta", type=float, default=None)
    ap.add_argument("--grpo-learning-rate", type=float, default=None)
    return ap.parse_args()


def _apply_overrides(mode: dict, args: argparse.Namespace) -> dict:
    if args.mode is not None:
        mode["eval_target"] = args.mode
    for key in (
        "remote_env_url",
        "n_baseline_episodes",
        "n_post_train_episodes",
        "grpo_max_steps",
        "grpo_num_generations",
        "grpo_temperature",
        "grpo_beta",
        "grpo_learning_rate",
    ):
        v = getattr(args, key.replace("-", "_"), None)
        if v is not None:
            mode[key] = v
    return mode


def main() -> int:
    args = _parse_args()
    _ensure_deps(args.skip_install)

    from cybersec import (
        list_eval_scenarios, list_scenarios, list_train_scenarios,
    )

    mode = _apply_overrides(MODE.copy(), args)
    print(f"[runtime] {RUNTIME}  workdir={WORKDIR}")
    print(f"[mode] {json.dumps(mode, indent=2, default=str)}")

    if mode["eval_target"] == "remote":
        probe_space(mode["remote_env_url"])

    train_scenarios = list_train_scenarios()
    heldout_scenarios = list_eval_scenarios()
    all_scenarios = list_scenarios()

    eval_runner = make_eval_runner(mode["eval_target"], mode["remote_env_url"])

    baseline_runs = run_baselines(
        eval_runner, all_scenarios, mode["n_baseline_episodes"]
    )
    if args.baseline_only:
        print("[main] --baseline-only set; exiting.")
        return 0

    rows = build_dataset(
        train_scenarios,
        n_seeds=mode["n_dataset_seeds"],
        max_rows=mode["max_dataset_rows"],
    )

    adapter_dir = WORKDIR / "qwen_cybersec_lora"
    train_log = WORKDIR / "training_log.json"
    reward_names = train_grpo(rows, mode, adapter_dir, train_log)

    plot_training_diagnostics(
        train_log, WORKDIR / "training_diagnostics.png", reward_names
    )

    trained_runs, heldout_runs = evaluate_trained(
        adapter_dir, eval_runner, train_scenarios, heldout_scenarios, mode
    )

    persist_artifacts(
        WORKDIR, baseline_runs, trained_runs, heldout_runs,
        train_scenarios, heldout_scenarios, mode,
    )
    assert_canaries(baseline_runs, trained_runs, heldout_runs, train_scenarios)
    print(f"[main] artifacts in {WORKDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
