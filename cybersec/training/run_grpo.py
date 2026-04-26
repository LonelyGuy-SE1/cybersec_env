"""Headless GRPO outer-loop training (same algorithm as ``notebooks/cybersec_grpo.ipynb`` §5).

Requires a CUDA machine with Unsloth, TRL, PyTorch, and datasets installed separately
(see repository README). Import this module only in training environments.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from ..baselines import HeuristicPolicy
from ..models import ActionType, CybersecAction
from ..scenarios import list_train_scenarios
from .rewards import (
    SYSTEM_PROMPT,
    collect_grpo_rows_from_rollouts,
    default_reward_funcs,
    parsed_action,
    render_observation,
)


def default_grpo_mode(*, fast: bool = False) -> dict[str, Any]:
    """Hyperparameters shared with the GRPO notebook (``MODE`` dict)."""
    if fast:
        return {
            "strict_canary": False,
            "n_dataset_seeds": 2,
            "max_dataset_rows": 128,
            "on_policy_outer_loops": 2,
            "warmup_heuristic_outer0": True,
            "grpo_steps_per_outer_loop": 60,
            "grpo_num_train_epochs": 1,
            "grpo_num_generations": 8,
            "grpo_per_device_bs": 1,
            "grpo_grad_accum": 6,
            "grpo_logging_steps": 5,
            "grpo_save_steps": 50,
            "grpo_temperature": 1.4,
            "grpo_beta": 0.04,
            "grpo_learning_rate": 3e-5,
            "rollout_do_sample": True,
            "rollout_temperature": 1.2,
            "rollout_top_p": 0.92,
        }
    return {
        "strict_canary": True,
        "n_dataset_seeds": 30,
        "max_dataset_rows": 1500,
        "on_policy_outer_loops": 2,
        "warmup_heuristic_outer0": True,
        "grpo_steps_per_outer_loop": 60,
        "grpo_num_train_epochs": 1,
        "grpo_num_generations": 8,
        "grpo_per_device_bs": 1,
        "grpo_grad_accum": 6,
        "grpo_logging_steps": 5,
        "grpo_save_steps": 50,
        "grpo_temperature": 1.4,
        "grpo_beta": 0.04,
        "grpo_learning_rate": 3e-5,
        "rollout_do_sample": True,
        "rollout_temperature": 1.2,
        "rollout_top_p": 0.92,
    }


def _drop_gen_max_length(lm: Any) -> None:
    gen_cfg = getattr(lm, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "max_length", None) is not None:
        gen_cfg.max_length = None


class RolloutLLMPolicy:
    """On-policy rollout policy using current model weights."""

    name = "rollout-llm"

    def __init__(
        self,
        lm: Any,
        tok: Any,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> None:
        self._lm = lm
        self._tok = tok
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_p = top_p

    def reset(self) -> None:
        pass

    def act(self, obs: Any) -> CybersecAction:
        import torch

        self._lm.eval()
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(obs)},
        ]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._lm.device)
        gen_kw: dict[str, Any] = dict(
            max_new_tokens=self._max_new_tokens,
            pad_token_id=self._tok.eos_token_id,
        )
        if self._do_sample:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = self._temperature
            gen_kw["top_p"] = self._top_p
        else:
            gen_kw["do_sample"] = False
        with torch.inference_mode():
            out = self._lm.generate(**inputs, **gen_kw)
        completion = self._tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        a = parsed_action(completion)
        return a or CybersecAction(action_type=ActionType.INVESTIGATE, target="INVALID_TARGET_SYNTAX")


def train_grpo(
    *,
    artifacts_dir: Path,
    model_name: str,
    mode: dict[str, Any] | None = None,
    max_prompt_len: int = 2048,
    max_new_tokens: int = 64,
) -> Path:
    """Run outer-loop data collection + ``GRPOTrainer``; save LoRA adapter and ``training_log.json``.

    Returns path to the adapter directory.
    """
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    mode = dict(mode or default_grpo_mode(fast=False))
    mode["grpo_max_steps_effective"] = int(mode["grpo_steps_per_outer_loop"]) * int(
        mode["on_policy_outer_loops"]
    )

    artifacts_dir = Path(artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = artifacts_dir / "qwen_cybersec_lora"
    train_log = artifacts_dir / "training_log.json"
    manifest_path = artifacts_dir / "run_manifest.json"

    if not manifest_path.exists():
        eff_bs = int(mode["grpo_per_device_bs"]) * int(mode["grpo_grad_accum"])
        init_manifest = {
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "model_name": model_name,
            "max_prompt_len": max_prompt_len,
            "max_new_tokens": max_new_tokens,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "effective_optim_batch": eff_bs,
            "bf16": False,
            "fp16": bool(torch.cuda.is_available()),
            "entrypoint": "cybersec.training.run_grpo.train_grpo",
        }
        manifest_path.write_text(json.dumps(init_manifest, indent=2, default=str), encoding="utf-8")

    max_seq_len = max_prompt_len + max_new_tokens + 256
    train_scenarios = list_train_scenarios()
    n_seeds = int(mode["n_dataset_seeds"])
    seeds_dataset = list(range(n_seeds))
    reward_funcs = default_reward_funcs()
    reward_names = [f.__name__ for f in reward_funcs]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    _drop_gen_max_length(model)

    def to_chat_prompt(row: dict[str, Any]) -> dict[str, str]:
        msgs = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": text}

    use_fp16 = bool(torch.cuda.is_available())
    combined_history: list[dict[str, Any]] = []
    global_step_offset = 0
    grpo_total_sec = 0.0
    outer_n = int(mode["on_policy_outer_loops"])
    last_rows = 0

    for outer_i in range(outer_n):
        use_h = outer_i == 0 and bool(mode.get("warmup_heuristic_outer0", True))
        tag = "heuristic_warmup" if use_h else "llm_on_policy"
        print(f"=== outer {outer_i + 1}/{outer_n}  collect: {tag} ===")
        t0 = time.time()
        if use_h:
            pol: Any = HeuristicPolicy()
        else:
            pol = RolloutLLMPolicy(
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=bool(mode.get("rollout_do_sample", True)),
                temperature=float(mode.get("rollout_temperature", 0.9)),
                top_p=float(mode.get("rollout_top_p", 0.92)),
            )
        rows = collect_grpo_rows_from_rollouts(
            train_scenarios,
            seeds_dataset,
            pol,
            max_rows=int(mode["max_dataset_rows"]),
            shuffle_seed=outer_i + 7,
        )
        last_rows = len(rows)
        print(f"  collected {last_rows} rows in {time.time() - t0:.1f}s")
        ds = Dataset.from_list(rows)
        ds_chat = ds.map(to_chat_prompt)

        cfg = GRPOConfig(
            output_dir=str(artifacts_dir / f"grpo_checkpoints_outer{outer_i}"),
            learning_rate=float(mode["grpo_learning_rate"]),
            per_device_train_batch_size=int(mode["grpo_per_device_bs"]),
            gradient_accumulation_steps=int(mode["grpo_grad_accum"]),
            max_prompt_length=max_prompt_len,
            max_completion_length=max_new_tokens,
            num_generations=int(mode["grpo_num_generations"]),
            num_train_epochs=int(mode.get("grpo_num_train_epochs", 1)),
            max_steps=int(mode["grpo_steps_per_outer_loop"]),
            logging_steps=int(mode["grpo_logging_steps"]),
            save_steps=int(mode["grpo_save_steps"]),
            save_total_limit=1,
            bf16=False,
            fp16=use_fp16,
            report_to=[],
            seed=outer_i,
            temperature=float(mode["grpo_temperature"]),
            beta=float(mode["grpo_beta"]),
        )
        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=ds_chat,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
        )
        t1 = time.time()
        trainer.train()
        loop_sec = time.time() - t1
        grpo_total_sec += loop_sec
        raw_hist = getattr(trainer.state, "log_history", []) or []
        for row in raw_hist:
            r = dict(row)
            s = r.get("step")
            if s is not None:
                r["global_step"] = global_step_offset + int(s)
            r["outer_loop"] = outer_i
            combined_history.append(r)
        global_step_offset += int(getattr(trainer.state, "global_step", 0) or 0)
        print(f"  GRPO outer {outer_i} elapsed: {loop_sec:.1f}s")
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"=== all outers GRPO time: {grpo_total_sec:.1f}s ===")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    train_log.write_text(
        json.dumps(
            {
                "reward_names": reward_names,
                "history": combined_history,
                "model_name": model_name,
                "mode": mode,
            },
            indent=2,
        )
    )
    mf: dict[str, Any] = {}
    if manifest_path.exists():
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
    mf.update(
        {
            "grpo_train_elapsed_s": round(grpo_total_sec, 2),
            "on_policy_outer_loops": outer_n,
            "training_phase_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_rows_last_outer": last_rows,
            "training_log_path": str(train_log),
            "training_log_rows": len(combined_history),
            "model_name": model_name,
            "mode": mode,
        }
    )
    manifest_path.write_text(json.dumps(mf, indent=2, default=str), encoding="utf-8")
    print("saved adapter to", adapter_dir)
    print("wrote", train_log)
    return adapter_dir
