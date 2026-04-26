#!/usr/bin/env python3
"""Train the cybersec defender with GRPO (Unsloth QLoRA + TRL).

Prerequisites (GPU):

  pip install -e ./cybersec
  pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl peft accelerate bitsandbytes datasets

Run from the repository root:

  python scripts/train_cybersec_grpo.py --output-dir ./_artifacts
  python scripts/train_cybersec_grpo.py --fast --output-dir ./_artifacts_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="Cybersec GRPO training (see cybersec.training.run_grpo).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "_artifacts",
        help="Directory for adapter, training_log.json, run_manifest.json, checkpoints.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model id on the Hub.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Small budgets for a quick GPU sanity check (not for submission quality).",
    )
    args = parser.parse_args()

    from cybersec.training.run_grpo import default_grpo_mode, train_grpo

    mode = default_grpo_mode(fast=args.fast)
    train_grpo(artifacts_dir=args.output_dir, model_name=args.model_name, mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
