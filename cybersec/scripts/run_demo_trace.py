"""Generate a deterministic episode trace for quick qualitative review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines import HeuristicPolicy, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one deterministic trace")
    parser.add_argument(
        "--scenario",
        type=str,
        default="supply_chain_token_drift",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evals/demo_trace.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy = HeuristicPolicy()
    result = run_episode(
        policy=policy,
        scenario_id=args.scenario,
        seed=args.seed,
        horizon=args.horizon,
    )

    payload = {
        "episode": result.to_flat_dict(),
        "steps": result.step_records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Trace written to {output_path}")


if __name__ == "__main__":
    main()
