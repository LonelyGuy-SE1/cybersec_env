"""Export deterministic trajectory datasets for downstream policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines import HeuristicPolicy, RandomPolicy, run_episode
from server.scenario_loader import list_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training trajectories")
    parser.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["heuristic", "random"],
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(list_scenarios()),
        help="Comma-separated scenario ids",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303,404",
        help="Comma-separated integer seeds",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evals/trajectory_dataset.jsonl",
    )
    return parser.parse_args()


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _build_record(*, episode: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode": episode,
        "step": step,
        "target": {
            "action_type": step.get("action_type"),
            "target": step.get("target"),
            "parameter": step.get("parameter"),
            "urgency": step.get("urgency"),
        },
        "reward": step.get("reward"),
        "done": step.get("done"),
    }


def main() -> None:
    args = parse_args()
    scenarios = _parse_str_list(args.scenarios)
    seeds = _parse_int_list(args.seeds)
    if not scenarios:
        raise ValueError("No scenarios provided")
    if not seeds:
        raise ValueError("No seeds provided")

    if args.policy == "heuristic":
        policy = HeuristicPolicy()
    else:
        policy = RandomPolicy(seed=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for scenario_id in scenarios:
            for seed in seeds:
                result = run_episode(
                    policy=policy,
                    scenario_id=scenario_id,
                    seed=seed,
                    horizon=args.horizon,
                )
                episode_meta = result.to_flat_dict()
                for step in result.step_records:
                    record = _build_record(episode=episode_meta, step=step)
                    handle.write(json.dumps(record) + "\n")
                    line_count += 1

    print(f"Dataset export complete: {output_path}")
    print(f"Policy: {args.policy}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Rows: {line_count}")


if __name__ == "__main__":
    main()
