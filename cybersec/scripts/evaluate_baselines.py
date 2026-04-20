"""Run deterministic baseline evaluations for CybersecEnv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines import HeuristicPolicy, RandomPolicy, aggregate_results, run_episode
from server.scenario_loader import list_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(list_scenarios()),
        help="Comma-separated scenario ids",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303",
        help="Comma-separated integer seeds",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon override for all runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evals",
        help="Output directory for JSON/CSV/Markdown reports",
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    values: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_episode_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    *,
    scenarios: List[str],
    seeds: List[int],
    summary_by_policy: Dict[str, Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Baseline Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Scenarios: {', '.join(scenarios)}")
    lines.append(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    lines.append("")
    lines.append(
        "| Policy | Mean Reward | Mean Grader | Success Rate | Exfiltration Rate | Mean First Detection |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy_name, summary in summary_by_policy.items():
        detection = summary["mean_first_detection_step"]
        detection_str = "n/a" if detection is None else f"{detection:.2f}"
        lines.append(
            "| "
            f"{policy_name} | "
            f"{summary['mean_episode_reward']:.4f} | "
            f"{summary['mean_grader_score']:.4f} | "
            f"{summary['success_rate']:.4f} | "
            f"{summary['exfiltration_rate']:.4f} | "
            f"{detection_str} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    scenarios = _parse_str_list(args.scenarios)
    seeds = _parse_int_list(args.seeds)
    if not scenarios:
        raise ValueError("No scenarios supplied")
    if not seeds:
        raise ValueError("No seeds supplied")

    policies = [RandomPolicy(seed=0), HeuristicPolicy()]

    all_results = []
    results_by_policy: Dict[str, List[Any]] = {policy.name: [] for policy in policies}

    for scenario_id in scenarios:
        for seed in seeds:
            for policy in policies:
                result = run_episode(
                    policy=policy,
                    scenario_id=scenario_id,
                    seed=seed,
                    horizon=args.horizon,
                )
                all_results.append(result)
                results_by_policy[policy.name].append(result)

    summary_by_policy = {
        policy_name: aggregate_results(results)
        for policy_name, results in results_by_policy.items()
    }

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    json_payload = {
        "scenarios": scenarios,
        "seeds": seeds,
        "horizon_override": args.horizon,
        "summary_by_policy": summary_by_policy,
        "episodes": [result.to_flat_dict() for result in all_results],
    }
    _write_json(output_dir / "baseline_summary.json", json_payload)

    _write_episode_csv(
        output_dir / "baseline_episodes.csv",
        [result.to_flat_dict() for result in all_results],
    )

    _write_markdown(
        output_dir / "baseline_report.md",
        scenarios=scenarios,
        seeds=seeds,
        summary_by_policy=summary_by_policy,
    )

    print("Baseline evaluation complete")
    print(f"Report: {output_dir / 'baseline_report.md'}")
    print(f"JSON:   {output_dir / 'baseline_summary.json'}")
    print(f"CSV:    {output_dir / 'baseline_episodes.csv'}")


if __name__ == "__main__":
    main()
