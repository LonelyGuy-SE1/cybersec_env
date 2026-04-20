# Evaluation Protocol

## Deterministic Benchmarking

Use fixed seeds and scenario ids to compare policies:

1. Random baseline
2. Rule-based heuristic baseline
3. Trained policy

Collect:

- average episode reward
- terminal `grader_score`
- exfiltration rate
- mean time-to-first-detection
- false-positive actions per episode

## Recommended Reporting Structure

1. Describe scenario and hidden attacker campaign.
2. Compare baseline and trained policy trajectories on fixed seeds.
3. Report terminal scores, exfiltration rates, and containment efficiency.
4. Present learning curves and error analysis.

## Reproducibility

- Pin scenario id and seed in each reported run.
- Report environment config overrides (`horizon`, false positive rate, etc.).
- Keep run logs with observation `info` payloads for forensic traceability.

## Built-in Baseline Runner

Use the included scripts for deterministic baseline evaluation:

```bash
python scripts/evaluate_baselines.py
python scripts/run_demo_trace.py --scenario supply_chain_token_drift --seed 101
```

Outputs:

- `outputs/evals/baseline_summary.json`
- `outputs/evals/baseline_episodes.csv`
- `outputs/evals/baseline_report.md`
- `outputs/evals/demo_trace.json`
- `outputs/evals/trajectory_dataset.jsonl` (via `training/export_dataset.py`)

The current baseline set includes:

- random policy (`RandomPolicy`)
- rule-based heuristic policy (`HeuristicPolicy`)
