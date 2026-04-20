# Training Pipeline Scaffold

This directory provides a production-oriented scaffold for policy training and benchmarking.

## Current Scope

- deterministic baseline evaluation (`scripts/evaluate_baselines.py`)
- deterministic single-trace export (`scripts/run_demo_trace.py`)
- deterministic trajectory dataset export (`training/export_dataset.py`)
- deterministic inference runs (`inference.py`)
- benchmark artifacts under `outputs/evals/`

## Planned Expansion

The next iteration will add:

1. trajectory export adapters for TRL and Unsloth
2. reproducible train/validation split orchestration
3. checkpoint and run metadata management
4. automatic comparison reports between baseline and trained models

## Minimal Workflow

```bash
python scripts/evaluate_baselines.py
python scripts/run_demo_trace.py --scenario supply_chain_token_drift --seed 101
python training/export_dataset.py --policy heuristic --scenarios supply_chain_token_drift,federated_identity_takeover --seeds 101,202,303
```

Outputs:

- `outputs/evals/baseline_summary.json`
- `outputs/evals/baseline_episodes.csv`
- `outputs/evals/baseline_report.md`
- `outputs/evals/demo_trace.json`
- `outputs/evals/trajectory_dataset.jsonl`
