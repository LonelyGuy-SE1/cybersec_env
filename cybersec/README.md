---
title: Cybersec Environment Server
emoji: "🛡️"
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - cybersecurity
  - long-horizon
  - multi-agent
---

# CybersecEnv

CybersecEnv is a long-horizon, partially observable enterprise incident-response environment for OpenEnv.
It is designed for reinforcement learning agents that must make cost-aware decisions under uncertainty while
containing a staged adversarial campaign before exfiltration.

The environment is intentionally structured for rigorous research and operational evaluation:

- **Innovation**: hidden attacker policy, workflow-gated operations, delayed effects, noisy telemetry.
- **Storytelling**: replayable campaign timelines over cloud, code, identity, and network systems.
- **Measurable Improvement**: deterministic seeded scenarios and benchmark-friendly grading channels.
- **Pipeline readiness**: OpenEnv-native API contract (`reset`, `step`, `state`) and deployable Docker stack.

## Current Status

This repository now contains:

- Multi-agent world simulation (defender + adaptive attacker + enterprise workflow actor).
- Long-horizon campaign progression with deterministic scenario definitions.
- Typed, strict action/observation models aligned with OpenEnv schemas.
- Reward decomposition and terminal grading signals for training and evaluation.

## Environment Summary

### Agent and world roles

- **Controllable RL agent**: SOC Commander (defender).
- **Simulated adversary**: adaptive attacker policy with campaign path switching.
- **Simulated enterprise workflow actor**: ticket review and delayed execution windows.

### Partial observability

- Agent sees noisy alerts, ticket states, and asynchronous forensics updates.
- Ground-truth compromise state and full attacker internals remain hidden.

### Long-horizon dynamics

- Episodes run for a fixed horizon (default from scenario, typically 30+ steps).
- Attacker progresses through multi-stage campaign chains.
- Defender actions can have delayed impact via workflow and forensics mechanisms.

## Action Contract

`CybersecAction` uses a strict typed contract:

- `action_type`: one of
  - `MONITOR`
  - `QUERY_LOGS`
  - `TRIAGE_ALERT`
  - `REQUEST_FORENSICS`
  - `OPEN_TICKET`
  - `EXECUTE_TICKET`
  - `ISOLATE_ASSET`
  - `REVOKE_IDENTITY`
  - `ROTATE_SECRET`
  - `BLOCK_EGRESS`
  - `PATCH_ASSET`
- `target`: required for target-bound actions.
- `parameter`:
  - required as log source for `QUERY_LOGS`
  - required as requested control action for `OPEN_TICKET`
- `urgency`: `low|normal|high` (used for ticket workflow prioritization).

Invalid action contracts are rejected by model validators before environment logic runs.

## Observation Contract

`CybersecObservation` includes:

- Scenario metadata (`scenario_id`, `scenario_title`, `scenario_objective`).
- Time info (`tick`, `horizon`).
- Risk signal (`enterprise_risk_score`).
- Alert stream (`alerts`) and ticket queue (`open_tickets`).
- Async updates (`ticket_updates`, `forensics_updates`).
- Defender-known compromise sets and activity narrative.
- Action affordances (`available_actions`, `valid_targets`).
- `info` channel with machine-readable diagnostics:
  - reward breakdown
  - attacker event summary
  - workflow and forensics updates
  - terminal grading fields when episode ends

## Reward and Grading

Step reward uses weighted channels:

- Positive: detection gain, containment gain, terminal outcome quality.
- Negative: adversary progress, disruption cost, governance bypass cost,
  efficiency cost, false positives, invalid actions.

Terminal grading (in `info`) includes:

- `grader_score` in `[0,1]`
- `grader_success` boolean
- `terminal_reason` (`horizon_reached` or `adversary_exfiltration`)

## Scenario Suite

Deterministic scenario catalog (`server/scenario_loader.py`):

- `supply_chain_token_drift`
- `federated_identity_takeover`
- `insider_repo_pivot`

Each scenario defines:

- enterprise assets and identities
- at least two attack paths
- path steps with prerequisites, detection strengths, and progression weights
- default horizon and telemetry noise characteristics

## Quick Start

### 1) Install

```bash
uv sync
```

### 2) Run locally

```bash
uv run server --port 8000
```

### 3) Minimal client interaction

```python
from cybersec import CybersecAction, CybersecEnv

with CybersecEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.scenario_id)

    result = env.step(CybersecAction(action_type="QUERY_LOGS", parameter="cloud"))
    print(result.observation.enterprise_risk_score, result.reward)

    if result.observation.alerts:
        alert_id = result.observation.alerts[0].alert_id
        result = env.step(CybersecAction(action_type="TRIAGE_ALERT", target=alert_id))
        print(result.observation.info.get("reward_breakdown"))
```

### 4) Inference run

```bash
python inference.py --env local --mode heuristic --scenarios supply_chain_token_drift --seeds 101 --max-steps 32
```

## Workflow Mechanics

High-impact controls can be applied either:

- directly (faster, but higher governance penalty), or
- through enterprise tickets (`OPEN_TICKET` + `EXECUTE_TICKET`) with review/approval delays.

This creates realistic trade-offs between speed and compliance.

## Configuration

Server supports environment variables:

- `CYBERSEC_SCENARIO_ID` (default `supply_chain_token_drift`)
- `CYBERSEC_HORIZON` (optional override)
- `CYBERSEC_FALSE_POSITIVE_RATE` (optional override)
- `CYBERSEC_DEFAULT_SEED` (optional deterministic default)
- `CYBERSEC_MAX_CONCURRENT_ENVS` (default `4`)

## Build and Deploy

### Docker build

```bash
docker build -t cybersec-env:latest -f server/Dockerfile .
```

### Runtime validation

```bash
python -m pytest -q
openenv validate .
```

### OpenEnv validation

```bash
openenv validate .
```

### Push to Hugging Face Space

```bash
openenv push --repo-id <your-namespace>/<your-space-name>
```

## Project Structure

```text
cybersec/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py
    ├── cybersec_environment.py
    ├── attacker_policy.py
    ├── telemetry.py
    ├── workflow.py
    ├── world_state.py
    ├── reward_model.py
    ├── scenario_loader.py
    ├── requirements.txt
    └── Dockerfile
```

## Detailed Documentation

- `docs/spec.md` formal environment contract and POMDP framing.
- `docs/reward.md` reward decomposition and grading logic.
- `docs/scenarios.md` deterministic scenario catalog.
- `docs/evaluation.md` benchmark and demo protocol.

## Baselines and Evaluation Scripts

- Baseline policies and runner: `baselines/policy.py`, `baselines/runner.py`
- Multi-scenario benchmark runner: `scripts/evaluate_baselines.py`
- Deterministic trace exporter: `scripts/run_demo_trace.py`
- Inference runner (local or remote): `inference.py`

Example:

```bash
python scripts/evaluate_baselines.py --scenarios supply_chain_token_drift,federated_identity_takeover --seeds 101,202,303
python scripts/run_demo_trace.py --scenario insider_repo_pivot --seed 404
```

Generated artifacts are written to `outputs/evals/`.

## Inference Runner

Run deterministic inference episodes (baseline heuristic/random or LLM-backed policy):

```bash
# local heuristic inference
python inference.py --env local --mode heuristic --scenarios supply_chain_token_drift --seeds 101

# remote inference against a running endpoint
ENV_BASE_URL=http://localhost:8000 python inference.py --env remote --mode heuristic

# remote LLM-guided inference (falls back to heuristic on parse/contract errors)
HF_TOKEN=<token> API_BASE_URL=https://router.huggingface.co/v1 python inference.py --env remote --mode llm
```

Output summary is written to `outputs/evals/inference_results.json` by default.

## Training Scaffold

- Training scaffold docs: `training/README.md`
- Dataset export utility: `training/export_dataset.py`
- This includes baseline/evaluation entry points plus trajectory export for downstream RL/behavioral training.
