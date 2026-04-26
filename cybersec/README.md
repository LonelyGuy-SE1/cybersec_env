---
title: Cybersec OpenEnv Environment
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - cybersecurity
  - mitre-attack
  - llm-rl
---

# Cybersec OpenEnv

Long-horizon **enterprise cyber-defense** simulation for RL and LLM agents, built on **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)**. A **scripted attacker** advances a MITRE ATT&CK–style stage DAG while a **defender** chooses one structured action per tick from noisy telemetry.

[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-fastapi-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/pyproject.toml)

**Full project** (story, training, notebooks, results): [repository root README](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/README.md) · **Design blog**: [BLOG.md](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/BLOG.md)

---

## Table of contents

1. [Why this environment?](#1-why-this-environment)
2. [Quick start](#2-quick-start)
3. [Environment overview](#3-environment-overview)
4. [Scenarios](#4-scenarios)
5. [Attacker personalities](#5-attacker-personalities)
6. [Defender actions](#6-defender-actions)
7. [Observations](#7-observations)
8. [Reward](#8-reward)
9. [Reset & server configuration](#9-reset--server-configuration)
10. [HTTP API (`/web`)](#10-http-api-web)
11. [Python client](#11-python-client)
12. [Docker & Hugging Face Spaces](#12-docker--hugging-face-spaces)
13. [Layout & tests](#13-layout--tests)
14. [References](#14-references)

---

## 1. Why this environment?

Real SOC work is a **partially observable**, **multi-step** game: alerts lag, benign noise appears, and containment has operational cost. This package implements that **POMDP** in-process (`CybersecEnvironment`) and behind a **FastAPI** server for remote clients and Spaces.

---

## 2. Quick start

### Docker (recommended for local server)

From the **repository root** (build context is the `cybersec/` folder):

```bash
docker build -t cybersec-env:latest -f cybersec/server/Dockerfile cybersec
docker run --rm -p 8000:8000 cybersec-env:latest
```

OpenEnv HTTP routes are served under **`/web`** when deployed as a Space (see `base_path` in this file’s frontmatter). Locally, the app listens on port **8000** (see `openenv.yaml` / `CYBERSEC_PORT`).

### Python (in-process)

```python
from cybersec import CybersecEnvironment
from cybersec.baselines import HeuristicPolicy, run_episode

env = CybersecEnvironment()
result = run_episode(env, HeuristicPolicy(), seed=0, scenario_id="supply_chain_token_drift")
print(result.cumulative_reward, result.terminal_reason)
```

### Python (remote WebSocket client)

```python
from cybersec import CybersecEnv, CybersecAction, ActionType

async with CybersecEnv(base_url="https://YOUR-SPACE.hf.space") as env:
    r = await env.reset(seed=42, scenario_id="federated_identity_takeover")
    while not r.done:
        r = await env.step(CybersecAction(action_type=ActionType.MONITOR))
```

See [`client.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/client.py) for the async API.

---

## 3. Environment overview

| Aspect | Detail |
|--------|--------|
| **Roles** | Scripted attacker vs defender (your policy / LLM) |
| **Time** | Discrete **ticks**; each defender action consumes one tick |
| **Termination** | Horizon reached, attacker exfil completes, or attacker DAG exhausted (see server implementation) |
| **Observation** | Alerts, forensics, containment lists, `valid_targets`, `available_actions`, `info.reward_breakdown` |
| **Training vs eval** | `list_train_scenarios()` = 3 IDs; `list_scenarios()` adds one **held-out** OOD scenario |

---

## 4. Scenarios

Stage and horizon counts match [`cybersec/scenarios.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/scenarios.py).

### Training (`list_train_scenarios()`)

| ID | Summary | Stages | Horizon |
|----|---------|:------:|:-------:|
| `supply_chain_token_drift` | CI token → artifact → payments → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Phish / MFA fatigue → pivots → cloud egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo → secrets → staging → prod → DB exfil | 6 | 80 |

### Held-out evaluation (`list_eval_scenarios()`)

| ID | Summary | Stages | Horizon |
|----|---------|:------:|:-------:|
| `cloud_metadata_ssrf` | SSRF → metadata creds → role chain → KMS → cloud exfil | 5 | 70 |

---

## 5. Attacker personalities

Passed to `reset(..., attacker_personality=...)` or sampled when omitted (see [`cybersec_environment.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/server/cybersec_environment.py)).

| Personality | Dwell × | Detection × | Pause after defender | Reroutes |
|-------------|:------:|:-----------:|:--------------------:|:--------:|
| `stealthy` | 1.5× | 0.55× | 50% | no |
| `aggressive` | 0.6× | 1.30× | 0% | no |
| `opportunistic` | 1.0× | 1.0× | 15% | yes |

---

## 6. Defender actions

Defined in [`cybersec/models.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/models.py) (`CybersecAction`, `ActionType`).

| `action_type` | `target` | Rule |
|---------------|----------|------|
| `MONITOR` | omit / `null` | Must not include a target |
| `INVESTIGATE` | required | Asset or identity in `valid_targets` |
| `ISOLATE_ASSET`, `BLOCK_EGRESS`, `PATCH_ASSET` | required | Asset in `valid_targets.assets` |
| `REVOKE_IDENTITY` | required | Identity in `valid_targets.identities` |

Invalid actions **still advance one tick** and incur **`invalid_action_penalty`** in [`reward.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/reward.py).

---

## 7. Observations

[`CybersecObservation`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/models.py) includes:

- **Episode:** `tick`, `horizon`, `scenario_id`, `attacker_personality`
- **Telemetry:** `alerts` (`signal`, `severity`, `asset`, `identity`, `description`, …), `forensics`
- **Defender state:** `isolated_assets`, `revoked_identities`, `blocked_egress_assets`, `patched_assets`, `confirmed_compromised`
- **Grounding:** `valid_targets` as `{"assets": [...], "identities": [...]}`, `available_actions`
- **Diagnostics:** `info` dict with per-step `reward_breakdown` and terminal metadata on the last step

---

## 8. Reward

Seven channels are summed (with clipping) into the step reward; each step exposes them under `obs.info["reward_breakdown"]`. See [`reward.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/reward.py).

| Channel | Role |
|---------|------|
| `detection` | First confirmation of attacker-path compromise |
| `containment` | Active / preemptive containment net of costs |
| `evidence_bonus` | Contain after confirmed compromise |
| `false_positive_penalty` | Containment off attack path |
| `disruption_penalty` | Cost of isolation / egress blocks (uncapped by default) |
| `invalid_action_penalty` | Illegal target or verb |
| `terminal_score` | Episode outcome |

---

## 9. Reset & server configuration

Implemented in [`cybersec_environment.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/server/cybersec_environment.py) and [`app.py`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/server/app.py).

| Input / env var | Effect |
|-----------------|--------|
| **`scenario_id`** in reset body | Fixes scenario to one of `list_scenarios()`. If **omitted**: use process default **`CYBERSEC_SCENARIO_ID`** if set; else choose via `list_scenarios()[rng_seed % len]` |
| **`seed`** in reset body | Fixes RNG for reproducibility. If **omitted**: a **new random seed** is drawn each reset (avoids always picking the same scenario index) |
| **`attacker_personality`** | Optional override; else sampled or default |
| **`CYBERSEC_SCENARIO_ID`** | Server default `scenario_id` when client omits it |
| **`CYBERSEC_ATTACKER_PERSONALITY`** | Server default personality when client omits it |
| **`CYBERSEC_MAX_CONCURRENT_ENVS`** | WebSocket concurrency cap (default 8) |

---

## 10. HTTP API (`/web`)

Use **`Content-Type: application/json`**. Replace `BASE` with your Space or `http://localhost:8000`.

### `POST BASE/web/reset`

```json
{}
```

```json
{"seed": 42, "scenario_id": "insider_repo_pivot", "attacker_personality": "stealthy"}
```

### `POST BASE/web/step`

OpenEnv wraps the action:

```json
{"action": {"action_type": "MONITOR"}}
```

```json
{"action": {"action_type": "INVESTIGATE", "target": "ci-runner-01"}}
```

Training-time LLM completions use a **flat** JSON object `{"action_type": "...", "target": "..."}` per [`SYSTEM_PROMPT`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/training/rewards.py); HTTP step uses the **`action`** wrapper above.

---

## 11. Python client

Install the package from the repo:

```bash
pip install -e ./cybersec
```

Entry points: `cybersec-server` (see `pyproject.toml`).

---

## 12. Docker & Hugging Face Spaces

- **Build / run:** see §2.
- **Publish:** from `cybersec/`, use OpenEnv CLI **`openenv push`** with [`openenv.yaml`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/openenv.yaml).
- **Live Space:** [Lonelyguyse1/cybersec on Hugging Face](https://huggingface.co/spaces/Lonelyguyse1/cybersec).

---

## 13. Layout & tests

```
cybersec/
├── README.md          # this file (Space card + env spec)
├── openenv.yaml
├── pyproject.toml
├── models.py
├── scenarios.py
├── reward.py
├── baselines.py
├── client.py
├── server/
│   ├── app.py
│   ├── cybersec_environment.py
│   └── Dockerfile
└── training/
    └── rewards.py
```

Tests live at repo root: `pytest -q`.

---

## 14. References

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) / [openenv-core](https://github.com/meta-pytorch/openenv-core)
- [MITRE ATT&CK](https://attack.mitre.org/matrices/enterprise/)
- [TRL — GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)

**License:** MIT — see [`pyproject.toml`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/pyproject.toml).
