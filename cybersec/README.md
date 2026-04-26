---
title: Cybersec OpenEnv Environment
emoji: "\U0001F6E1"
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

# Cybersec OpenEnv: long-horizon multi-agent cyber defense

A production-oriented [OpenEnv](https://github.com/meta-pytorch/openenv-core)
environment for training LLM defenders against scripted attackers on
[MITRE ATT&CK](https://attack.mitre.org/)-aligned kill chains.

**Summary:** Three enterprise breach scenarios for training, one held-out
scenario for out-of-distribution evaluation, multiple attacker personalities,
a partially observable defender with six actions, and a six-channel step
reward suitable for LLM RL on a single consumer GPU (e.g. Colab T4 class
hardware).

This document is the canonical environment specification. It is also used as
the Hugging Face Spaces card when the environment is published with `openenv push`.

---

## 1. Design goals

The environment emphasizes **long-horizon planning** and **multi-agent
interaction**:

* **Staged attacks:** Credentials, dwell time, pivot, and exfiltration unfold
  over many ticks with stochastic timing; early signals can be weak.
* **Two roles:** A scripted attacker drives ground truth; the defender
  observes partial state and selects one structured action per tick. The
  attacker implementation is pluggable (see §7).
* **Information asymmetry:** Detection lags, noisy alerts, and branching
  across assets require non-trivial investigation and containment policies.

The action and reward surfaces are kept compact so modest-sized language
models can consume observations in a few thousand tokens and emit reliable
structured actions.

---

## 2. The world

Each episode runs **one scenario** with **one attacker personality**, derived
from the seed (or set via reset arguments).

### Scenarios (`scenarios.py`)

**Training scenarios:**

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (evaluation-only) scenario** — excluded from default GRPO training
to measure generalisation:

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → assumed role → cloud storage exfil | 4 | 60 |

Stages reference MITRE tactics and techniques (e.g. `T1552.004`, `T1621`,
`T1041`).

### Attacker personalities (`attacker.py`)

| Personality     | Dwell × | Detection × | Pause-after-defender | Reroutes |
|---|---|---|---|---|
| `stealthy`      | 1.5×    | 0.55×       | 50%                 | no       |
| `aggressive`    | 0.6×    | 1.30×       | 0%                  | no       |
| `opportunistic` | 1.0×    | 1.0×        | 15%                 | yes      |

The default attacker is scripted for reproducibility and clear credit
assignment. An optional `LLMAttacker` adapter (§7) can substitute for extended
experiments.

### Defender contract (`models.py`)

Actions use a single optional string `target`:

```json
{"action_type": "ISOLATE_ASSET", "target": "payments-svc"}
```

| Action            | Target           | Effect |
|---|---|---|
| `MONITOR`         | (none)           | Low cost; advances time and observations. |
| `INVESTIGATE`     | asset / identity | Noisy forensic signal on the target. |
| `ISOLATE_ASSET`   | asset            | Quarantine; can interrupt in-progress stages. |
| `REVOKE_IDENTITY` | identity         | Revoke credentials; blocks identity pivots. |
| `BLOCK_EGRESS`    | asset            | Containment oriented to exfiltration. |
| `PATCH_ASSET`     | asset            | One-shot hardening; lowers stage success odds. |

Observations expose `available_actions`, `valid_targets`, and containment
state. Invalid actions consume a tick and incur `invalid_action_penalty`.

### Reward (`reward.py`)

Six channels, surfaced in `obs.info["reward_breakdown"]` each step:

| Channel                    | Sign | Role |
|---|---|---|
| `detection`               | +    | First confirmed compromise of an attacker target |
| `containment`             | +/-  | Active and preemptive containment, net of action cost |
| `false_positive_penalty`  | −    | Containment on non-attack-path targets |
| `disruption_penalty`      | −    | Operational cost of isolations and egress blocks |
| `invalid_action_penalty`  | −    | Structurally invalid actions |
| `terminal_score`          | ±    | Episode outcome: clean resolution vs exfiltration |

Per-step total magnitude is bounded so rare spikes do not dominate learning.
Detection and active containment credit repeat targets only once, which
limits trivial repeated-isolation strategies.

---

## 3. Project layout

```
cybersec/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── __init__.py
├── client.py
├── models.py
├── scenarios.py
├── attacker.py
├── telemetry.py
├── reward.py
├── baselines.py
├── py.typed
├── server/
│   ├── cybersec_environment.py
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── training/
    ├── __init__.py
    └── rewards.py
```

`tests/` and `notebooks/` live at the repository root. A `pip install` of the
`cybersec` package includes only the `cybersec/` tree.

---

## 4. Quickstart

### Install (from `cybersec/`)

```bash
cd cybersec/
pip install -e .[dev]
```

### Tests (from repository root)

```bash
pytest -q
```

### In-process episode

```python
from cybersec import CybersecEnvironment
from cybersec.baselines import HeuristicPolicy, run_episode

env = CybersecEnvironment()
result = run_episode(env, HeuristicPolicy(), seed=0)
print(result.cumulative_reward, result.terminal_reason)
```

### OpenEnv server

```bash
cybersec-server
# or: uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Remote client:

```python
from cybersec import CybersecEnv, ActionType, CybersecAction

with CybersecEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=0)
    while not result.observation.done:
        result = env.step(CybersecAction(action_type=ActionType.MONITOR))
```

### Docker

```bash
cd cybersec/
docker build -t cybersec-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 cybersec-env:latest
```

### Hugging Face Spaces

```bash
cd cybersec/
openenv push
```

---

## 5. Reference baselines

`RandomPolicy` and `HeuristicPolicy` in `baselines.py` define simple reference
policies. Example mean returns below were measured over 30 seeds per scenario
against a deployed Space using the WebSocket client (environment version at
time of measurement):

| Scenario                       | Random — return | Heuristic — return | Heuristic — exfil rate |
|---|---|---|---|
| `supply_chain_token_drift`     |  3.34           |  **3.36**          | 13%                    |
| `federated_identity_takeover`  | -1.10           |  **2.93**          | 7%                     |
| `insider_repo_pivot`           |  **2.66**       | -2.69              | 7%                     |
| `cloud_metadata_ssrf` (OOD)    |  1.45           |  **2.55**          | 0%                     |

Reproduce with [`../notebooks/cybersec_grpo.ipynb`](../notebooks/cybersec_grpo.ipynb)
or by pointing `CybersecEnv` at the Space URL. Random can exceed the heuristic
on some scenarios under this reward definition; that gap motivates learning a
policy that improves on both.

---

## 6. GRPO training

**Notebook:** [`../notebooks/cybersec_grpo.ipynb`](../notebooks/cybersec_grpo.ipynb) — baselines, plots, eval, canaries.

**Headless script (same GRPO outer loop):** [`../scripts/train_cybersec_grpo.py`](../scripts/train_cybersec_grpo.py) calls [`training/run_grpo.py`](training/run_grpo.py) (`train_grpo`). Install `./cybersec[grpo]` plus Unsloth for your CUDA stack; see the repository root `README.md`.

Flow ([TRL GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer), [OpenEnv + TRL](https://huggingface.co/docs/trl/en/openenv)): baselines → on-policy rows via `collect_grpo_rows_from_rollouts` in [`training/rewards.py`](training/rewards.py) → Unsloth 4-bit QLoRA + `GRPOTrainer` → save adapter + `training_log.json`.

Set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` for Hub pulls on Colab or CI.

Regression expectations for reward shaping live in `tests/test_reward_hack_canaries.py`.

---

## 7. Design notes

* **Six actions:** A wider surface previously increased invalid JSON and
  target-type errors on small models; the current contract improves validity
  rates.
* **Containment semantics:** Splitting active vs preemptive credit reduces
  reward for undifferentiated mass isolation.
* **Investigation:** Investigating clean targets does not add a false-positive
  penalty; time cost provides natural regularisation.
* **Single in-flight attacker stage:** Simplifies credit assignment and keeps
  long-horizon structure readable.
* **Pluggable attacker:** `ScriptedAttacker` is the default; the interface
  supports substitution (e.g. `LLMAttacker`) without changing the environment
  core.
* **`SUPPORTS_CONCURRENT_SESSIONS = True`:** Each session owns an independent
  world. Concurrency limits are configurable (e.g. `CYBERSEC_MAX_CONCURRENT_ENVS`).

---

## 8. References

* [OpenEnv core](https://github.com/meta-pytorch/openenv-core)
* [MITRE ATT&CK Enterprise Matrix](https://attack.mitre.org/matrices/enterprise/)
* [TRL — GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)
* [Unsloth](https://github.com/unslothai/unsloth)

## 9. License

MIT.
