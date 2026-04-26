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

# Cybersec OpenEnv (Space)

Long-horizon defender vs scripted attacker on MITRE-style kill chains. This Space serves the **OpenEnv** HTTP API under **`/web`** (e.g. `POST …/web/reset`).

**Main project (install, training, blog):**  
[github.com/LonelyGuy-SE1/cybersec_env — root README](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/README.md)

**GRPO / Colab notebook:**  
[notebooks/cybersec_grpo.ipynb](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb)

---

## Expected API inputs (HTTP)

All bodies are **`Content-Type: application/json`**. Replace `BASE` with this Space’s URL (e.g. `https://*.hf.space`).

### `POST BASE/web/reset`

Optional fields:

| Field | Type | Notes |
|-------|------|--------|
| `seed` | int | Omit for a fresh random episode seed (recommended for variety). Set for reproducibility. |
| `scenario_id` | string | One of the scenario IDs below. Omit to pick from the list using the seed (or server default / env). |
| `attacker_personality` | string | Optional: `stealthy`, `aggressive`, `opportunistic`. |

**Examples**

```json
{}
```

```json
{"seed": 42}
```

```json
{"seed": 0, "scenario_id": "federated_identity_takeover"}
```

```json
{"seed": 7, "scenario_id": "insider_repo_pivot", "attacker_personality": "stealthy"}
```

### `POST BASE/web/step`

Wrap the defender action in an **`action`** object (OpenEnv shape).

**`MONITOR`** — no target:

```json
{"action": {"action_type": "MONITOR"}}
```

**Actions that require `target`** — must be an **asset** or **identity** from the current observation’s `valid_targets` (not shown here; they vary by scenario):

```json
{"action": {"action_type": "INVESTIGATE", "target": "ci-runner-01"}}
```

```json
{"action": {"action_type": "ISOLATE_ASSET", "target": "payments-svc"}}
```

```json
{"action": {"action_type": "REVOKE_IDENTITY", "target": "svc-ci-deploy"}}
```

```json
{"action": {"action_type": "BLOCK_EGRESS", "target": "egress-proxy"}}
```

```json
{"action": {"action_type": "PATCH_ASSET", "target": "api-gateway"}}
```

Allowed `action_type` values: `MONITOR`, `INVESTIGATE`, `ISOLATE_ASSET`, `REVOKE_IDENTITY`, `BLOCK_EGRESS`, `PATCH_ASSET`.  
`MONITOR` must not include `target`. Invalid targets still **consume a tick** and incur a penalty.

### Scenario IDs

**Training:** `supply_chain_token_drift`, `federated_identity_takeover`, `insider_repo_pivot`  
**Held-out (eval):** `cloud_metadata_ssrf`

---

## Clients & config

* **Python:** `from cybersec import CybersecEnv, CybersecAction, ActionType` — point `CybersecEnv(base_url=…)` at this Space (see [client.py](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/client.py) on GitHub).
* **Server env (optional):** `CYBERSEC_SCENARIO_ID`, `CYBERSEC_ATTACKER_PERSONALITY`, `CYBERSEC_MAX_CONCURRENT_ENVS` — see [server/app.py](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/server/app.py).

---

## More detail

Full narrative, scenario tables, reward channels, layout, and `openenv push` workflow live in the **repository**:

| Doc | Link |
|-----|------|
| Project overview & motivation | [README.md](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/README.md) |
| This file (editable) | [cybersec/README.md](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/README.md) |

**References:** [OpenEnv](https://github.com/meta-pytorch/openenv-core) · [MITRE ATT&CK](https://attack.mitre.org/matrices/enterprise/) · [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)

License: MIT (see repo).
