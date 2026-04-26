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

| Doc | Link |
|-----|------|
| Project overview, install, training | [Root README](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/README.md) |
| Narrative: GRPO, reward hacks, dispersion | [**BLOG.md**](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/BLOG.md) |
| GRPO notebook (Colab-ready) | [cybersec_grpo.ipynb](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb) |

---

## Environment contract (summary)

### Scenarios

| ID | Summary | Stages | Horizon |
|----|---------|--------|---------|
| `supply_chain_token_drift` | CI token → artifact → payments → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Phish / MFA fatigue → pivots → cloud egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo → secrets → staging → prod → DB exfil | 6 | 80 |
| `cloud_metadata_ssrf` *(held-out / eval)* | SSRF → metadata creds → role chain → KMS → cloud exfil | 5 | 70 |

Training code should use **`list_train_scenarios()`** (first three only). **`list_scenarios()`** includes the held-out row for OOD evaluation.

### Attacker personalities (`reset(..., attacker_personality=...)`)

| Value | Dwell × | Detection × | Pause after defender | Reroutes |
|-------|---------|-------------|----------------------|----------|
| `stealthy` | 1.5× | 0.55× | 50% | no |
| `aggressive` | 0.6× | 1.30× | 0% | no |
| `opportunistic` | 1.0× | 1.0× | 15% | yes |

If omitted, personality is **sampled** from this set using the episode RNG.

### Defender actions

| `action_type` | `target` | Notes |
|---------------|----------|--------|
| `MONITOR` | omit / `null` | Must not send a target. |
| `INVESTIGATE` | required | Asset or identity string from `valid_targets`. |
| `ISOLATE_ASSET`, `BLOCK_EGRESS`, `PATCH_ASSET` | required | Asset id from `valid_targets.assets`. |
| `REVOKE_IDENTITY` | required | Identity id from `valid_targets.identities`. |

Invalid actions **still advance one tick** and add **`invalid_action_penalty`** (see `cybersec/reward.py`).

### Observation highlights

Per tick the defender sees: `tick`, `horizon`, `scenario_id`, `attacker_personality`, `alerts[]` (each: `signal`, `severity`, `asset`/`identity`, `description`, …), `forensics[]` (from prior investigations), lists of isolated / revoked / blocked / patched / `confirmed_compromised`, **`valid_targets`**, `available_actions`, and `info` (includes `reward_breakdown` each step; terminal bundle on the last step).

### Reset semantics

* **`scenario_id`** in the JSON body selects the scenario. If omitted: use server default **`CYBERSEC_SCENARIO_ID`** (if set), else pick from **`list_scenarios()`** using the episode RNG.
* **`seed`** (int): fixes RNG for a **reproducible** episode. If omitted: a **fresh random seed** is drawn so successive `{}` resets are not all identical (and scenario rotation is not stuck on list position 0).
* Optional **`CYBERSEC_ATTACKER_PERSONALITY`** on the server pins default personality when the client omits it.

### Reward (`obs.info["reward_breakdown"]`)

Seven channels (sum ≈ step reward, clipped): `detection`, `containment`, `evidence_bonus`, `false_positive_penalty`, `disruption_penalty`, `invalid_action_penalty`, `terminal_score`. See `cybersec/reward.py` and the **BLOG** for design rationale (disruption exploit, evidence bonus, uncapped disruption).

---

## Expected API inputs (HTTP)

All bodies are **`Content-Type: application/json`**. Replace `BASE` with this Space’s URL (e.g. `https://*.hf.space`).

### `POST BASE/web/reset`

Optional fields:

| Field | Type | Notes |
|-------|------|--------|
| `seed` | int | Omit for a **new random** episode each reset. Set for reproducibility. |
| `scenario_id` | string | One of the scenario IDs in the table above. Omit to use **`CYBERSEC_SCENARIO_ID`** if the Space set it; else scenario follows the episode RNG (see Reset semantics). |
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
`MONITOR` must not include `target`. Invalid targets still **consume a tick** and incur a penalty. Scenario IDs are listed in **Environment contract** above.

---

## Clients & config

* **Python:** `from cybersec import CybersecEnv, CybersecAction, ActionType` — point `CybersecEnv(base_url=…)` at this Space (see [client.py](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/client.py) on GitHub).
* **Server env (optional):** `CYBERSEC_SCENARIO_ID`, `CYBERSEC_ATTACKER_PERSONALITY`, `CYBERSEC_MAX_CONCURRENT_ENVS` — see [server/app.py](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/server/app.py).

---

## Layout, tests, `openenv push`

Package tree, `pytest`, Docker, and publishing this Space: [root README § install / server](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/README.md#install-and-train). From `cybersec/`: `openenv push` (see `openenv.yaml`).

**References:** [OpenEnv](https://github.com/meta-pytorch/openenv-core) · [MITRE ATT&CK](https://attack.mitre.org/matrices/enterprise/) · [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)

License: MIT (see repo `pyproject.toml`).
