<div align="center">

<img src="https://img.shields.io/badge/Model-Qwen2.5--1.5B-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white" />
<img src="https://img.shields.io/badge/Algorithm-GRPO-ff6b35?style=for-the-badge" />
<img src="https://img.shields.io/badge/Framework-MITRE%20ATT%26CK-cc0000?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />
<img src="https://img.shields.io/badge/GPU-T4%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />

<br /><br />

```
 ██████╗██╗   ██╗██████╗ ███████╗██████╗ ███████╗███████╗ ██████╗
██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝
██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝███████╗█████╗  ██║
██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗╚════██║██╔══╝  ██║
╚██████╗   ██║   ██████╔╝███████╗██║  ██║███████║███████╗╚██████╗
 ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝
                        O P E N E N V
```

### *Teaching Small LLMs Surgical Cyber-Defense*

**A long-horizon, adversarial RL environment where a 1.5B-parameter model learns to think like a senior SOC analyst — without burning down the network.**

<br />

[**Play the Live Environment**](https://huggingface.co/spaces/Lonelyguyse1/cybersec) &nbsp;|&nbsp; [**Design Blog**](BLOG.md) &nbsp;|&nbsp; [**Training notebook (Colab)**](https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb)

</div>

---

## Table of contents

1. [Why I built this](#1-why-i-built-this)
2. [The small model manifesto](#2-the-small-model-manifesto)
3. [Environment: technical specs](#3-environment-technical-specs)
   - [Scenarios](#scenarios)
   - [The adversary (scripted)](#the-adversary-scripted)
   - [The defender & action space](#the-defender-the-llm--action-space)
   - [Observation & action schema](#observation--action-schema)
4. [Training methodology](#4-training-methodology)
   - [Reward model (7 channels)](#the-reward-model-7-channels)
   - [The disruption exploit & the fix](#the-disruption-exploit--the-fix)
5. [Results](#5-results)
6. [Evaluation tasks & grading](#6-evaluation-tasks--grading)
7. [Notebooks](#7-notebooks)
8. [Quickstart](#8-quickstart)
9. [Running the server](#9-running-the-server)

---

## 1. Why I built this

> *The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of your own.*

Thanks to the rise of powerful SOTA models, almost anyone now has access to a tool of immense intelligence. This is incredible — but it also arms bad actors. I have watched this play out firsthand with recent attacks on major internet infrastructure, where absolute amateurs, purely harnessing AI, penetrate systems that had been considered hardened for years. Projects like Mythos have found 0-days in established codebases with millions of lines of code.

Existing cyber environments treat the problem as a **static, point-in-time classification task** ("is this payload bad?"). Real attacks don't work that way. Attackers plan, gather intelligence, wait, and pivot across systems over days or weeks.

**Cybersec OpenEnv** was built to simulate the real *fog of war* inside a Security Operations Center:

- **Long-horizon attacks** — staged, multi-step campaigns that unfold over many environment ticks
- **Delayed telemetry** — early signals are deliberately weak and noisy
- **Real business costs** — taking systems offline has consequences, modeled explicitly

Training an RL agent on a live production network is risky and expensive. This environment makes that training safe, fast, and reproducible.

---

## 2. The small model manifesto

The defender is a **`Qwen2.5-1.5B-Instruct`** model — not a 70B behemoth. Here's why that's a feature, not a limitation:

| Constraint | Why it matters |
|------------|----------------|
| **Iteration speed** | Multiple RL training cycles fit on a single Colab T4 GPU |
| **Privacy & security** | No sensitive network telemetry ever has to leave your infrastructure to an external API |
| **Edge deployment** | 70B models aren't viable for mass enterprise adoption; this is built for the real world |

The goal: prove that small, open-source LLMs can be fine-tuned to reason like **risk-aware senior SOC analysts**.

---

## 3. Environment: technical specs

Cybersec OpenEnv is a **long-horizon, partially observable Markov decision process (POMDP)** in which two agents — a scripted adversary and an LLM defender — interact within a simulated enterprise network.

The core challenge: **staged attacks unfold over many ticks with stochastic timing, and early compromise signals can be extremely weak.**

Full HTTP examples, reset semantics, and the **OpenEnv Space card** live in **[`cybersec/README.md`](cybersec/README.md)** (same layout style as [OpenEnv wildfire_env](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/wildfire_env)).

### Scenarios

Three training scenarios and one held-out evaluation scenario (counts and horizons match `cybersec/scenarios.py`):

**Training**

| ID | Campaign | Stages | Horizon |
|----|----------|:------:|:-------:|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (evaluation only)**

| ID | Campaign | Stages | Horizon |
|----|----------|:------:|:-------:|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → role-chain → KMS replicate → cloud storage exfil | 5 | 70 |

### The adversary (scripted)

The attacker walks a **deterministic MITRE ATT&CK-aligned directed acyclic graph (DAG)**. Each node is a sequential stage the attacker must complete to reach exfiltration. One of three personalities is sampled per episode (unless overridden on `reset`):

| Personality | Dwell time | Detection risk | Pauses after defender | Reroutes? |
|-------------|:----------:|:----------------:|:---------------------:|:---------:|
| `stealthy` | 1.5× slower | 0.55× lower | 50% chance | No |
| `aggressive` | 0.6× faster | 1.30× higher | Never | No |
| `opportunistic` | Baseline | Baseline | 15% chance | Yes |

### The defender (the LLM) & action space

Each tick, the LLM receives a **partial observation** — lagged alerts, forensic results from past `INVESTIGATE` calls, containment state, and **`valid_targets`** as `{"assets": [...], "identities": [...]}` — and must emit a single structured JSON action (see `SYSTEM_PROMPT` in `cybersec/training/rewards.py`).

| Action | Target | Effect |
|--------|--------|--------|
| `MONITOR` | *(none)* | Low-cost; advances time and accumulates new observations |
| `INVESTIGATE` | asset / identity | Returns a noisy forensic signal on the target |
| `ISOLATE_ASSET` | asset | Quarantines the asset; can interrupt in-progress attack stages |
| `REVOKE_IDENTITY` | identity | Revokes credentials; blocks identity-based pivots |
| `BLOCK_EGRESS` | asset | Containment focused on preventing data exfiltration |
| `PATCH_ASSET` | asset | One-shot hardening; lowers future stage success probability |

> Invalid or out-of-set targets **still consume a tick** and incur an **`invalid_action_penalty`** (`cybersec/reward.py`).

### Observation & action schema

<details>
<summary><strong>Expand: example observation (tick 4 of <code>federated_identity_takeover</code>)</strong></summary>

```json
{
  "tick": 4,
  "horizon": 70,
  "scenario_id": "federated_identity_takeover",
  "attacker_personality": "stealthy",
  "alerts": [
    {
      "tick": 4,
      "signal": "auth_anomaly",
      "asset": null,
      "identity": "u-platform-eng",
      "severity": 0.72,
      "description": "Impossible travel + new device fingerprint"
    }
  ],
  "forensics": [],
  "valid_targets": {
    "assets": ["api-gateway", "egress-proxy"],
    "identities": ["u-platform-eng", "svc-ci-deploy"]
  }
}
```

</details>

<details>
<summary><strong>Expand: example LLM completion (training)</strong></summary>

```json
{"action_type": "INVESTIGATE", "target": "u-platform-eng"}
```

After a forensic row is confirmed, the model might follow with:

```json
{"action_type": "REVOKE_IDENTITY", "target": "u-platform-eng"}
```

</details>

**HTTP / OpenEnv web UI:** actions are posted as `{"action": {"action_type": "...", "target": "..."}}` to **`POST …/web/step`**. See [`cybersec/README.md`](cybersec/README.md) for curl examples.

**Reset parameters:** optional `scenario_id`, `seed`, `attacker_personality` on **`POST …/web/reset`**. If **`scenario_id`** is omitted, the server uses **`CYBERSEC_SCENARIO_ID`** when set; otherwise the scenario index is derived from the episode RNG. If **`seed`** is omitted, a **fresh random seed** is drawn each reset (so bare `{}` is not stuck on a single scenario). Details: `cybersec/server/cybersec_environment.py`, `cybersec/server/app.py`.

---

## 4. Training methodology

Training uses **Group Relative Policy Optimization (GRPO)** via Hugging Face **TRL** and **Unsloth** (4-bit QLoRA), runnable end-to-end on a single T4-class GPU.

### The reward model (7 channels)

Rather than sparse 0/1 win/loss feedback, the environment exposes a **dense 7-channel** signal every tick (`obs.info["reward_breakdown"]`):

| Channel | Sign | Purpose |
|---------|:----:|---------|
| `detection` | + | First confirmed compromise of an attacker target |
| `containment` | +/− | Net reward for active and preemptive containment, minus action cost |
| `evidence_bonus` | + | Extra reward for containing targets **already confirmed** via `INVESTIGATE` |
| `false_positive_penalty` | − | Penalty for containment on non-attack-path targets |
| `disruption_penalty` | − | Operational cost of isolations and egress blocks |
| `invalid_action_penalty` | − | Illegal actions (bad target, wrong verb type, etc.) |
| `terminal_score` | ± | Episode outcome: clean resolution bonus vs exfiltration penalty |

### The disruption exploit & the fix

> **The model found a degenerate cheat code.**

During initial GRPO training, the policy achieved a high average return — but the **standard deviation of returns across the batch was exactly `0.0`**. Every rollout was scoring identically.

The LLM had discovered that isolating every asset on the network at **Tick 0** — completely nuking the entire business — was the mathematically optimal play under the original reward function. The disruption penalty had a hard per-tick cap, making mass isolation cheap at scale.

*"I solved the hack by unplugging the internet."*

**Three fixes, applied together:**

1. **Remove the disruption cap** — penalty scales with isolation load (default uncapped in `RewardWeights.disruption_cap_per_tick`).
2. **Evidence-based containment bonus** — containing a target already confirmed via `INVESTIGATE` yields **`evidence_bonus_per_target`** (default 1.5 in `reward.py`); blind Tick-0 isolation does not.
3. **On-policy iterative self-play** — no static offline dataset for the outer loops.

**Training loop (high level):**

```
Outer loop 0 (warmup)
└── Heuristic policy generates seed rows (e.g. 1,500)
└── GRPO steps: model learns JSON schema and mechanics

Outer loop 1+ (true RL)
└── LLM rollouts at high temperature (~1.4 in default MODE)
└── Fresh rows each loop; mistakes + false positives provide signal
```

GRPO training also uses **dispersion / shaping** helpers in `cybersec/training/rewards.py` (e.g. `reward_action_diversity`, `reward_observation_aware`, `reward_batch_action_entropy`, `reward_evidence_containment`) so groups retain variance. See **[BLOG.md](BLOG.md)** for the full narrative.

---

## 5. Results

> Training curves, per-scenario breakdowns, and ablation tables will be filled in after the current training run completes.

### Training performance

| Metric | Outer loop 0 (heuristic) | Outer loop 1+ (self-play) |
|--------|:------------------------:|:-------------------------:|
| Mean episode reward | — | — |
| Detection rate | — | — |
| False positive rate | — | — |
| Exfiltration prevented | — | — |

### Per-scenario breakdown

| Scenario | Attacker personality | Stages stopped | Exfil prevented | Terminal outcome |
|----------|----------------------|:--------------:|:----------------:|:----------------:|
| `supply_chain_token_drift` | — | — | — | — |
| `federated_identity_takeover` | — | — | — | — |
| `insider_repo_pivot` | — | — | — | — |
| `cloud_metadata_ssrf` *(held-out)* | — | — | — | — |

### Reward channel ablation

| Ablation | Mean reward | Std dev | Exploit present? |
|----------|:-----------:|:-------:|:-----------------:|
| Baseline (capped disruption) | — | 0.0 | Yes |
| + Linear disruption penalty | — | — | — |
| + Evidence containment bonus | — | — | — |
| + Dispersion signals (full) | — | — | — |

---

## 6. Evaluation tasks & grading

The trained defender is evaluated along three axes (see notebook metrics and `run_episode` diagnostics):

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                           │
├──────────────────┬──────────────────────┬───────────────────────┤
│  DETECTION       │  CONTAINMENT         │  SURVIVAL               │
│                  │                      │                         │
│  Identify and    │  Block in-progress   │  Prevent final          │
│  confirm         │  lateral movement    │  exfiltration stage.    │
│  compromised     │  or exfiltration     │                         │
│  targets.        │  attempts.           │  Terminal penalty on    │
│                  │                      │  failure; bonus on      │
│  True-positive   │  Stages prevented.   │  clean resolution.      │
│  style signals.  │                      │                         │
└──────────────────┴──────────────────────┴─────────────────────────┘
```

---

## 7. Notebooks

The **[`notebooks/`](notebooks/)** directory is meant to hold **two** Jupyter flows:

| Notebook | Role |
|----------|------|
| **[`notebooks/cybersec_grpo.ipynb`](notebooks/cybersec_grpo.ipynb)** | **Full pipeline:** install/runtime, baselines, Unsloth load, outer-loop GRPO, base vs trained LLM eval, plots, optional **strict canary** asserts. Same algorithm as [`scripts/train_cybersec_grpo.py`](scripts/train_cybersec_grpo.py). [Open in Colab](https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb). |
| **`notebooks/cybersec_grpo_post_train.ipynb`** *(recommended snapshot name)* | **Post-training only:** duplicate the main notebook after a successful run, then **delete or skip** cells through training; keep artifact paths, adapter load, eval, and plots so you can re-run evaluation **without** retraining. **Add this file when you freeze a post-train copy** — it is not committed by default so the repo does not drift from your last artifact layout. |

**Headless training** (no notebook): from repo root,

```bash
python scripts/train_cybersec_grpo.py --output-dir ./_artifacts
```

---

## 8. Quickstart

**Install and run tests:**

```bash
pip install -e ./cybersec[dev]
pytest -q
```

**Install training dependencies** (match your CUDA / Unsloth install instructions):

```bash
pip install -e "./cybersec[grpo]"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

**Outputs (training script or notebook):**

| Path | Contents |
|------|----------|
| `qwen_cybersec_lora/` (or `ARTIFACTS` / `_artifacts`) | LoRA adapter + tokenizer |
| `training_log.json` | GRPO log history |
| `run_manifest.json` | Run configuration |
| `grpo_checkpoints_outer*/` | Checkpoints per outer loop |

---

## 9. Running the server

**Local:**

```bash
cybersec-server
```

**Docker:**

```bash
docker build -t cybersec-env:latest -f cybersec/server/Dockerfile cybersec
docker run --rm -p 8000:8000 cybersec-env:latest
```

> **Hugging Face Spaces:** the served app uses **`base_path: /web`** (see `cybersec/README.md` frontmatter). Routes are **`/web/reset`**, **`/web/step`**, **`/web/ws`**, etc.

---

<div align="center">

**License:** MIT — see [`cybersec/pyproject.toml`](cybersec/pyproject.toml)

<br />

*Built to show that small, local, privacy-preserving models can match the reasoning of senior SOC analysts — tick by tick.*

</div>
