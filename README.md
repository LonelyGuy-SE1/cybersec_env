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

[**▶ Play the Live Environment**](https://huggingface.co/spaces/Lonelyguyse1/cybersec) &nbsp;|&nbsp; [**📝 Design Blog**](BLOG.md) &nbsp;|&nbsp; [**🧪 Training Notebook**](notebooks/cybersec_grpo.ipynb)

</div>

---

## Table of Contents

1. [Why I Built This](#1-why-i-built-this)
2. [The Small Model Manifesto](#2-the-small-model-manifesto)
3. [Environment: Technical Specs](#3-environment-technical-specs)
   - [Scenarios](#scenarios)
   - [The Adversary (Scripted)](#the-adversary-scripted)
   - [The Defender & Action Space](#the-defender-the-llm--action-space)
   - [Observation & Action Schema](#observation--action-schema)
4. [Training Methodology](#4-training-methodology)
   - [Reward Model (7 Channels)](#the-reward-model-7-channels)
   - [The Disruption Exploit & The Fix](#the-disruption-exploit--the-fix)
5. [Results](#5-results)
6. [Evaluation Tasks & Grading](#6-evaluation-tasks--grading)
7. [Quickstart](#7-quickstart)
8. [Running the Server](#8-running-the-server)

---

## 1. Why I Built This

> *The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of your own.*

Thanks to the rise of powerful SOTA models, almost anyone now has access to a tool of immense intelligence. This is incredible — but it also arms bad actors. I have watched this play out firsthand with recent attacks on major internet infrastructure, where absolute amateurs, purely harnessing AI, penetrate systems that had been considered hardened for years. Projects like Mythos have found 0-days in established codebases with millions of lines of code.

Existing cyber environments treat the problem as a **static, point-in-time classification task** ("is this payload bad?"). Real attacks don't work that way. Attackers plan, gather intelligence, wait, and pivot across systems over days or weeks.

**Cybersec OpenEnv** was built to simulate the real *fog of war* inside a Security Operations Center:

- ⏳ **Long-horizon attacks** — staged, multi-step campaigns that unfold over many environment ticks
- 📡 **Delayed telemetry** — early signals are deliberately weak and noisy
- 💸 **Real business costs** — taking systems offline has consequences, modeled explicitly

Training an RL agent on a live production network is risky and expensive. This environment makes that training safe, fast, and reproducible.

---

## 2. The Small Model Manifesto

The defender is a **`Qwen2.5-1.5B-Instruct`** model — not a 70B behemoth. Here's why that's a feature, not a limitation:

| Constraint | Why It Matters |
|---|---|
| 🚀 **Iteration Speed** | Multiple RL training cycles fit on a single Colab T4 GPU |
| 🔒 **Privacy & Security** | No sensitive network telemetry ever leaves your infrastructure to an external API |
| 🌐 **Edge Deployment** | 70B models aren't viable for mass enterprise adoption; this is built for the real world |

The goal: prove that small, open-source LLMs can be fine-tuned to reason like **risk-aware senior SOC analysts**.

---

## 3. Environment: Technical Specs

Cybersec OpenEnv is a **long-horizon, partially observable Markov decision process (POMDP)** in which two agents — a scripted adversary and an LLM defender — interact within a simulated enterprise network.

The core challenge: **staged attacks unfold over many ticks with stochastic timing, and early compromise signals can be extremely weak.**

HTTP examples, `/web` routes, and the Hugging Face **Space card** for this package live in **[`cybersec/README.md`](cybersec/README.md)**.

### Scenarios

Three training scenarios and one held-out evaluation scenario, all grounded in real **MITRE ATT&CK** techniques:

**Training**

| ID | Campaign | Stages | Horizon |
|---|---|:---:|:---:|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (Evaluation Only)**

| ID | Campaign | Stages | Horizon |
|---|---|:---:|:---:|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → role-chain → KMS replicate → cloud storage exfil | 5 | 70 |

---

### The Adversary (Scripted)

The attacker walks a **deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG)**. Each node represents a sequential attack stage the adversary must complete to reach exfiltration. To simulate realism, the attacker is assigned one of three personalities:

| Personality | Dwell Time | Detection Risk | Pauses After Defender | Reroutes? |
|---|:---:|:---:|:---:|:---:|
| `stealthy` | 1.5× slower | 0.55× lower | 50% chance | ✗ |
| `aggressive` | 0.6× faster | 1.30× higher | Never | ✗ |
| `opportunistic` | Baseline | Baseline | 15% chance | ✓ |

---

### The Defender (The LLM) & Action Space

Each tick, the LLM receives a **partial observation** — lagged alerts, forensic results from past `INVESTIGATE` calls, containment state, and **`valid_targets`** as `{"assets": [...], "identities": [...]}` — and must emit a single structured JSON action.

| Action | Target | Effect |
|---|---|---|
| `MONITOR` | *(none)* | Low-cost; advances time and accumulates new observations |
| `INVESTIGATE` | asset / identity | Returns a noisy forensic signal on the target |
| `ISOLATE_ASSET` | asset | Quarantines the asset; can interrupt in-progress attack stages |
| `REVOKE_IDENTITY` | identity | Revokes credentials; blocks identity-based pivots |
| `BLOCK_EGRESS` | asset | Containment focused on preventing data exfiltration |
| `PATCH_ASSET` | asset | One-shot hardening; lowers future stage success probability |

> ⚠️ Invalid or out-of-set targets **still consume a tick** and incur an `invalid_action_penalty`.

---

### Observation & Action Schema

<details>
<summary><strong>▶ Expand: Example observation (tick 4 of <code>federated_identity_takeover</code>)</strong></summary>

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
<summary><strong>▶ Expand: Example action response</strong></summary>

```json
{"action_type": "INVESTIGATE", "target": "u-platform-eng"}
```

After a forensic row is confirmed, the model might follow with:

```json
{"action_type": "REVOKE_IDENTITY", "target": "u-platform-eng"}
```

</details>

**HTTP / OpenEnv Web UI:** actions are posted as `{"action": {"action_type": "...", "target": "..."}}` to **`POST …/web/step`**. Full curl examples and scenario IDs in [`cybersec/README.md`](cybersec/README.md).

**Reset parameters:** pass `scenario_id`, `seed` (int, for reproducibility), and optionally `attacker_personality` to **`POST …/web/reset`**. If **`scenario_id`** is omitted, the server uses **`CYBERSEC_SCENARIO_ID`** when set; otherwise the scenario index comes from the episode RNG. If **`seed`** is omitted, a **fresh random seed** is drawn each reset (so `{}` is not stuck on one scenario). See `cybersec/server/cybersec_environment.py` and `cybersec/server/app.py`.

---

## 4. Training Methodology

Training uses **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL and Unsloth (4-bit QLoRA), running end-to-end on a single T4 GPU.

### The Reward Model (7 Channels)

Rather than sparse 0/1 win/loss feedback, the environment provides a **dense 7-channel reward signal** that shapes every tick of behavior:

| Channel | Signal | Purpose |
|---|:---:|---|
| `detection` | ✅ + | Reward for first confirmed compromise of an attacker target |
| `containment` | ✅ +/− | Net reward for active and preemptive containment, minus action cost |
| `evidence_bonus` | ✅ + | Extra reward for containing targets **already confirmed** via `INVESTIGATE` |
| `false_positive_penalty` | ❌ − | Penalty for containment actions on non-attack-path targets |
| `disruption_penalty` | ❌ − | Operational cost of isolations and egress blocks |
| `invalid_action_penalty` | ❌ − | Penalty for illegal actions (bad target, wrong verb type) |
| `terminal_score` | ✅/❌ ± | Episode outcome: clean resolution bonus vs. exfiltration penalty |

---

### The Disruption Exploit & The Fix

> **The model found a degenerate cheat code.**

During initial GRPO training, the policy achieved a high average return — but the **standard deviation of the returns across the generated batches was exactly `0.0`**.

The LLM had discovered that isolating every asset on the network at **Tick 0** — completely nuking the entire business — was the mathematically optimal play under the original reward function. The disruption penalty had a hard per-tick cap, making mass isolation cheap at scale.

*"I solved the hack by unplugging the internet."*

**Three fixes, applied together:**

1. **Remove the disruption cap** — penalty now scales linearly with the number of isolated assets, making mass isolation self-defeating
2. **Add the evidence-based containment bonus** — the model receives a **+1.5** bonus for containing a target it has *already confirmed compromised* via an `INVESTIGATE` action
3. **Switch to on-policy iterative self-play** — no more static offline datasets

**The training loop:**

```
Outer Loop 0 (Warmup)
└── Heuristic policy generates 1,500 rows of seed data
└── GRPO optimizer steps: model learns JSON schema and basic mechanics

Outer Loop 1+ (True RL)
└── The LLM takes the wheel at temperature 1.4
└── Generates 1,500 rows by playing the environment with its own weights
└── Makes mistakes, explores new states, endures false positive penalties
└── Learns surgical defense to find genuine mathematical advantage
```

To prevent **mode collapse** into repetitive actions, three additional dispersion signals are added to the GRPO reward function:

- **`reward_action_diversity`** — penalises uniform outputs within a candidate group
- **`reward_observation_aware`** — rewards state-conditioned responses to active alerts
- **`reward_evidence_containment`** — dense proxy for the evidence bonus; reinforces the investigate-then-contain workflow

---

## 5. Results

> 📊 *Training curves, per-scenario breakdowns, and ablation tables will appear here.*

### Training Performance

<!-- INSERT: Reward curve over GRPO outer loops (total reward, detection rate, false positive rate) -->
<!-- Suggested format: single wide image, e.g. assets/reward_curve.png -->

| Metric | Outer Loop 0 (Heuristic) | Outer Loop 1+ (Self-Play) |
|---|:---:|:---:|
| Mean Episode Reward | — | — |
| Detection Rate | — | — |
| False Positive Rate | — | — |
| Exfiltration Prevented | — | — |

---

### Per-Scenario Breakdown

<!-- INSERT: Bar chart or table comparing defender win rate across all 4 scenarios -->
<!-- Include held-out scenario (cloud_metadata_ssrf) to show OOD generalisation -->

| Scenario | Attacker Personality | Stages Stopped | Exfil Prevented | Terminal Outcome |
|---|---|:---:|:---:|:---:|
| `supply_chain_token_drift` | — | — | — | — |
| `federated_identity_takeover` | — | — | — | — |
| `insider_repo_pivot` | — | — | — | — |
| `cloud_metadata_ssrf` *(held-out)* | — | — | — | — |

---

### Reward Channel Ablation

<!-- INSERT: Ablation showing impact of each reward fix (disruption cap removal, evidence bonus, dispersion signals) -->
<!-- Suggested format: grouped bar chart or delta table vs. baseline -->

| Ablation | Mean Reward | Std Dev | Exploit Present? |
|---|:---:|:---:|:---:|
| Baseline (capped disruption) | — | 0.0 | ✓ |
| + Linear disruption penalty | — | — | — |
| + Evidence containment bonus | — | — | — |
| + Dispersion signals (full) | — | — | — |

---

## 6. Evaluation Tasks & Grading

The trained defender is evaluated across three programmatic task axes:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                         │
├──────────────────┬──────────────────────┬───────────────────────┤
│  DETECTION       │  CONTAINMENT         │  SURVIVAL             │
│                  │                      │                       │
│  Accurately      │  Block in-progress   │  Prevent final        │
│  identify and    │  lateral movement    │  exfiltration stage.  │
│  confirm         │  or exfiltration     │                       │
│  compromised     │  attempts.           │  Massive terminal     │
│  targets.        │                      │  penalty on failure;  │
│                  │  Scored by attack    │  survival bonus on    │
│  Scored by true  │  stages prevented.   │  network preservation.│
│  positive rate.  │                      │                       │
└──────────────────┴──────────────────────┴───────────────────────┘
```

---

## 7. Quickstart

**Install and run tests:**

```bash
pip install -e ./cybersec[dev]
pytest -q
```

**Install training dependencies** (match your CUDA version):

```bash
pip install -e "./cybersec[grpo]"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

**Run iterative on-policy training:**

```bash
python scripts/train_cybersec_grpo.py --output-dir ./_artifacts
```

**Outputs:**

| Path | Contents |
|---|---|
| `qwen_cybersec_lora/` | Final LoRA adapter weights |
| `training_log.json` | Per-step reward metrics |
| `run_manifest.json` | Full run configuration |
| `grpo_checkpoints_outer*/` | Per-outer-loop checkpoints |

### Notebooks (`notebooks/`)

There are **two** intended Jupyter flows:

| Notebook | What it is |
|---|---|
| **[`cybersec_grpo.ipynb`](notebooks/cybersec_grpo.ipynb)** | **Full pipeline** — install/runtime, baselines, Unsloth, outer-loop GRPO, base vs trained eval, plots, optional strict canaries. Same algorithm as `scripts/train_cybersec_grpo.py`. [Open in Colab](https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb). |
| **`cybersec_grpo_post_train.ipynb`** *(you add this)* | **Post-training snapshot** — duplicate the main notebook after a good run, strip training cells, keep adapter load + eval + plots so you can rerun evaluation without retracing training. |

---

## 8. Running the Server

**Local:**

```bash
cybersec-server
```

**Docker:**

```bash
docker build -t cybersec-env:latest -f cybersec/server/Dockerfile cybersec
docker run --rm -p 8000:8000 cybersec-env:latest
```

> **Hugging Face Spaces note:** the packaged app mounts at `base_path: /web`, so all OpenEnv routes are prefixed — e.g. `/web/reset`, `/web/step`, `/web/ws`. Full curl-style examples in [`cybersec/README.md`](cybersec/README.md).

---

<div align="center">

**License:** MIT — see `cybersec/pyproject.toml`

<br />

*Built to prove that small, local, privacy-preserving models*  
*can match the reasoning of senior SOC analysts — tick by tick.*

</div>
