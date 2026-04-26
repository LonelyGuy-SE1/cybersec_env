# Cybersec OpenEnv: Teaching Small LLMs Surgical Cyber-Defense

▶️ **Play the Environment:** [huggingface.co/spaces/Lonelyguyse1/cybersec](https://huggingface.co/spaces/Lonelyguyse1/cybersec)

---

## 1. The Origin Story: Why I Built This

Thanks to the rise of powerful SOTA models like Mythos, Opus, and rapidly improving open-source frontier models, almost anyone has gained access to a tool of immense intelligence. This brings incredible benefits—accelerating research, building applications faster, and decreasing workloads. But it also arms bad actors. 

I am already seeing this play out with recent attacks on major internet infrastructure. Some of these attackers are absolute amateurs, purely harnessing AI to penetrate systems. Software I thought was secure for years has been broken open (e.g., Project Glassdoor / Mythos finding 0-days in established codebases with millions of lines of code). The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of my own.

But to train that guard, I need a realistic environment. Training an RL agent on a live production network is incredibly risky and expensive. Existing cyber environments often lack an adaptive, multi-agent adversarial attacker, treating cybersecurity as a static point-in-time classification task ("is this payload bad?"). I built **Cybersec OpenEnv** to simulate the real "fog of war" in a Security Operations Center (SOC): long-horizon attacks, delayed telemetry, and the very real business cost of taking systems offline.

## 2. The Small Model Manifesto

Why use a tiny `Qwen2.5-1.5B-Instruct` model instead of a 70B parameter behemoth?

1. **Iteration Speed & Compute:** Real-world systems run on constraints. To get multiple iterations of RL training and testing done, a large model is simply too slow and compute-heavy for a single consumer GPU (like a Colab T4).
2. **Privacy & Security:** When it comes to enterprise security, you do not want to take chances by streaming your internal network topology and sensitive telemetry to an external API provider. It is fundamentally safer to use local, air-gapped models.
3. **Edge Deployment:** Loading a 70B model locally isn't viable for mass adoption in standard enterprise environments. If I want autonomous SOC agents deployed everywhere, I have to make this work with smaller, hyper-efficient models.

## 3. The Environment: Technical Specs

This is a long-horizon, partially observable Markov decision process (POMDP) where two agents interact within an enterprise network topology.

### The Scenarios
The environment includes three training scenarios and one held-out scenario for out-of-distribution evaluation, all based on real MITRE ATT&CK techniques:
* **Training:** `supply_chain_token_drift` (5 stages, 70 ticks), `federated_identity_takeover` (5 stages, 70 ticks), `insider_repo_pivot` (6 stages, 80 ticks).
* **Held-out:** `cloud_metadata_ssrf` (4 stages, 60 ticks).

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG) for the given scenario. To simulate realism, the attacker is assigned one of three personalities:
* **AGGRESSIVE:** Simulates an amateur. Fast, loud, brute-force approach (0.6x dwell time). Generates high alerts (1.3x detection) but progresses quickly.
* **STEALTHY:** Simulates an experienced APT. Higher success rate, but heavily penalized for generating noise (0.55x detection). The defender must proactively hunt them (1.5x dwell time).
* **OPPORTUNISTIC:** A balanced, middle-ground approach that can reroute paths based on network defenses.

### The Defender (The LLM) & Action Space
The LLM reads partial observations (lagged alerts, investigation forensics, valid targets) and chooses one of six structured actions per tick:
1. `MONITOR`: Low cost; advances time and observations. Used to wait for telemetry.
2. `INVESTIGATE`: Target an asset/identity. Provides a noisy, probabilistic forensic confidence score of compromise.
3. `ISOLATE_ASSET`: Quarantine a server; can interrupt in-progress attack stages.
4. `REVOKE_IDENTITY`: Revoke user credentials; blocks identity pivots.
5. `BLOCK_EGRESS`: Containment oriented specifically to stopping exfiltration.
6. `PATCH_ASSET`: One-shot hardening; lowers the attacker's stage success odds.

### A Typical Interaction
```json
// Observation (Tick 4)
{
  "tick": 4,
  "recent_alerts": [
    {"signal": "Suspicious login attempt", "target": "u-vp-sales", "severity": "high"}
  ],
  "valid_targets": ["u-vp-sales", "idp-okta", "vpn-gw", "sales-crm"]
}

// LLM Action
{"action_type": "INVESTIGATE", "target": "u-vp-sales"}

// Observation (Tick 5)
{
  "tick": 5,
  "recent_forensics": [
    {"target": "u-vp-sales", "is_compromised": true, "confidence": 0.85}
  ]
}

// LLM Action
{"action_type": "REVOKE_IDENTITY", "target": "u-vp-sales"}
```

## 4. Training Methodology

I used **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL and Unsloth (4-bit QLoRA) to train the LLM. 

### The Reward Model (6 Channels)
Instead of a sparse 0/1 win/loss reward, the environment provides a dense 6-channel reward signal:
* `detection (+)`: First confirmed compromise of an attacker target via `INVESTIGATE`.
* `containment (+/-)`: Active and preemptive containment, net of action cost.
* `false_positive_penalty (-)`: Containment on non-attack-path targets.
* `disruption_penalty (-)`: The operational cost of isolations and egress blocks.
* `invalid_action_penalty (-)`: Structurally invalid JSON actions.
* `terminal_score (+/-)`: Episode outcome (survival bonus vs. exfiltration penalty).

### The Struggle: The Disruption Exploit
During my initial GRPO training loop, the model achieved a massive score, but I noticed the standard deviation of the returns across the batch was exactly `0.0`. 

The LLM had found a degenerate cheat code. Because the "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** I removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. I also threw out the static offline dataset and restructured the training into **On-Policy Iterative Self-Play Loops**:
* **Outer Loop 0 (The Warmup):** The environment runs a deterministic Heuristic Policy to generate an initial 1,500 rows of training data. The model performs its first batch of GRPO optimizer steps on this dataset just to learn the JSON schema and basic mechanics.
* **Outer Loop 1+ (True RL):** The LLM takes the wheel. It generates a fresh 1,500 rows of data by playing the environment itself using its own weights at a high temperature (`1.2`). Because it makes mistakes and explores new states, the model is forced to actually read the telemetry, endure the pain of false positive penalties, and learn surgical defense in order to find a true mathematical advantage.

## 5. Tasks & Grading

The environment evaluates the defender across three programmatic tasks:
* **Detection Task:** Accurately identifying and confirming compromised targets. Scored by true positive confirmations.
* **Containment Task:** Successfully blocking an attacker's in-progress lateral movement or exfiltration. Scored by the number of attack stages prevented.
* **Survival Task:** Preventing the attacker from completing the exfiltration stage. Scored via a massive terminal penalty if exfiltration occurs, or a survival bonus if the network is preserved.

## 6. The Vibe & Takeaway

The potential of multi-agent, long-horizon adaptive systems is massive. An attacker rarely strikes all at once; they plan, gather info, wait, and pivot. By successfully training a 1.5B parameter model to navigate this environment without burning down the network, I've proven that small, open-source LLMs can be fine-tuned to reason like senior, risk-aware SOC analysts. 

---

## Install and train

```bash
pip install -e ./cybersec[dev]
pytest -q
```

**Dependencies** (from repo root; match your CUDA):

```bash
pip install -e "./cybersec[grpo]"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

**Run the iterative on-policy training script**

```bash
python scripts/train_cybersec_grpo.py --output-dir ./_artifacts
```

Outputs: `qwen_cybersec_lora/`, `training_log.json`, `run_manifest.json`, per-outer checkpoints under `grpo_checkpoints_outer*`.

**Notebook (Colab / Jupyter)** — same algorithm, baselines, plots, eval: [notebooks/cybersec_grpo.ipynb](notebooks/cybersec_grpo.ipynb).

---

## Server

```bash
cybersec-server
```

Docker: `docker build -t cybersec-env:latest -f cybersec/server/Dockerfile cybersec` then `docker run --rm -p 8000:8000 cybersec-env:latest`.

---

## License
MIT — see `cybersec/pyproject.toml`.