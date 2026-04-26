# Cybersec OpenEnv: Teaching Small LLMs Surgical Cyber-Defense

▶️ **Play the Environment:** [huggingface.co/spaces/Lonelyguyse1/cybersec](https://huggingface.co/spaces/Lonelyguyse1/cybersec)

---

## 1. The Origin Story: Why We Built This

Thanks to the rise of powerful SOTA models like Mythos, Opus, and rapidly improving open-source frontier models, almost anyone has gained access to a tool of immense intelligence. This brings incredible benefits—accelerating research, building applications faster, and decreasing workloads. But it also arms bad actors. 

We are already seeing this play out with recent attacks on major internet infrastructure. Some of these attackers are absolute amateurs, purely harnessing AI to penetrate systems. Software we thought was secure for years has been broken open (e.g., Project Glassdoor / Mythos finding 0-days in established codebases with millions of lines of code). The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of our own.

But to train that guard, we need a realistic environment. Training an RL agent on a live production network is incredibly risky and expensive. Existing cyber environments often lack an adaptive, multi-agent adversarial attacker, treating cybersecurity as a static classification task ("is this payload bad?"). We built this OpenEnv environment to simulate the real "fog of war" in a Security Operations Center (SOC): long-horizon attacks, delayed telemetry, and the very real business cost of taking systems offline.

## 2. The Small Model Manifesto

Why use a tiny `Qwen2.5-1.5B-Instruct` model instead of a 70B parameter behemoth?

1. **Iteration Speed:** Real-world systems run on constraints. To get multiple iterations of RL training and testing done to see what actually works, a large model is simply too slow and compute-heavy.
2. **Privacy & Security:** When it comes to enterprise security, you do not want to take chances by streaming your internal network topology and telemetry to an external API provider. It is fundamentally safer to use local models.
3. **Edge Deployment:** Loading a 70B model locally isn't viable for mass adoption in standard enterprise environments. If we want autonomous SOC agents deployed everywhere, we have to make this work with smaller, hyper-efficient models.

## 3. The Environment: Technical Specs

This is a long-horizon, partially observable Markov decision process (POMDP). There are two agents in the room:

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG). To simulate realism, the attacker is assigned one of three personalities:
* **AGGRESSIVE:** Simulates an amateur. Fast, loud, brute-force approach. Generates high alerts but progresses quickly.
* **STEALTHY:** Simulates an experienced APT. Higher success rate, but heavily penalized for generating noise. The defender must proactively hunt them.
* **OPPORTUNISTIC:** A balanced, middle-ground approach.

### The Defender (The LLM)
The LLM reads partial observations (lagged alerts, investigation forensics) and chooses one of six actions per tick:
1. `MONITOR`: Do nothing, advance the clock, wait for telemetry.
2. `INVESTIGATE`: Target an asset to get a probabilistic forensic confidence score of compromise.
3. `ISOLATE_ASSET`: Disconnect a server from the network (incurs a business disruption penalty).
4. `REVOKE_IDENTITY`: Kill a user's session/credentials.
5. `BLOCK_EGRESS`: Shut down outbound traffic pathways.
6. `PATCH_ASSET`: Apply a fix to a vulnerable node.

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

## 4. The Struggle: The Disruption Exploit

Training an RL agent is never a straight line. During our initial GRPO training loop, the model achieved a massive score, but we noticed the standard deviation of the returns was exactly `0.0`. 

The LLM had found a degenerate cheat code. 

Because our "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** We removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. We also threw out the heuristic offline dataset and forced the model into *On-Policy Iterative Self-Play*. The model was forced to actually read the telemetry, endure the pain of false positives, and learn surgical defense.

## 5. Tasks & Grading

The environment evaluates the defender across three programmatic tasks:
* **Detection Task:** Accurately identifying and confirming compromised targets using `INVESTIGATE`. Scored by true positive confirmations.
* **Containment Task:** Successfully blocking an attacker's in-progress lateral movement or exfiltration using containment actions. Scored by the number of attack stages prevented.
* **Survival Task:** Preventing the attacker from completing the exfiltration stage. Scored via a massive terminal penalty if exfiltration occurs, or a survival bonus if the network is preserved.

## 6. The Vibe

The potential of multi-agent, long-horizon adaptive systems is massive. An attacker rarely strikes all at once; they plan, gather info, wait, and pivot. By successfully training a 1.5B parameter model to navigate this environment without burning down the network, we've proven that small, open-source LLMs can be fine-tuned to reason like senior, risk-aware SOC analysts. 

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