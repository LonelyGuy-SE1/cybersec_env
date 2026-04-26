# Teaching a 1.5B LLM to be a SOC Analyst (Without Burning Down the Network)

Thanks to the rise of powerful SOTA models like Mythos, Opus, and rapidly improving open-source frontier models, almost anyone has gained access to a tool of immense intelligence. This brings incredible benefits—accelerating research, building applications faster, and decreasing workloads. But it also arms bad actors. 

We are already seeing this play out with recent attacks on major internet infrastructure. Some of these attackers are absolute amateurs, purely harnessing AI to penetrate systems. Software we thought was secure for years has been broken open (e.g., Project Glassdoor / Mythos finding 0-days in established codebases with millions of lines of code). 

The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of our own. But to train that guard, we need a realistic environment. Training an RL agent on a live production network is incredibly risky and expensive. Existing cyber environments often lack an adaptive, multi-agent adversarial attacker, treating cybersecurity as a static point-in-time classification task ("is this payload bad?"). 

We built **Cybersec OpenEnv** to simulate the real "fog of war" in a Security Operations Center (SOC): long-horizon attacks, delayed telemetry, and the very real business cost of taking systems offline.

## The Small Model Manifesto

We chose to train a tiny `Qwen2.5-1.5B-Instruct` model via GRPO (Group Relative Policy Optimization) instead of a 70B parameter behemoth for three key reasons:

1. **Iteration Speed & Compute:** Real-world systems run on constraints. To get multiple iterations of RL training and testing done to see what actually works, a large model is simply too slow and compute-heavy. We trained this entire system using Unsloth 4-bit QLoRA on a single consumer GPU.
2. **Privacy & Security:** When it comes to enterprise security, you do not want to take chances by streaming your internal network topology and sensitive telemetry to an external API provider. It is fundamentally safer to use local, air-gapped models.
3. **Edge Deployment:** Loading a 70B model locally isn't viable for mass adoption in standard enterprise environments. If we want autonomous SOC agents deployed everywhere, we have to make this work with smaller, hyper-efficient models.

## The Environment: A Multi-Agent POMDP

This is a long-horizon, partially observable Markov decision process (POMDP) featuring 4 highly detailed scenarios (3 for training, 1 held-out for out-of-distribution evaluation). There are two agents in the room.

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG), pulling off attacks like `supply_chain_token_drift` and `federated_identity_takeover`. To simulate realism, the attacker is assigned one of three personalities:
* **AGGRESSIVE:** Simulates an amateur. Fast, loud, brute-force approach. Generates high alerts but progresses quickly.
* **STEALTHY:** Simulates an experienced APT. Higher success rate, but heavily penalized for generating noise. The defender must proactively hunt them using investigations.
* **OPPORTUNISTIC:** A balanced, middle-ground approach that dynamically reroutes paths based on defenses.

### The Defender (The LLM)
The LLM reads partial observations (lagged alerts, investigation forensics) and chooses one of six structured actions per tick: 
1. `MONITOR`: Advance the clock and wait for telemetry.
2. `INVESTIGATE`: Target an asset for a forensic confidence score.
3. `ISOLATE_ASSET`: Disconnect a server (incurs a heavy disruption penalty).
4. `REVOKE_IDENTITY`: Kill user credentials.
5. `BLOCK_EGRESS`: Shut down outbound traffic.
6. `PATCH_ASSET`: Harden a vulnerable node.

A typical interaction forces the LLM to ingest noisy data, decide to investigate, and then act upon the findings:

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

## The Struggle: The Disruption Exploit

Training an RL agent is never a straight line. During our initial GRPO training loop, the model achieved a massive score, but we noticed the standard deviation of the returns across our generated batches was exactly `0.0`. 

The LLM had found a degenerate cheat code. 

Our environment utilizes a dense 6-channel reward system (Detection, Containment, False Positives, Disruption, Invalid Syntax, and Terminal Survival). Because our "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move for the LLM was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** We went back to the lab. We removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. We also restructured the training script into **On-Policy Iterative Self-Play Loops**:
* **Outer Loop 0 (The Warmup):** We generate a quick 1,500 rows of data using a hardcoded heuristic script. The LLM runs its first batch of optimizer steps on this dataset simply to learn the required JSON schema and basic valid actions.
* **Outer Loop 1+ (True RL):** This is where the magic happens. The LLM generates a brand new dataset by playing the environment itself. By generating rollouts using the LLM's own weights at a high temperature (`1.2`), the model explores strange new states the heuristic script never reached. It makes mistakes, gets penalized, and is forced to actually read the telemetry to learn surgical defense.

## The Takeaway

The potential of multi-agent, long-horizon adaptive systems is massive. An attacker rarely strikes all at once; they plan, gather info, wait, and pivot. By successfully training a 1.5B parameter model to navigate this environment without burning down the network, we've proven that small, open-source LLMs can be fine-tuned to reason like senior, risk-aware SOC analysts. 

Check out the environment and run it yourself on Hugging Face Spaces!