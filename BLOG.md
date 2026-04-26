# Teaching a 1.5B LLM to be a SOC Analyst (Without Burning Down the Network)

Thanks to the rise of powerful SOTA models like Mythos, Opus, and rapidly improving open-source frontier models, almost anyone has gained access to a tool of immense intelligence. This brings incredible benefits—accelerating research, building applications faster, and decreasing workloads. But it also arms bad actors. 

I am already seeing this play out with recent attacks on major internet infrastructure. Some of these attackers are absolute amateurs, purely harnessing AI to penetrate systems. Software I thought was secure for years has been broken open (e.g., Project Glassdoor / Mythos finding 0-days in established codebases with millions of lines of code). 

The only way to protect a system in real-time against an AI-augmented threat is to deploy a 24/7 AI guard of my own. But to train that guard, I need a realistic environment. Training an RL agent on a live production network is incredibly risky and expensive. Existing cyber environments often lack an adaptive, multi-agent adversarial attacker, treating cybersecurity as a static point-in-time classification task ("is this payload bad?"). 

I built **Cybersec OpenEnv** to simulate the real "fog of war" in a Security Operations Center (SOC): long-horizon attacks, delayed telemetry, and the very real business cost of taking systems offline.

## The Small Model Manifesto

I chose to train a tiny `Qwen2.5-1.5B-Instruct` model via GRPO (Group Relative Policy Optimization) instead of a 70B parameter behemoth for three key reasons:

1. **Iteration Speed & Compute:** Real-world systems run on constraints. To get multiple iterations of RL training and testing done to see what actually works, a large model is simply too slow and compute-heavy. I trained this entire system using Unsloth 4-bit QLoRA on a single consumer GPU.
2. **Privacy & Security:** When it comes to enterprise security, you do not want to take chances by streaming your internal network topology and sensitive telemetry to an external API provider. It is fundamentally safer to use local, air-gapped models.
3. **Edge Deployment:** Loading a 70B model locally isn't viable for mass adoption in standard enterprise environments. If I want autonomous SOC agents deployed everywhere, I have to make this work with smaller, hyper-efficient models.

## The Environment: A Multi-Agent POMDP

This is a long-horizon, partially observable Markov decision process (POMDP) featuring 4 highly detailed scenarios (3 for training, 1 held-out for out-of-distribution evaluation). There are two agents in the room.

**Training Scenarios**
| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (Evaluation-only) Scenario**
| ID | Title | Stages | Horizon |
|---|---|---|---|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → assumed role → cloud storage exfil | 4 | 60 |

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG). To simulate realism, the attacker is assigned one of three personalities (its sample space):

| Personality     | Dwell × | Detection × | Pause-after-defender | Reroutes |
|---|---|---|---|---|
| `stealthy`      | 1.5×    | 0.55×       | 50%                 | no       |
| `aggressive`    | 0.6×    | 1.30×       | 0%                  | no       |
| `opportunistic` | 1.0×    | 1.0×        | 15%                 | yes      |

### The Defender (The LLM)
The LLM reads partial observations (lagged alerts, investigation forensics) and chooses one of six structured actions per tick:

| Action            | Target           | Effect |
|---|---|---|
| `MONITOR`         | (none)           | Low cost; advances time and observations. |
| `INVESTIGATE`     | asset / identity | Noisy forensic signal on the target. |
| `ISOLATE_ASSET`   | asset            | Quarantine; can interrupt in-progress stages. |
| `REVOKE_IDENTITY` | identity         | Revoke credentials; blocks identity pivots. |
| `BLOCK_EGRESS`    | asset            | Containment oriented to exfiltration. |
| `PATCH_ASSET`     | asset            | One-shot hardening; lowers stage success odds. |

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

Training an RL agent is never a straight line. During my initial GRPO training loop, the model achieved a massive score, but I noticed the standard deviation of the returns across the generated batches was exactly `0.0`. 

The LLM had found a degenerate cheat code. 

The environment utilizes a dense 6-channel reward system:

| Channel                    | Sign | Role |
|---|---|---|
| `detection`               | +    | First confirmed compromise of an attacker target |
| `containment`             | +/-  | Active and preemptive containment, net of action cost |
| `false_positive_penalty`  | −    | Containment on non-attack-path targets |
| `disruption_penalty`      | −    | Operational cost of isolations and egress blocks |
| `invalid_action_penalty`  | −    | Structurally invalid JSON actions |
| `terminal_score`          | ±    | Episode outcome: clean resolution vs exfiltration |

Because the "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move for the LLM was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** I removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. I also restructured the training script into **On-Policy Iterative Self-Play Loops**:
* **Outer Loop 0 (The Warmup):** The script generates a quick 1,500 rows of data using a hardcoded heuristic policy. The LLM runs its first batch of optimizer steps on this dataset simply to learn the required JSON schema and basic valid actions.
* **Outer Loop 1+ (True RL):** This is where the magic happens. The LLM generates a brand new dataset by playing the environment itself. By generating rollouts using the LLM's own weights at a high temperature (`1.2`), the model explores strange new states the heuristic script never reached. It makes mistakes, gets penalized, and is forced to actually read the telemetry to learn surgical defense.

## The Takeaway

The potential of multi-agent, long-horizon adaptive systems is massive. An attacker rarely strikes all at once; they plan, gather info, wait, and pivot. By successfully training a 1.5B parameter model to navigate this environment without burning down the network, I've proven that small, open-source LLMs can be fine-tuned to reason like senior, risk-aware SOC analysts. 

Check out the environment and run it yourself on Hugging Face Spaces!