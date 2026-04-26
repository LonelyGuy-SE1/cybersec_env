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

The environment is built around the concept of **long-horizon planning**: attacks are not instantaneous. Credential theft, dwell times, lateral pivots, and data exfiltration unfold over many ticks (up to 80) with stochastic timing. Because early alert signals can be incredibly weak, the LLM must learn to anticipate future states rather than just reacting to the immediate present.

**Training Scenarios**
| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (evaluation-only) scenario** (OOD generalisation; not in default GRPO `list_train_scenarios()`):

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → role-chain → KMS replicate → cloud storage exfil | 5 | 70 |

### Try it on Hugging Face Spaces

The live OpenEnv server is at **[huggingface.co/spaces/Lonelyguyse1/cybersec](https://huggingface.co/spaces/Lonelyguyse1/cybersec)**. The Space uses OpenEnv’s **`/web`** base path: `POST …/web/reset` and `POST …/web/step` with JSON bodies. Example actions wrap the defender move as `{"action": {"action_type": "MONITOR"}}` or `{"action": {"action_type": "INVESTIGATE", "target": "<asset-or-identity>"}}` — targets must appear under the current observation’s `valid_targets.assets` / `valid_targets.identities`. Copy-paste examples, scenario IDs, and optional env vars (`CYBERSEC_SCENARIO_ID`, …) are in the repo’s **[`cybersec/README.md`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/README.md)** (same file powers the Space README card).

**Reset behaviour worth knowing:** if you omit **`scenario_id`**, the environment chooses among the four scenarios using the episode RNG. Omitting **`seed`** on a fresh reset uses a **new random seed** each time (so casual HTTP `/web/reset` with `{}` is not stuck on one scenario). Pass an explicit **`seed`** when you want a reproducible episode. The server can still pin a default scenario via **`CYBERSEC_SCENARIO_ID`** if that env var is set in the Space.

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG). Each node in the DAG is a sequential stage the attacker must complete to win. To simulate realism, the attacker is assigned one of three personalities (its sample space):

| Personality     | Dwell × | Detection × | Pause-after-defender | Reroutes |
|---|---|---|---|---|
| `stealthy`      | 1.5×    | 0.55×       | 50%                 | no       |
| `aggressive`    | 0.6×    | 1.30×       | 0%                  | no       |
| `opportunistic` | 1.0×    | 1.0×        | 15%                 | yes      |

### The Defender (The LLM)
The LLM reads **partial** observations: **alerts** (noisy, lagged; each has a coarse `signal` enum, `severity` in `[0,1]`, optional `asset` / `identity`, and a short `description`), **forensics** (past `INVESTIGATE` results with `is_compromised` and `confidence`), containment lists, and **`valid_targets` as a dict** `{"assets": [...], "identities": [...]}` (not a flat list). It emits **one JSON object per step** during training (`render_observation` + `SYSTEM_PROMPT` in `cybersec/training/rewards.py`).

| Action            | Target           | Effect |
|---|---|---|
| `MONITOR`         | (none)           | Low cost; advances time and observations. |
| `INVESTIGATE`     | asset / identity | Noisy forensic signal on the target. |
| `ISOLATE_ASSET`   | asset            | Quarantine; can interrupt in-progress stages. |
| `REVOKE_IDENTITY` | identity         | Revoke credentials; blocks identity pivots. |
| `BLOCK_EGRESS`    | asset            | Containment oriented to exfiltration. |
| `PATCH_ASSET`     | asset            | One-shot hardening; lowers stage success odds. |

Sketch aligned with the real **`CybersecObservation`** / **`CybersecAction`** schema (scenario-specific names differ):

```json
{
  "tick": 4,
  "horizon": 70,
  "scenario_id": "supply_chain_token_drift",
  "alerts": [
    {
      "tick": 4,
      "signal": "lateral_movement",
      "asset": "artifact-registry",
      "identity": null,
      "severity": 0.44,
      "description": "RDP session between non-paired hosts"
    }
  ],
  "forensics": [
    {
      "tick": 2,
      "target": "ci-runner-01",
      "target_kind": "asset",
      "is_compromised": false,
      "confidence": 0.68
    }
  ],
  "valid_targets": {
    "assets": ["ci-runner-01", "artifact-registry", "api-gateway"],
    "identities": ["svc-ci-deploy", "u-platform-eng"]
  }
}
```

```json
{"action_type": "INVESTIGATE", "target": "artifact-registry"}
```

Over the **HTTP** API, the same move is wrapped as:

```json
{"action": {"action_type": "INVESTIGATE", "target": "artifact-registry"}}
```

## The Struggle: The Disruption Exploit

Training an RL agent is never a straight line. During my initial GRPO training loop, the model achieved a massive score, but I noticed the standard deviation of the returns across the generated batches was exactly `0.0`. 

The LLM had found a degenerate cheat code. 

The environment utilizes a dense 7-channel reward system:

| Channel                    | Sign | Role |
|---|---|---|
| `detection`               | +    | First confirmed compromise of an attacker target |
| `containment`             | +/-  | Active and preemptive containment, net of action cost |
| `evidence_bonus`          | +    | Containment actions on targets previously confirmed via INVESTIGATE |
| `false_positive_penalty`  | −    | Containment on non-attack-path targets |
| `disruption_penalty`      | −    | Operational cost of isolations and egress blocks |
| `invalid_action_penalty`  | −    | Illegal defender moves (e.g. target not in `valid_targets`); tick still advances |
| `terminal_score`          | ±    | Episode outcome: clean resolution vs exfiltration |

Because the "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move for the LLM was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** I removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. I also restructured the training script into **On-Policy Iterative Self-Play Loops**:
* **Outer Loop 0 (The Warmup):** The script generates a quick 1,500 rows of data using a hardcoded heuristic policy. The LLM runs its first batch of optimizer steps on this dataset simply to learn the required JSON schema and basic valid actions.
* **Outer Loop 1+ (True RL):** This is where the magic happens. The LLM generates a brand new dataset by playing the environment itself. By generating rollouts using the LLM's own weights at a high temperature (`1.4`), the model explores strange new states the heuristic script never reached. It makes mistakes, gets penalized, and is forced to actually read the telemetry to learn surgical defense.

But one exploit led to another. Even after removing the disruption cap, the model found a new shortcut: on tick 0 (before any alerts had fired), it would immediately `ISOLATE_ASSET` the first valid target. Because the attack kill-chain was a linear DAG, blocking the root stage on tick 0 instantly terminated the attack with a perfect `terminal_clean_bonus`. The model never learned to investigate, never read alerts, never made multi-step decisions. It just said *"unplug the first server, collect the prize."*

**The Second Fix: Evidence-Based Containment.** I added a new reward channel: `evidence_bonus`. The model receives a +1.5 bonus for containing a target it has *already confirmed compromised* via an `INVESTIGATE` action. This means the optimal strategy is no longer "blindly isolate on tick 0" — it's "investigate first, confirm compromise, then surgically contain." The investigate-then-contain workflow scores higher than blind isolation, forcing the model to actually engage with the environment's telemetry.

On top of that, GRPO training uses **dispersion-style rewards** (action diversity, observation-aware completions, batch entropy) so the policy does not collapse to a single repeated action when group-relative updates would otherwise wash out the learning signal. The full loop—baselines, outer-loop data collection, plots, and **strict canary** checks—is in **[`notebooks/cybersec_grpo.ipynb`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb)** and the headless driver **`scripts/train_cybersec_grpo.py`**.

## The Takeaway

The potential of multi-agent, long-horizon adaptive systems is massive. An attacker rarely strikes all at once; they plan, gather info, wait, and pivot. By successfully training a 1.5B parameter model to navigate this environment without burning down the network, I've proven that small, open-source LLMs can be fine-tuned to reason like senior, risk-aware SOC analysts. 

**Play the env:** [Hugging Face Space](https://huggingface.co/spaces/Lonelyguyse1/cybersec) · **Repo & install:** [github.com/LonelyGuy-SE1/cybersec_env](https://github.com/LonelyGuy-SE1/cybersec_env) · **API cheat sheet:** [`cybersec/README.md`](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/cybersec/README.md)