# Cybersec OpenEnv: Teaching Small LLMs Surgical Cyber-Defense

▶️ **Play the Environment:** [huggingface.co/spaces/Lonelyguyse1/cybersec](https://huggingface.co/spaces/Lonelyguyse1/cybersec) — HTTP examples and reset semantics: [`cybersec/README.md`](cybersec/README.md)

📝 **Blog (design narrative, reward exploits, fixes):** [BLOG.md](BLOG.md) — *Teaching a 1.5B LLM to be a SOC Analyst (Without Burning Down the Network)* · [on GitHub](https://github.com/LonelyGuy-SE1/cybersec_env/blob/main/BLOG.md)

🧪 **Training Notebook:** [notebooks/cybersec_grpo.ipynb](notebooks/cybersec_grpo.ipynb) — Colab-ready, runs end-to-end on a T4 GPU

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

This is a long-horizon, partially observable Markov decision process (POMDP) where two agents interact within an enterprise network topology. The environment emphasizes **long-horizon planning**: staged attacks (credential theft, dwell time, lateral pivots, exfiltration) unfold over many ticks with stochastic timing, and early signals can be incredibly weak.

### The Scenarios
The environment includes three training scenarios and one held-out scenario for out-of-distribution evaluation, all based on real MITRE ATT&CK techniques:

**Training Scenarios**
| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (Evaluation-only) Scenario**
| ID | Title | Stages | Horizon |
|---|---|---|---|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → role-chain → KMS replicate → cloud storage exfil | 5 | 70 |

### The Adversary (Scripted)
The attacker walks a deterministic MITRE ATT&CK-aligned Directed Acyclic Graph (DAG) for the given scenario. Each node in the DAG represents a sequential attack stage the adversary must complete to reach their exfiltration goal. To simulate realism, the attacker is assigned one of three personalities (its sample space):

| Personality     | Dwell × | Detection × | Pause-after-defender | Reroutes |
|---|---|---|---|---|
| `stealthy`      | 1.5×    | 0.55×       | 50%                 | no       |
| `aggressive`    | 0.6×    | 1.30×       | 0%                  | no       |
| `opportunistic` | 1.0×    | 1.0×        | 15%                 | yes      |

### The Defender (The LLM) & Action Space
The LLM reads partial observations (lagged **alerts**, **forensics** from past `INVESTIGATE` calls, containment state, and **`valid_targets`** as `{"assets": [...], "identities": [...]}`**) and chooses one structured **JSON** action per tick (training uses a single-line object; see `cybersec/training/rewards.py` `SYSTEM_PROMPT`).

| Action            | Target           | Effect |
|---|---|---|
| `MONITOR`         | (none)           | Low cost; advances time and observations. |
| `INVESTIGATE`     | asset / identity | Noisy forensic signal on the target. |
| `ISOLATE_ASSET`   | asset            | Quarantine; can interrupt in-progress stages. |
| `REVOKE_IDENTITY` | identity         | Revoke credentials; blocks identity pivots. |
| `BLOCK_EGRESS`    | asset            | Containment oriented to exfiltration. |
| `PATCH_ASSET`     | asset            | One-shot hardening; lowers stage success odds. |

**HTTP / OpenEnv web UI:** actions are posted as `{"action": {"action_type": "...", "target": "..."}}` under the Space **`/web`** prefix (e.g. `POST …/web/step`). Full examples and scenario IDs: [`cybersec/README.md`](cybersec/README.md) (also the Hugging Face Space card).

Illustrative shapes (field names match `cybersec/models.py`; alert `signal` values are enums such as `auth_anomaly`, `lateral_movement`, `egress_anomaly`):

```json
{
  "tick": 4,
  "horizon": 70,
  "scenario_id": "federated_identity_takeover",
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

```json
{"action_type": "INVESTIGATE", "target": "u-platform-eng"}
```

After a forensic row exists, the model might emit `REVOKE_IDENTITY` / `ISOLATE_ASSET` with a target that still appears under `valid_targets`. Invalid or out-of-set targets **still consume a tick** and draw `invalid_action_penalty` (see `cybersec/reward.py`).

### Reset & scenario selection
On **`reset`**, you may pass **`scenario_id`** (one of the four IDs above), **`seed`** (int, for reproducibility), and optionally **`attacker_personality`**. If **`scenario_id`** is omitted, the env picks a scenario from `list_scenarios()` using the episode RNG (with today’s server: omitting **`seed`** draws a fresh random seed each reset so HTTP clients are not pinned to a single scenario). Optional process env **`CYBERSEC_SCENARIO_ID`** on the server sets a default scenario when the client omits `scenario_id`. Details: `cybersec/server/cybersec_environment.py`, `cybersec/server/app.py`.

## 4. Training Methodology

I used **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL and Unsloth (4-bit QLoRA) to train the LLM. 

### The Reward Model (7 Channels)
Instead of a sparse 0/1 win/loss reward, the environment provides a dense 7-channel reward signal:

| Channel                    | Sign | Role |
|---|---|---|
| `detection`               | +    | First confirmed compromise of an attacker target |
| `containment`             | +/-  | Active and preemptive containment, net of action cost |
| `evidence_bonus`          | +    | Containment actions on targets previously confirmed via INVESTIGATE |
| `false_positive_penalty`  | −    | Containment on non-attack-path targets |
| `disruption_penalty`      | −    | Operational cost of isolations and egress blocks |
| `invalid_action_penalty`  | −    | Illegal actions (bad target, wrong type for verb, etc.); still consumes a tick |
| `terminal_score`          | ±    | Episode outcome: clean resolution vs exfiltration |

### The Struggle: The Disruption Exploit
During my initial GRPO training loop, the model achieved a massive score, but I noticed the standard deviation of the returns across the batch was exactly `0.0`. 

The LLM had found a degenerate cheat code. Because the "disruption penalty" (the cost of taking a business asset offline) had a hard cap per tick, the mathematically optimal move was to instantly `ISOLATE_ASSET` every single server in the company at Tick 0. The LLM essentially said, *"I solved the hack by unplugging the internet."*

**The Fix:** I removed the disruption cap, forcing a linearly scaling penalty for every isolated asset. I also added an **evidence-based containment bonus**: the model receives a +1.5 reward for containing targets it has *already confirmed compromised* via `INVESTIGATE`. This means "surgically investigate then contain" scores higher than "blindly isolate everything on tick 0." I also threw out the static offline dataset and restructured the training into **On-Policy Iterative Self-Play Loops**:
* **Outer Loop 0 (The Warmup):** The environment runs a deterministic Heuristic Policy to generate an initial 1,500 rows of training data. The model performs its first batch of GRPO optimizer steps on this dataset just to learn the JSON schema and basic mechanics.
* **Outer Loop 1+ (True RL):** The LLM takes the wheel. It generates a fresh 1,500 rows of data by playing the environment itself using its own weights at a high temperature (`1.4`). Because it makes mistakes and explores new states, the model is forced to actually read the telemetry, endure the pain of false positive penalties, and learn surgical defense in order to find a true mathematical advantage.

To prevent mode collapse into repetitive actions, the GRPO reward function includes three **dispersion signals**: `reward_action_diversity` (penalises uniform outputs within a candidate group), `reward_observation_aware` (rewards state-conditioned responses to alerts), and `reward_evidence_containment` (a dense proxy for the evidence bonus, reinforcing the investigate-then-contain workflow).

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

On **Hugging Face Spaces**, the packaged app is mounted at **`base_path: /web`**, so health and OpenEnv routes are under **`/web`** (e.g. `/web/reset`, `/web/step`, `/web/ws`). See [`cybersec/README.md`](cybersec/README.md) for curl-style examples.

---

## License
MIT — see `cybersec/pyproject.toml`.