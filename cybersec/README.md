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

# Cybersec OpenEnv: Long-Horizon Multi-Agent Cyber Defense

A clean, production-shaped [OpenEnv](https://github.com/meta-pytorch/openenv-core)
environment for training LLM defenders against scripted attackers walking real
[MITRE ATT&CK](https://attack.mitre.org/) kill chains.

> **TL;DR** — Three MITRE-aligned enterprise breach scenarios, three attacker
> personalities, a partially observable defender with six actions, and a six-channel
> reward designed for stable LLM-RL on a single consumer GPU (Colab T4 / Kaggle P100 /
> a few dollars of HF credits).

This README is the canonical spec for the environment. It also doubles as the
Hugging Face Spaces card when the env is pushed via `openenv push`.

---

## 1. Why this environment

The judging axis we are aiming at is **long-horizon planning + multi-agent
interaction**. Cybersecurity is a saturated topic, but it is uniquely a good
fit for that axis:

* **Real attackers don't push one button.** They steal a credential at tick
  *t*, sit on it, escalate at *t + d₁*, pivot at *t + d₂*, exfiltrate at
  *t + d₃*. Every pair (*d₁*, *d₂*, *d₃*) is stochastic and the early signals
  are quiet. This is what makes the task a *planning* problem rather than a
  classification one.
* **There are two agents in the room.** A scripted attacker with a chosen
  personality drives ground truth; a controllable defender (the agent we
  train) reads partial observations and chooses one of six actions per tick.
  The attacker is intentionally pluggable — see §7.
* **The defender has an honest information disadvantage.** Detection is
  lagged, alerts are noisy, the chain forks across multiple assets and
  identities. The defender must learn when to investigate and when to
  contain — and when *not* to act.

We deliberately keep the action and reward surfaces small so a 1.5B-parameter
Qwen can both fit a useful prompt in 2k tokens and produce structured JSON
actions reliably.

---

## 2. The world

Each episode runs **one scenario** with **one attacker personality**, sampled
from the seed (or pinned via reset kwargs).

### Scenarios (`scenarios.py`)

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

Every stage is tagged with a MITRE tactic + technique (e.g. `T1552.004 Private Keys`,
`T1621 MFA Request Generation`, `T1041 Exfiltration Over C2 Channel`).

### Attacker personalities (`attacker.py`)

The same DAG behaves very differently under each personality:

| Personality     | Dwell × | Detection × | Pause-after-defender | Reroutes |
|---|---|---|---|---|
| `stealthy`      | 1.5×    | 0.55×       | 50%                 | no       |
| `aggressive`    | 0.6×    | 1.30×       | 0%                  | no       |
| `opportunistic` | 1.0×    | 1.0×        | 15%                 | yes      |

This is the entire multi-agent surface. The attacker is scripted on purpose —
we want the defender's reward to be cleanly attributable, and we want each
personality to demand a measurably different defender strategy. An optional
`LLMAttacker` adapter (§7) can be hot-swapped for evaluation runs.

### Defender contract (`models.py`)

All actions take at most a single string `target`:

```json
{"action_type": "ISOLATE_ASSET", "target": "payments-svc"}
```

| Action            | Target           | What it does                                       |
|---|---|---|
| `MONITOR`         | (none)           | Cheap — advances time, observes new alerts.        |
| `INVESTIGATE`     | asset / identity | Returns a noisy ForensicResult on the target.      |
| `ISOLATE_ASSET`   | asset            | Quarantines an asset; stops in-progress stages.    |
| `REVOKE_IDENTITY` | identity         | Burns credentials; stops identity-pivot stages.    |
| `BLOCK_EGRESS`    | asset            | Special-purpose containment for exfiltration.      |
| `PATCH_ASSET`     | asset            | One-shot harden; reduces stage success probability.|

The observation gives the policy everything it needs to be hard-constrained
(list of `available_actions`, `valid_targets["assets" | "identities"]`,
plus the current containment state of every control). Invalid actions still
cost a tick and accrue an `invalid_action_penalty`.

### Reward (`reward.py`)

Six channels, all visible in `obs.info["reward_breakdown"]` every step:

| Channel                    | Sign | When it fires                                                    |
|---|---|---|
| `detection`               | +    | First confirmed compromise of a real attacker target           |
| `containment`             | +/-  | Active block (full credit) + preemptive block (small) − action cost |
| `false_positive_penalty`  | −    | Containment action on a non-attack-path target                 |
| `disruption_penalty`      | −    | Per-tick cost of running with isolations / egress blocks active |
| `invalid_action_penalty`  | −    | Action whose target is missing from `valid_targets`            |
| `terminal_score`          | ±    | Big bonus for a clean episode, big penalty if exfil happens   |

The per-step total is clipped to ±5 (terminal added separately) so a single
rare burst can't dominate training. Detection and active containment only
fire on the *first* qualifying event per target, which kills the obvious
spam-isolate reward hack.

---

## 3. Project layout

This folder follows the official `openenv init` scaffold exactly — the env
folder *is* the package, with the FastAPI server in a `server/` subpackage.

```
cybersec/                              # the OpenEnv env folder (package root)
├── README.md                          # this file (also the HF Spaces card)
├── openenv.yaml                       # OpenEnv space manifest, app: server.app:app
├── pyproject.toml                     # packaging + dev/notebook extras
├── __init__.py                        # public surface re-exports
├── client.py                          # CybersecEnv (WebSocket client)
├── models.py                          # CybersecAction / Observation / State
├── scenarios.py                       # 3 MITRE-aligned scenarios
├── attacker.py                        # ScriptedAttacker + 3 personalities
├── telemetry.py                       # Background noise + INVESTIGATE oracle
├── reward.py                          # 6-channel reward + terminal grader
├── baselines.py                       # Random + Heuristic + run_episode
├── py.typed
└── server/
    ├── __init__.py
    ├── cybersec_environment.py        # CybersecEnvironment subclass
    ├── app.py                         # FastAPI app via openenv create_app()
    ├── Dockerfile                     # container image (HF Spaces build context)
    └── requirements.txt               # server-only runtime pins
```

The repo also ships `tests/` and `notebooks/` *outside* this folder — those
are training infrastructure, not part of the env package itself.

---

## 4. Quickstart

### Install (run from `cybersec/`)

```bash
cd cybersec/
pip install -e .[dev]
```

### Run the unit tests (run from repo root)

```bash
pytest -q
```

### Drive an episode in-process

```python
from cybersec import CybersecEnvironment
from cybersec.baselines import HeuristicPolicy, run_episode

env = CybersecEnvironment()
result = run_episode(env, HeuristicPolicy(), seed=0)
print(result.cumulative_reward, result.terminal_reason)
```

### Run as an OpenEnv server

```bash
# console-script entry point installed by pip:
cybersec-server

# or directly with uvicorn from inside cybersec/:
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then drive it from any other machine:

```python
from cybersec import CybersecEnv, ActionType, CybersecAction

with CybersecEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=0)
    while not result.observation.done:
        # ... your policy here ...
        result = env.step(CybersecAction(action_type=ActionType.MONITOR))
```

### Run via Docker

```bash
cd cybersec/
docker build -t cybersec-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 cybersec-env:latest
```

### Push to Hugging Face Spaces

```bash
cd cybersec/
openenv push
```

`openenv push` reads `openenv.yaml` in the current directory, builds the
Docker image using `server/Dockerfile`, and uploads to a Spaces repo.

---

## 5. Baselines (50 episodes per scenario)

Random and Heuristic are the two reference policies in `baselines.py`. Numbers
below are from `pytest`-stable seeds 0-49 with `attacker_personality` sampled
by RNG (so each cell averages over all three personalities):

| Scenario                       | Random — return | Heuristic — return | Heuristic — exfil rate |
|---|---|---|---|
| `supply_chain_token_drift`     |  1.93           |  **2.20**          | 10%                    |
| `federated_identity_takeover`  |  0.59           |  **2.55**          | 4%                     |
| `insider_repo_pivot`           |  **3.09**       |  2.30              | 8%                     |

Random's "win" on the insider scenario is exactly the kind of artifact we
*want* a good reward function to reveal: random wins by burning the network
down (~5 isolations in the first 8 ticks), and the heuristic refuses to do so
because waiting for evidence is more legible defender behavior. The trained
LLM defender's job is to land between them — surgical, evidence-driven
containment.

---

## 6. Training story

The notebooks in `../notebooks/` walk through the canonical RLVR training arc:

1. **`01_baseline_eval.ipynb`** — runs Random, Heuristic, and (optionally) a
   frozen `Qwen2.5-1.5B-Instruct` defender for 50 seeds × 3 scenarios. Plots
   reward curves and dumps `_artifacts/baseline_metrics.json` plus a JSONL of
   every heuristic trajectory for SFT warm-up.
2. **`02_grpo_train.ipynb`** — Unsloth-loaded Qwen2.5-1.5B-Instruct, 4-bit
   QLoRA, then TRL GRPO using **six independent reward functions** (JSON
   validity, schema validity, target-in-valid_targets, no-redundant-containment,
   actual env step reward via cloned snapshot, exfil-path bonus). Designed
   to fit Colab T4 / Kaggle P100 / one HF GPU dollar.
3. **`03_post_train_eval.ipynb`** — re-runs the *exact same seeds* from
   notebook 1 against the trained policy and produces before/after plots on
   identical axes. The judge reads this notebook last.

The core trick: every reward channel is already a separately-scored, dense
signal, so GRPO can use them as independent reward functions instead of
relying on a single scalar. That's what makes the gradient stable on a 1.5B
model with batch size ≪ episode horizon.

---

## 7. Design notes (read these before you tweak anything)

* **The action space is six verbs on purpose.** Earlier iterations had eleven
  actions with a target/parameter/urgency triple per action. Small LLMs
  reliably emitted invalid JSON or wrong target types ~30% of the time. With
  six verbs and a single optional `target`, structural validation rate
  approaches 100% even on Qwen2.5-0.5B in our smoke runs.
* **Reward channels split into `active` vs `preemptive` containment.** This
  is what kills the "isolate every asset on tick 1" reward hack that random
  exploits. Active interruption pays full freight (2.5/stage); preemption
  pays a fraction (0.4/stage) and incurs a per-action cost.
* **Investigation has no FP penalty.** Investigating clean targets is the
  defender's main exploration signal, and we don't want to discourage it.
  The opportunity cost (a whole tick) is friction enough.
* **The attacker only has one stage in_progress at a time.** That keeps the
  long-horizon planning challenge legible — the defender is always racing
  one specific dwell timer — and makes credit assignment to defender actions
  a clean transition rather than a credit-assignment puzzle.
* **The attacker is pluggable, not static.** `attacker.py` ships
  `ScriptedAttacker` (fast, deterministic, used in training) but the
  environment talks to it through the `DefenderView` / `AttackerEvent`
  contract, so you can drop in an `LLMAttacker` for richer evaluations
  without re-wiring the training loop.
* **`SUPPORTS_CONCURRENT_SESSIONS = True`.** Each WebSocket connection gets
  its own world; nothing is shared. Stress-tested at 8 concurrent sessions
  via `CYBERSEC_MAX_CONCURRENT_ENVS`.

---

## 8. References

* [OpenEnv core](https://github.com/meta-pytorch/openenv-core)
* [MITRE ATT&CK Enterprise Matrix](https://attack.mitre.org/matrices/enterprise/)
* [TRL — GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)
* [Unsloth](https://github.com/unslothai/unsloth)

## 9. License

MIT.
