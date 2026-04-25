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

**Training scenarios:**

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `supply_chain_token_drift` | CI-token theft → poisoned artifact → payments pivot → warehouse exfil | 5 | 70 |
| `federated_identity_takeover` | Spearphish → MFA fatigue → helpdesk pivot → HR portal → cloud-egress exfil | 5 | 70 |
| `insider_repo_pivot` | Repo recon → secret harvest → staging → prod cluster → DB exfil | 6 | 80 |

**Held-out (eval-only) scenario** — never seen during GRPO training; used to
measure whether the trained policy *generalises* or just memorises:

| ID | Title | Stages | Horizon |
|---|---|---|---|
| `cloud_metadata_ssrf` | SSRF → cloud metadata → assumed role → cloud storage exfil | 4 | 60 |

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
├── scenarios.py                       # 3 train + 1 held-out MITRE-aligned scenarios
├── attacker.py                        # ScriptedAttacker + 3 personalities
├── telemetry.py                       # Background noise + INVESTIGATE oracle
├── reward.py                          # 6-channel reward + terminal grader
├── baselines.py                       # Random + Heuristic + run_episode + EpisodeResult
├── py.typed
├── server/
│   ├── __init__.py
│   ├── cybersec_environment.py        # CybersecEnvironment subclass
│   ├── app.py                         # FastAPI app via openenv create_app()
│   ├── Dockerfile                     # container image (HF Spaces build context)
│   └── requirements.txt               # server-only runtime pins
└── training/
    ├── __init__.py
    └── rewards.py                     # 9 TRL-compatible reward funcs + parsers
```

The repo also ships `train.py`, `tests/`, and `notebooks/` *outside* this
folder — those are training/eval infrastructure, not part of the env package
itself. Judges installing the package via `pip install` only ever see the
`cybersec/` folder.

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

## 5. Baselines (30 episodes per scenario, against the live HF Space)

Random and Heuristic are the two reference policies in `baselines.py`. The
numbers below come from running 30 seeds per scenario over the deployed
HF Space WebSocket API (i.e. the same transport judges use):

| Scenario                       | Random — return | Heuristic — return | Heuristic — exfil rate |
|---|---|---|---|
| `supply_chain_token_drift`     |  3.34           |  **3.36**          | 13%                    |
| `federated_identity_takeover`  | -1.10           |  **2.93**          | 7%                     |
| `insider_repo_pivot`           |  **2.66**       | -2.69              | 7%                     |
| `cloud_metadata_ssrf` (OOD)    |  1.45           |  **2.55**          | 0%                     |

(Numbers above are the iter-3 measured baseline; they do not change in
iter-4 since the env itself didn't change. Reproducible via
`python train.py --baseline-only`.)

Random's "win" on the insider scenario is exactly the kind of artifact we
*want* a good reward function to reveal: random wins by burning the network
down (~5 isolations in the first 8 ticks), and the heuristic refuses to do so
because waiting for evidence is more legible defender behavior. The trained
LLM defender's job is to land between them — surgical, evidence-driven
containment that beats heuristic on the held-out scenario.

---

## 6. Training story

There are two equivalent UX surfaces. Both run the same pipeline,
write the same artifacts, and use the same reward functions —
choose whichever your reviewer prefers:

| Entry point | Best for |
|---|---|
| [`../train.py`](../train.py) | Headless GPU runs (`python train.py`). One command, end-to-end, fits in a tmux pane. |
| [`../notebooks/cybersec_grpo.ipynb`](../notebooks/cybersec_grpo.ipynb) | Click-through Colab / Kaggle. Same code, but Run-All shows every step. |

Both pull the env package straight from the deployed HF Space, so the
Space *is* the canonical source — no GitHub clone required. Roughly
**65-85 min** of wall clock from cold install to before/after plots on
a free Colab T4 / Kaggle P100, at the iter-4 defaults below.

**Pipeline:**

1. **Baseline** — `RandomPolicy` + `HeuristicPolicy` over 30 seeds × 3
   training scenarios + 1 held-out OOD scenario (`cloud_metadata_ssrf`,
   never seen during training). Writes `_artifacts/baseline_metrics.json`
   and the per-scenario reward-curve plot. Episodes go over the **OpenEnv
   WebSocket protocol** to the deployed HF Space — same transport a judge
   will use.
2. **Dataset build** (local) — ~1500 `(state-snapshot, prompt)` pairs
   harvested from heuristic rollouts on the **training scenarios only**.
   Each prompt is paired with a pickled `CybersecEnvironment`, so GRPO
   reward functions can clone the env and score candidate actions
   against the *real* environment dynamics. This step is local because
   `pickle` and a live WebSocket don't mix.
3. **GRPO training** — Unsloth-loaded Qwen2.5-1.5B-Instruct, 4-bit QLoRA,
   then TRL GRPO for **120 steps × 6 generations/prompt** at
   `temperature=1.2`, `beta=0.04`, `lr=3e-6` (iter-4 defaults), scored
   by **nine independent reward functions** in
   [`cybersec/training/rewards.py`](training/rewards.py):
   - 4 schema/validity rewards (JSON valid, schema valid, target in
     `valid_targets`, no redundant containment),
   - 1 actual env-step reward via cloned snapshot,
   - 1 exfil-path shaping prior,
   - 3 **anti-collapse rewards**: `reward_action_diversity`
     (within-group rarity), `reward_observation_aware` (state-conditioned
     behaviour), and **iter-4's `reward_batch_action_entropy`** — a
     batch-wide entropy bonus that gives a non-zero gradient *out* of
     full collapse, not just a barrier preventing entry to it.

   Saves the LoRA adapter to `_artifacts/qwen_cybersec_lora/`, per-step
   reward components to `_artifacts/training_log.json`, and
   KL/loss/per-component diagnostic plots to
   `_artifacts/training_diagnostics.png`.
4. **Trained-policy eval** — reloads the adapter via Unsloth (so the
   same fused-QKV attention patch used during training is present at
   inference) and re-runs the same 30 seeds × 3 train scenarios from
   step 1, then a separate **held-out OOD eval** on `cloud_metadata_ssrf`,
   all over the live HF Space. Writes `_artifacts/before_after_curves.png`,
   `_artifacts/summary_table.md`, `_artifacts/post_train_metrics.json`,
   and `_artifacts/heldout_metrics.json`.
5. **Sanity canaries** — the iter-4 gate before a run is shipped:
   - heuristic > random on aggregate baseline return,
   - trained-policy invalid-action rate ≤ 20%,
   - trained-policy never >5pts below random on any train scenario,
   - **≥ 2 of 3 train scenarios must have `std_return > 0.1`** (iter-3
     passed the previous "any one" version while still being collapsed
     on the other two),
   - **`monitor_fallback_rate` ≤ 50%** — catches the case where a
     "trained" policy is actually a wall of unparseable text saved by
     the `MONITOR` fallback in `llm_act`.

A single `MODE` dict (in both the notebook and `train.py`) controls
every dial. `train.py` also accepts CLI overrides:

```bash
python train.py --grpo-max-steps 200 --n-baseline-episodes 50
python train.py --baseline-only --mode local           # smoke test
python train.py --grpo-temperature 1.4 --grpo-beta 0.06 # nudge anti-collapse
```

The core trick: every reward channel is already a separately-scored
dense signal, so GRPO uses them as independent reward functions
instead of one scalar. That, plus iter-4's three anti-collapse rewards
and the bumped `num_generations`/`temperature`/`beta` defaults, is
what makes gradients stable on a 1.5B model with batch size ≪ horizon.
The canary suite at `tests/test_reward_hack_canaries.py` is the alarm
that catches future mode-collapse regressions before they ship.

### Iteration history

The hackathon submission is iter-4. Each iteration was driven by a
specific failure observed in the previous run:

| Iter | Change | Why |
|---|---|---|
| **iter-1** | One scalar reward, fixed seeds. | Trained policy "won" with `std_return = 0.0` on 2/3 scenarios — a memorised canned plan. |
| **iter-2** | Add `reward_action_diversity` + `reward_observation_aware`; held-out scenario `cloud_metadata_ssrf`. | Make mode collapse *visible* and shape against it. |
| **iter-3** | Move eval to OpenEnv WebSocket protocol; package as canonical HF Space; runtime detection. | Make the eval the judges actually run match the eval the developer runs. |
| **iter-4** | `reward_batch_action_entropy` + bumped `num_generations` (4→6) / `temperature` (1.0→1.2) / `beta` (0→0.04) / `lr` (5e-6→3e-6); `monitor_fallback_rate` metric; tightened std=0 canary to ≥2/3 scenarios. | Iter-3 still showed `std_return=0` on 2/3 train scenarios. The diversity reward is a *prevention* signal, not a recovery one — once everything is identical, every candidate scores the same. The new entropy reward gives a positive gradient out of collapse; bumped sampling makes that gradient findable. |

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
