# Cybersec OpenEnv — next-step playbook

**Audience:** an LLM agent (or human operator) picking up where the
previous Sonnet/Opus session left off. You do **not** need the prior chat
history — everything you need is in this file plus the repo. Scan section
0 for a TL;DR, then walk the sections in order.

---

## 0. TL;DR

* The env is a **long-horizon multi-agent cybersec OpenEnv** with three
  training scenarios and one **held-out OOD scenario**. The HF Space is
  live at <https://lonelyguyse1-cybersec.hf.space> (`Lonelyguyse1/cybersec`).
* Iter-1 trained Qwen2.5-1.5B-Instruct with GRPO and **mode-collapsed**
  (`std_return = 0` on 2/3 scenarios). The cause: a single canned plan
  was high-EV under the original 6-reward signal mix.
* Iter-2 (this commit) ships:
  * **Two anti-collapse rewards** in `cybersec/training/rewards.py`
    (`reward_action_diversity`, `reward_observation_aware`).
  * **A 4th held-out scenario** (`cloud_metadata_ssrf`) for OOD reporting.
  * **Reward-hack canary tests** (`tests/test_reward_hack_canaries.py`).
  * **A unified notebook** (`notebooks/cybersec_grpo.ipynb`) that
    installs from the HF Space (no GitHub clone), suppresses warning
    spam, plots KL/loss/per-component reward, runs OOD eval, and asserts
    against iter-1's `std_return == 0` failure mode.
* **Recommended compute path: Colab Pro ($9.99 / 100 compute units).**
  HF $30 credits stay parked for keeping the Space awake on a small
  CPU upgrade so judges always see a live env. See section 5.

---

## 1. Repo map (only the parts you need)

```
cybersec_env/
├── cybersec/                         # the OpenEnv package (pyproject.toml at root)
│   ├── __init__.py                   # public surface (CybersecEnv, scenarios, ...)
│   ├── models.py                     # Pydantic Action / Observation / State
│   ├── scenarios.py                  # 4 scenarios; train + heldout split here
│   ├── attacker.py                   # scripted attacker w/ stochastic dwell times
│   ├── reward.py                     # six-channel env reward model
│   ├── telemetry.py                  # alert + forensic event log
│   ├── baselines.py                  # RandomPolicy, HeuristicPolicy, run_episode
│   ├── client.py                     # CybersecEnv (OpenEnv WebSocket client)
│   ├── server/
│   │   ├── app.py                    # FastAPI / OpenEnv server entry point
│   │   └── cybersec_environment.py   # CybersecEnvironment (the MDP)
│   ├── training/                     # NEW iter-2: TRL-shaped reward funcs
│   │   ├── __init__.py
│   │   └── rewards.py                # 8 reward funcs + helpers
│   ├── pyproject.toml                # packaged distribution metadata
│   ├── Dockerfile, openenv.yaml      # HF Space deployment
│   └── README.md                     # the canonical / Space card README
├── notebooks/
│   └── cybersec_grpo.ipynb           # ← THE notebook (39 cells)
├── tests/
│   ├── conftest.py
│   ├── test_action_validation.py
│   ├── test_env_contract.py
│   ├── test_scenarios.py
│   ├── test_training_rewards.py      # NEW iter-2
│   └── test_reward_hack_canaries.py  # NEW iter-2 (the alarm)
├── .scripts/                         # operator helpers (NOT shipped to the Space)
│   ├── build_unified_notebook.py     # author the notebook in plain Python
│   ├── push_bypass.py                # `openenv push` w/o whoami rate limit
│   ├── inspect_results.py            # parse a Colab-produced .ipynb
│   ├── smoke_notebook.py             # exec non-GPU cells locally
│   └── dump_cells.py / index_cells.py
├── README.md                         # 1-screen pointer; the real README is cybersec/README.md
└── NEXT_STEPS.md                     # this file
```

---

## 2. Pre-flight (10 min, no GPU, on your laptop)

Before you touch any GPU compute, check that the local env is still
healthy:

```bash
# from the repo root
python -m pytest tests/ -q
```

You must see **70 tests passing**. Anything else: stop, read the failure,
fix the regression. Do not proceed.

Then verify the HF Space is up:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('https://lonelyguyse1-cybersec.hf.space/health', timeout=10).read())"
```

You should see `{"status":"healthy"}`. If you don't, see section 6.

Finally, smoke-test the non-GPU notebook cells:

```bash
python .scripts/smoke_notebook.py
```

It should finish in ~45s and print "scenarios baseline ran on : [4 names]".

---

## 3. Run the iter-2 training

### 3.1 Open the notebook on Colab Pro (recommended)

1. Push current `main` to GitHub.
2. Open <https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb>.
3. Runtime → Change runtime type → **T4 GPU** (or A100 if available).
4. Optional sanity tweak: in cell 4 (`Imports & config`), set
   `MODE["use_remote_env"] = True` if you want to also smoke-test the
   live HF Space env from inside Colab. Off by default to keep
   Run-All hermetic.
5. **Run all** (Runtime → Run all). Total wall clock ~45–60 min.

### 3.2 What to look at while it runs

| Cell | Watch for                                                              |
| ---- | ---------------------------------------------------------------------- |
| 6    | Heuristic > random on at least 2/3 train scenarios. (We assert later.) |
| 12   | Dataset size = 1500. If much smaller, `max_dataset_rows` is wrong.     |
| 20   | GRPO trainer must START -- "Training" line, not a stack trace.         |
| 24   | KL/reward plots. If KL is rising > 5 in 100 steps -> instability.      |
| 27   | First trained-llm number per scenario. **All-zero std == bad.**        |
| 36   | The sanity asserts. Either all pass or you have a debug session.      |

### 3.3 What "good" looks like

* `summary_table.md` shows `std_return > 0.1` on at least one **training**
  scenario for `trained-llm`.
* `before_after_curves.png` shows the trained policy at-or-above heuristic
  on ≥ 1 train scenario, and at least within 5 points of heuristic on the
  others.
* `_artifacts/training_diagnostics.png` shows `reward_action_diversity`
  > `1 / num_generations` (i.e. > 0.25) for at least the first 30 steps.
  If it pegs at 0.25 immediately and stays there, the diversity reward is
  not biting hard enough — see section 4.3.
* The held-out scenario (`cloud_metadata_ssrf`) does **not** have to be
  a win, but `mean_return` should not be > 5 points below random; if it
  is, the training distribution is too narrow.

---

## 4. Debugging table (failure mode → fix)

### 4.1 GRPO trainer crashes immediately

* **Symptom:** `ImportError` in cell 20 about `trl.GRPOTrainer`.
* **Fix:** TRL 0.12+ moved things around. Bump in cell 2:
  `!pip install -q --upgrade "trl>=0.12"`. Re-run from cell 5.

* **Symptom:** `AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'`
  in cell 27 (NOT 20). This is iter-1's eval-time crash; should be fixed
  already. If it comes back, cell 26 must use
  `FastLanguageModel.from_pretrained(ADAPTER_DIR, ...)`, **not** vanilla
  `transformers.AutoModelForCausalLM`. Check the cell source.

### 4.2 Mode collapse (`std_return == 0`) re-appears

This is the iter-1 pathology. The iter-2 sanity assert in cell 36 catches
it. Steps:

1. Open `_artifacts/training_diagnostics.png`. If `reward_action_diversity`
   is pinned at `1/num_generations`, the policy collapsed. If KL is also
   near 0, the model never moved — bump LR (`5e-6` → `1e-5` in cell 20).
2. Increase `MODE["grpo_num_generations"]` from 4 to 6 or 8 — more
   candidates per prompt → more diverse gradients.
3. Add a stronger entropy term: in cell 20's `GRPOConfig`, set
   `temperature=1.2` and add `beta=0.05` (KL coefficient).
4. As a last resort, *down*-weight `reward_step_total` by wrapping it:
   ```python
   def reward_step_total_scaled(*a, **kw):
       return [0.5 * x for x in reward_step_total(*a, **kw)]
   ```
   The env reward is the strongest collapse pressure; halving it lets
   the diversity reward dominate early.

### 4.3 Held-out scenario tanks

* `mean_return` on `cloud_metadata_ssrf` more than 5 points below random.
* Likely cause: the policy memorised per-scenario asset names. Mitigation:
  in cell 12, also include 3–5 episodes of `cloud_metadata_ssrf` in the
  *prompt* (just the rendered observations) but NOT in the
  reward-bearing dataset rows — that is, expand the prompt distribution
  without expanding the training distribution. (This is a manual experiment;
  not in the default notebook.)

### 4.4 The HF Space serves but `/health` returns 5xx

See section 6.

---

## 5. Compute decision (Colab Pro vs HF Jobs vs Spaces upgrade)

You said "out of Colab GPU runtime; should we use HF credits or upgrade
Colab?" Recommendation:

**Pick Colab Pro for training, use HF credits to keep the Space alive.**

Why:

| Option                          | Cost (USD)    | Pro                                                   | Con                                                       |
| ------------------------------- | ------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| **Colab Pro**                   | $9.99/mo      | Same notebook works as-is, T4 + occasional A100.       | Monthly subscription.                                     |
| Colab free + wait 24h           | $0            | Free.                                                 | Quota cooldowns mid-run; you've already hit this once.    |
| **HF Spaces "CPU upgrade"**     | ~$0.03/hr     | Keeps the env Space *always-on* (judges always see live env). | Doesn't help training (CPU only).                |
| HF Spaces with T4 ($)           | ~$0.40/hr     | Could host inference of the trained model.            | Doesn't run a Jupyter kernel.                              |
| HF Inference Endpoints          | ~$0.50–$1/hr  | Can host the trained adapter for cheap inference.     | Doesn't run training, doesn't connect to a Colab kernel.  |
| HF Jobs (`run_uv`)              | ~$1–$1.50/hr  | Runs scripts on HF GPUs from your machine.            | No interactive Jupyter; you run `train.py`, then `eval.py`. |
| Lambda / Runpod A10/A100        | ~$0.50–$1/hr  | Real GPU, attachable to Colab via custom kernel.      | New tooling to learn; $30 HF credits don't apply.         |

**Concrete plan to spend the $30 HF credits + cheap subscription:**

1. **Spend $9.99 → Colab Pro.** Use it to run `cybersec_grpo.ipynb` end-to-end
   on a T4 (or A100 if you get lucky on Pro+). 100 compute units = ~50h of T4.
   That is enough for ~25 full notebook runs.
2. **Spend ~$5–10 of HF credits → CPU upgrade on the Space.** Settings →
   Spaces hardware → "CPU upgrade" or "small CPU + persistent storage".
   This keeps the env from sleeping mid-judge-evaluation.
3. **Optionally, spend the remaining $15–20 of HF credits → HF Inference
   Endpoint hosting the trained Qwen-1.5B-LoRA**. Push the merged adapter
   to a private model repo, point an Inference Endpoint at it, then
   include the endpoint URL in the README so judges can hit your trained
   defender from the browser. This is the "look how production-ready
   this is" demo bonus.

You do *not* need to upgrade Colab Pro+ ($49.99). Pro is sufficient.

### 5.1 If you want to run training without Colab at all

Use **HF Jobs**:

```bash
pip install huggingface_hub
hf jobs run --help            # CLI was renamed in late 2025
# OR python:
from huggingface_hub import run_uv
run_uv(
    "python notebooks/train_headless.py",
    flavor="t4-medium",
    secrets=["HF_TOKEN"],
)
```

You'd port the GRPO cells into a standalone `train_headless.py` script.
This is more work but doesn't need Colab. It is the "all-in on HF" path
if the judges care about it. It is not the default path.

---

## 6. Operating the HF Space

**URL:** <https://huggingface.co/spaces/Lonelyguyse1/cybersec>
**Live env:** <https://lonelyguyse1-cybersec.hf.space>

### 6.1 Push code changes to the Space

Use `python .scripts/push_bypass.py` (NOT `openenv push`). The bypass
script reuses openenv's staging logic but skips the `whoami-v2` API call
(which gets rate-limited after 10–15 pushes/day) and uses
`HfApi.upload_folder(delete_patterns=["*"])` to atomically replace the
Space contents. Without `delete_patterns`, stale files from older
revisions can break imports — this is what bit iter-1.

```bash
python .scripts/push_bypass.py     # uses HF_TOKEN from env or `~/.cache/huggingface`
```

### 6.2 The Space says "Runtime error"

1. Click the **Logs** tab on the Space page. Look for an `ImportError`.
2. The most likely cause is a broken relative import. The fix pattern is
   already in `cybersec/server/cybersec_environment.py`,
   `cybersec/attacker.py`, `cybersec/reward.py`, and `cybersec/telemetry.py`:

   ```python
   try:
       from ..models import ...   # relative -- works on dev machines
   except ImportError:
       from cybersec.models import ...   # absolute -- works on HF runtime
   ```

   If a new file you add fails to import on the Space, copy that
   try/except pattern.
3. Push again with `python .scripts/push_bypass.py`.
4. Watch the "Building" indicator turn green (~2 min for a CPU Space).

### 6.3 The Space sleeps

Free Spaces sleep after 48h of no traffic. Either:

* Enable a small CPU upgrade ($0.03/hr) to keep it permanently awake, or
* Hit `/health` from a daily cron (cheap-but-hacky).

---

## 7. Stretch goals (in priority order)

You don't have to do these — they are how you turn a working iter-2
into a polished hackathon submission.

### 7.1 LLM-driven attacker (high impact, ~half day)

Currently the attacker is scripted (see `cybersec/attacker.py`). Replace
the action selection at each tick with a tiny prompt to a remote LLM
(GPT-4o-mini, $0.30 per million tokens — total cost for 50 episodes ×
70 ticks ≈ $0.20). Then the env story becomes "two LLMs in a
long-horizon adversarial dance" instead of "one LLM vs a script". This
is the **headline novelty** of the submission.

Concrete patch sketch:

```python
# cybersec/attacker.py
class LLMAttacker(ScriptedAttacker):
    def __init__(self, *args, model="gpt-4o-mini", **kw):
        super().__init__(*args, **kw)
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model

    def _pick_next_stage(self, ctx):
        # build a 1-message prompt describing remaining stages + telemetry
        # call self._client.chat.completions.create(...)
        # parse JSON; fall back to ScriptedAttacker._pick_next_stage on parse error
        ...
```

Keep the scripted attacker as the default; the LLM one is opt-in via
a constructor flag. Tests stay deterministic this way.

### 7.2 Per-stage credit shaping (medium, ~2h)

Currently the env reward in `cybersec/reward.py` aggregates over
"compromise" / "containment" / "disruption" / "stealth" / "false_positive"
/ "terminal" channels. A long-horizon planner benefits from knowing
**which of the 5 stages of the current scenario** their action just
helped/hurt. Add a sixth `stage_progress` channel that returns
`+0.5 / -0.5` when a specific stage transitions to/from `success`. This
gives GRPO a denser long-horizon signal.

### 7.3 LLM-judged "explanation" reward (high impact, ~half day)

Add an optional reward that asks GPT-4o-mini "did this action plausibly
defend against the attacker's likely next move?". Costs ~$0.10 per 100
training steps. Adds a "qualitative" reward channel that the canned-plan
hack cannot trivially exploit, because the judge sees the *observation*.

### 7.4 Public README screenshots (low cost, high judge-ROI)

After a successful training run, screenshot:

* The HF Space card showing "Running" status.
* `_artifacts/before_after_curves.png` next to `_artifacts/training_diagnostics.png`.
* The notebook's headline-delta print-out.

Add them to `cybersec/README.md` under a "Results" section. Judges scan
images, not paragraphs.

---

## 8. What the previous (Sonnet/Opus) session left in flight

If anything in this list is checked off when you start, skip it.

* [x] Move 8 reward functions into `cybersec/training/rewards.py`.
* [x] Add 4th held-out scenario `cloud_metadata_ssrf`.
* [x] Add `tests/test_training_rewards.py` (21 tests).
* [x] Add `tests/test_reward_hack_canaries.py` (8 tests).
* [x] Rewrite the unified notebook to use the HF Space as install source,
      suppress warnings, plot KL/loss/components, eval the held-out
      scenario, and assert against `std_return == 0`.
* [x] Update `.scripts/build_unified_notebook.py` to author the new layout.
* [x] Update `cybersec/__init__.py` to export
      `list_train_scenarios`, `list_eval_scenarios`.
* [x] Update `cybersec/pyproject.toml` to package `cybersec.training`.
* [ ] **Push the iter-2 code to the HF Space.** Run
      `python .scripts/push_bypass.py` from the repo root once you're
      confident in this commit. The Space currently has iter-1 code
      and the new notebook's HF Space install will pull stale code
      until you push.
* [ ] **Run cybersec_grpo.ipynb on Colab Pro** (or a free-tier slot
      after cooldown) and record results in `_artifacts/`.
* [ ] **If the run produces good results** (mode collapse broken,
      headline delta positive on at least one train scenario), enable
      `PUSH_ARTIFACTS = True` in cell 38 to upload `_artifacts/` to the
      Space, and add screenshots to `cybersec/README.md`.

---

## 9. When you're stuck and want to escalate back to Sonnet/Opus

Bundle the following into one prompt for the next premium-model session:

1. The exact error or unexpected output. Include the cell number and
   the last 30 lines of output.
2. `_artifacts/training_log.json` (last 20 entries) and
   `_artifacts/training_diagnostics.png`.
3. `_artifacts/summary_table.md`.
4. `git log --oneline -10` to show what changed since iter-2.
5. The output of `python -m pytest tests/ -q`.

That gives the next session the entire state of the experiment without
re-reading the prior conversation.

---

*Last updated: end of iter-2 (Sonnet 4.5 session, just before model rotation).
The previous session's transcript is at the path the IDE reported as
`agent-transcripts/<uuid>.jsonl` — only consult it if this file is
ambiguous, since it is very large.*
