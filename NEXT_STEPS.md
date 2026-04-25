# Cybersec OpenEnv — next-step playbook (iter-3)

**Audience:** an LLM agent picking up where the Sonnet/Opus session left
off. You do **not** need the prior chat history. Scan section 0 for a
TL;DR, then walk the sections in order. There's a separate
`USER_GUIDE.md` for the human supervisor — keep them in sync.

---

## 0. TL;DR

* The env is a **long-horizon multi-agent cybersec OpenEnv** with three
  training scenarios and one **held-out OOD scenario**. Live at
  <https://lonelyguyse1-cybersec.hf.space> (`Lonelyguyse1/cybersec`).
* Iter-1 mode-collapsed (`std_return = 0` on 2/3 scenarios).
  Iter-2 added anti-collapse rewards, OOD scenario, canary tests.
  **Iter-3 (this commit)** rewrote the notebook so baseline + post-train
  evaluation actually go over the OpenEnv WebSocket protocol against the
  deployed HF Space, the way a hackathon judge will see it.
* **Final submission scope** = the `cybersec/` directory only
  (`pyproject.toml`, `Dockerfile`, `openenv.yaml`, package source). The
  notebook is shipped as the single training script the judges run in
  their own Colab/Kaggle. `tests/`, `.scripts/`, `notebooks/`,
  `NEXT_STEPS.md`, `USER_GUIDE.md`, `_artifacts/` are developer-only and
  are stripped before submission. They are gitignored or fenced behind
  a clean `cybersec/`-only export.
* **Compute**: the user has no Colab Pro this week. Use **Kaggle GPU
  notebooks** + remaining **HF credits**. Notebook is runtime-aware
  (Colab vs Kaggle vs local) so swap is one-line. Switching back to
  Colab Pro is fine; nothing in the code changes.
* **Space hardware constraint**: judge-locked at **2 vCPU / 8 GB**. No
  CPU upgrade. The env occupies <200 MB even at 8 concurrent envs
  (verified by `.scripts/memory_probe.py`), so this is fine.

---

## 1. Repo map (only the parts you need)

```
cybersec_env/
├── cybersec/                         # the OpenEnv package -- THIS IS THE SUBMISSION
│   ├── __init__.py                   # public surface (CybersecEnv, scenarios, ...)
│   ├── __version__.py
│   ├── client.py                     # CybersecEnv (OpenEnv WebSocket client)
│   ├── models.py                     # Pydantic Action / Observation / State
│   ├── scenarios.py                  # 4 scenarios; train + heldout split
│   ├── attacker.py                   # scripted attacker w/ stochastic dwell times
│   ├── reward.py                     # six-channel env MDP reward
│   ├── telemetry.py                  # alert + forensic event log
│   ├── baselines.py                  # RandomPolicy, HeuristicPolicy, run_episode
│   ├── server/
│   │   ├── app.py                    # FastAPI / OpenEnv server entry point
│   │   ├── cybersec_environment.py   # CybersecEnvironment (the MDP)
│   │   ├── Dockerfile                # HF Space build
│   │   └── requirements.txt
│   ├── training/                     # NEW iter-2: TRL-shaped reward funcs
│   │   ├── __init__.py
│   │   └── rewards.py                # 8 reward funcs + helpers
│   ├── pyproject.toml                # packaged distribution metadata
│   ├── openenv.yaml, uv.lock         # OpenEnv scaffolding
│   ├── py.typed                      # PEP 561 marker
│   └── README.md                     # the canonical / Space card README
├── notebooks/
│   └── cybersec_grpo.ipynb           # ← THE notebook (41 cells, iter-3 remote-env)
├── tests/                            # 70+ pytest tests; not in submission
│   ├── conftest.py
│   ├── test_action_validation.py
│   ├── test_env_contract.py
│   ├── test_scenarios.py
│   ├── test_training_rewards.py
│   └── test_reward_hack_canaries.py  # iter-2 alarm
├── .scripts/                         # operator helpers; gitignored, not in submission
│   ├── build_unified_notebook.py     # author the notebook from plain Python
│   ├── push_bypass.py                # `openenv push` w/o whoami rate limit
│   ├── probe_space.py                # check live schema
│   ├── remote_smoke.py               # one-episode WS smoke test
│   ├── smoke_remote_baselines.py     # 6-episode remote baseline smoke
│   ├── memory_probe.py               # measure 8-concurrent-env footprint
│   └── inspect_results.py / dump_cells.py / index_cells.py / smoke_notebook.py
├── README.md                         # 1-screen pointer; the real README is cybersec/README.md
├── NEXT_STEPS.md                     # this file (for next agent)
└── USER_GUIDE.md                     # step-by-step guide for the human supervisor
```

**OpenEnv layout sanity check**: run

```bash
openenv validate cybersec
```

You must see `[OK] cybersec: Ready for multi-mode deployment`. The
package layout has been diffed against `openenv init reference_env` —
all six scaffold files (`__init__.py`, `client.py`, `models.py`,
`openenv.yaml`, `pyproject.toml`, `README.md`, `uv.lock`,
`server/{app.py, Dockerfile, requirements.txt, __init__.py,
*_environment.py}`) are present. `training/` is a regular Python
sub-package; it doesn't conflict with the scaffold.

---

## 2. Pre-flight (10 min, no GPU, on your laptop)

Before you touch any GPU compute, check that the local env is healthy:

```bash
python -m pytest tests/ -q
```

You should see **70+ tests passing**. Anything else: stop, read the
failure, fix the regression. Do not proceed.

Verify the HF Space is up:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('https://lonelyguyse1-cybersec.hf.space/health', timeout=10).read())"
# {"status":"healthy"}

python .scripts/probe_space.py
# ActionType.enum on the deployed Space:
#   - MONITOR
#   - INVESTIGATE
#   - ISOLATE_ASSET
#   - REVOKE_IDENTITY
#   - BLOCK_EGRESS
#   - PATCH_ASSET

python .scripts/remote_smoke.py
# OK  steps=10  return=0.300  latency=4.94s done=True
```

If `/health` is 5xx or schema actions are wrong, see section 6.

Memory footprint check (one-shot):

```bash
python .scripts/memory_probe.py
# baseline RSS: ~150 MB; +0 MB for 8 concurrent envs.
```

8 concurrent envs use < 1 MB beyond the Python interpreter, well
within the judge's 2 vCPU / 8 GB Space constraint.

---

## 3. Run the iter-3 training

### 3.1 Open the notebook

The notebook auto-detects Colab / Kaggle / local. The same Run-All
works on all three.

| Runtime          | How to open                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| **Kaggle GPU**   | Upload `notebooks/cybersec_grpo.ipynb` to a new Kaggle Notebook → set Accelerator = GPU P100 → Run All. |
| **Colab (free)** | <https://colab.research.google.com/github/LonelyGuy-SE1/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb> → T4 → Run All. |
| Local (rare)     | `jupyter lab notebooks/cybersec_grpo.ipynb`. You won't have a GPU; only the non-training cells will work. |

### 3.2 What happens at runtime

1. **Cells 1–3**: detect runtime, install `cybersec` package straight
   from the HF Space (the canonical source — no GitHub clone), install
   GRPO stack.
2. **Cells 4–5**: imports + `MODE` config + `probe_space()` health
   check that fails fast if the Space is asleep.
3. **Cells 6–9 — Baseline against the live HF Space**: 30 episodes/
   scenario × 4 scenarios × 2 policies = 240 episodes over WebSocket.
   ~28 min wall-clock. **This is the cell where the judge sees the
   OpenEnv API working.**
4. **Cells 10–14 — Local dataset build + GRPO training**: 1500-prompt
   dataset, 100 GRPO steps × 4 generations, 8 reward functions.
   ~25 min on T4.
5. **Cells 18–22 — Post-train eval against the live HF Space**:
   30 episodes × 4 scenarios × 1 policy = 120 episodes. The trained
   adapter is reloaded via Unsloth (must be Unsloth, not vanilla
   transformers, because of the fused-QKV LoRA) and connected through
   the same SyncEnvClient transport used in section 3. ~10 min.
6. **Cells 23–28**: before/after curves, summary table, sanity asserts.

### 3.3 What "good" looks like

* `summary_table.md`: `std_return > 0.1` on at least one **training**
  scenario for `trained-llm`. (Iter-1 was 0 everywhere — that's the
  collapse signal.)
* `before_after_curves.png`: trained policy at-or-above heuristic on
  ≥ 1 train scenario, within 5 points on the others.
* `training_diagnostics.png`: `reward_action_diversity` > `1 /
  num_generations` (i.e. > 0.25) for at least the first 30 steps.
* Held-out (`cloud_metadata_ssrf`): not required to win, but
  `mean_return` should not be > 5 points below random.

---

## 4. Debugging table (failure mode → fix)

### 4.1 GRPO trainer crashes immediately

* **Symptom:** `ImportError` on `trl.GRPOTrainer`.
  **Fix:** `!pip install -q --upgrade "trl>=0.12"`.

* **Symptom:** `AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'`
  in the post-train eval cell. The cell must use
  `FastLanguageModel.from_pretrained(ADAPTER_DIR, ...)`, **not**
  `transformers.AutoModelForCausalLM`. Re-check the cell source.

### 4.2 Mode collapse (`std_return == 0`) re-appears

This is iter-1's pathology. The iter-3 sanity assert catches it.

1. Open `_artifacts/training_diagnostics.png`. If
   `reward_action_diversity` is pinned at `1/num_generations`, the
   policy collapsed. If KL is also near 0, the model never moved —
   bump LR (`5e-6` → `1e-5`).
2. Bump `MODE["grpo_num_generations"]` from 4 to 6 or 8 — more
   candidates per prompt → more diverse gradients.
3. In `GRPOConfig`, set `temperature=1.2` and add `beta=0.05` (KL
   coefficient).
4. Last resort: down-weight `reward_step_total` by wrapping it in a
   factor of 0.5. The env reward is the strongest collapse pressure.

### 4.3 Held-out scenario tanks

`mean_return` on `cloud_metadata_ssrf` more than 5 points below random.
Likely cause: policy memorised per-scenario asset names. Mitigation:
expand the prompt distribution (rendered observations from
`cloud_metadata_ssrf` in dataset rows but no reward signal) without
expanding the training reward distribution.

### 4.4 Remote eval is too slow

Each WebSocket episode is ~6–7s end-to-end. 240 baseline + 120
post-train episodes ≈ 35 min total. If you need more headroom:
reduce `MODE["n_baseline_episodes"]` and `MODE["n_post_train_episodes"]`
to 20 each, or set `MODE["eval_target"] = "local"` to bypass the
network entirely (then the judge demo loses the "look it works over
the API" win, so prefer the smaller count).

### 4.5 The HF Space serves but `/health` returns 5xx

See section 6.

---

## 5. Compute decision (no Colab Pro this week)

The user does NOT have Colab Pro this week. Use:

| Option                          | Notes                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| **Kaggle GPU notebook** (P100)  | Default this week. 30h/week free. Notebook detects Kaggle and writes to `/kaggle/working/`. |
| Colab free (T4, when quota refills) | Works; same notebook. ~12 min cooldowns kick in mid-run; not ideal for full runs. |
| HF credits (~$25 left)          | Use sparingly. Save for stretch goals like LLM-driven attacker (~$0.20/run) or hosted inference. |

**No CPU upgrade for the HF Space.** Judge constraint is 2 vCPU / 8 GB.
The Space will sleep after 48h idle; rely on the notebook's
`probe_space()` cell to wake it during evaluation. (First request after
sleep takes ~60s; the probe cell handles this.)

When Colab Pro is back next week, no code changes — just open the
notebook on Colab and Run All.

### 5.1 Why training stays on Kaggle/Colab and not HF Jobs

HF Jobs is fine for headless training, but the GRPO training relies on
TRL's `GRPOTrainer.train()` loop with mid-step env stepping (per
candidate completion). That's heavy I/O between Python and the env;
Jupyter's stickier session model makes iteration faster. Don't port to
HF Jobs unless the user explicitly asks.

---

## 6. Operating the HF Space

**Space:** <https://huggingface.co/spaces/Lonelyguyse1/cybersec>
**Live env:** <https://lonelyguyse1-cybersec.hf.space>

### 6.1 Push code changes

Use `python .scripts/push_bypass.py` (NOT `openenv push`). The bypass
script reuses openenv's staging logic but skips the `whoami-v2` API
call (rate-limited after 10–15 pushes/day) and uses
`HfApi.upload_folder(delete_patterns=["*"])` to atomically replace the
Space. Without `delete_patterns`, stale files break imports — that bit
iter-1.

```bash
python .scripts/push_bypass.py    # uses HF_TOKEN from env or ~/.cache/huggingface
```

After the push, wait ~2 min for the Space to rebuild, then re-run
`.scripts/probe_space.py` to confirm.

### 6.2 The Space says "Runtime error"

1. Open the **Logs** tab on the Space page. Look for `ImportError`.
2. The fix pattern is already in
   `cybersec/server/cybersec_environment.py`, `cybersec/attacker.py`,
   `cybersec/reward.py`, `cybersec/telemetry.py`:

   ```python
   try:
       from ..models import ...   # relative -- works on dev machines
   except ImportError:
       from cybersec.models import ...   # absolute -- works on HF runtime
   ```

3. Push again with `python .scripts/push_bypass.py`.
4. Watch the "Building" indicator turn green (~2 min).

### 6.3 The Space sleeps

Free Spaces sleep after 48h idle. Wake it by hitting `/health`. The
notebook's probe cell already does this; first request takes ~60s.

---

## 7. What to commit (and what NOT to commit)

The git repo is private during the hackathon, but the **final
submission** to the judges will be the contents of `cybersec/` only.
Treat the rest as developer scaffolding.

**Always commit:**
- `cybersec/**` — the env package
- `notebooks/cybersec_grpo.ipynb` — the train+eval script
- `tests/**` — 70+ pytest tests
- `README.md` (root pointer), `NEXT_STEPS.md`, `USER_GUIDE.md`
- `.gitignore`

**Never commit (already in `.gitignore`):**
- `.scripts/` — local operator scripts
- `_artifacts/`, `outputs/`, `eval_logs/` — run outputs
- `*.pdf`, `idea.txt`, `api.txt` — reference notes
- `**/__pycache__/`, `*.pyc`, `.venv/`
- `push.log`

When you're ready to submit: copy the `cybersec/` folder to a fresh
repo (or strip everything else from this one).

---

## 8. Stretch goals (in priority order)

These are how you turn a working iter-3 into a polished hackathon
submission. None are required.

### 8.1 LLM-driven attacker (high impact, ~half day)

The attacker is currently scripted. Replace `_pick_next_stage` in
`cybersec/attacker.py` with a small prompt to GPT-4o-mini ($0.30/1M
tokens; ~$0.20 for 50 episodes × 70 ticks). The env story becomes
"two LLMs in a long-horizon adversarial dance". This is the headline
novelty.

Keep scripted as default; the LLM one is opt-in via constructor flag
so tests stay deterministic.

### 8.2 Per-stage credit shaping (medium, ~2h)

Add a `stage_progress` channel to the env reward in `cybersec/reward.py`
that emits ±0.5 when a stage transitions to/from `success`. Denser
long-horizon signal.

### 8.3 LLM-judged "explanation" reward (high impact, ~half day)

Add an optional reward that asks GPT-4o-mini "did this action plausibly
defend against the attacker's likely next move?". Adds a qualitative
channel that canned-plan hacks can't trivially exploit.

### 8.4 Public README screenshots

After a successful run, screenshot the HF Space card,
`before_after_curves.png`, `training_diagnostics.png`, and the
notebook's headline-delta print. Add to `cybersec/README.md` under a
"Results" section. Judges scan images.

---

## 9. What the previous (Sonnet/Opus) session left in flight

If anything is checked off when you start, skip it.

* [x] `openenv validate cybersec` passes; layout matches `openenv init`.
* [x] HF Space deployed and serving the iter-3 6-action surface.
* [x] All 4 scenarios available on the live Space (incl.
      `cloud_metadata_ssrf`).
* [x] Memory probe verified: < 200 MB at 8 concurrent envs.
* [x] Notebook rewritten so baseline + post-train eval go over the
      live OpenEnv WebSocket protocol against the HF Space.
* [x] Notebook auto-detects Colab / Kaggle / local for path & install.
* [x] User-supervised step-by-step guide added (`USER_GUIDE.md`).
* [ ] **Run cybersec_grpo.ipynb on Kaggle** and record results in
      `_artifacts/`. Iter-2's results were never preserved because the
      Colab quota ran out mid-run.
* [ ] **If the run produces good results** (mode collapse broken,
      headline delta positive on at least one train scenario), enable
      `PUSH_ARTIFACTS = True` in the last cell to upload `_artifacts/`
      to the Space, and add screenshots to `cybersec/README.md`.

---

## 10. When you're stuck and want to escalate back to Sonnet/Opus

Bundle the following into one prompt for the next premium-model session:

1. The exact error / unexpected output. Include the cell number and
   the last 30 lines of output.
2. `_artifacts/training_log.json` (last 20 entries) and
   `_artifacts/training_diagnostics.png`.
3. `_artifacts/summary_table.md`.
4. `git log --oneline -10` to show what changed since iter-3.
5. The output of `python -m pytest tests/ -q`.

That gives the next session the entire state of the experiment without
re-reading the prior conversation.

---

*Last updated: iter-3 (Sonnet 4.5 session, just before model rotation).
The previous session's transcript is at the path the IDE reports as
`agent-transcripts/<uuid>.jsonl` — only consult it if this file is
ambiguous, since it is very large.*
