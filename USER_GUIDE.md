# User guide: how to drive the iter-4 training cycle yourself

**Audience:** you (the human owner of this repo). This guide assumes
you'll be supervising a cheaper model (e.g. Haiku, GPT-4o-mini, a free
local model) for the next phase rather than re-engaging Sonnet/Opus.
It tells you exactly which command to run, what success looks like,
and which screenshot or log line to copy back so the cheap model can
help you decide what's next.

If you ever feel lost, the long technical playbook is `NEXT_STEPS.md`.

---

## 0. Setup that you only do once

You should already have these. Run them once if you don't.

```powershell
git pull
python -m pip install -e cybersec
python -m pip install pytest psutil huggingface_hub openenv-core
```

If any of those fail, paste the full error into the cheap model and
ask "what's missing?". They are simple `pip` errors.

You should already have done `huggingface-cli login` once and the
token is cached in `~/.cache/huggingface/token`. If `python
.scripts/push_bypass.py` says "no token", run `huggingface-cli login`
again.

---

## 1. Pre-flight (5 min, on your laptop)

Run these 4 commands and read the last line of each. Paste any failures
into the cheap model.

```powershell
python -m pytest tests/ -q
```
**Expect**: `73 passed in N.Ns`. Anything else: stop.

```powershell
python -c "import urllib.request; print(urllib.request.urlopen('https://lonelyguyse1-cybersec.hf.space/health', timeout=10).read())"
```
**Expect**: `b'{"status":"healthy"}'`. If 5xx or it hangs > 60s, the
Space is asleep — open the Space page in a browser, click "Restart",
wait 2 min, then try again.

```powershell
python .scripts/probe_space.py
```
**Expect**: 6 actions listed (MONITOR, INVESTIGATE, ISOLATE_ASSET,
REVOKE_IDENTITY, BLOCK_EGRESS, PATCH_ASSET). If you see different
names, the Space has stale code — go to step 2.

```powershell
python train.py --baseline-only --n-baseline-episodes 2
```
**Expect**: 4 lines like `[baseline] supply_chain_token_drift random=N.NN heuristic=N.NN  (2 ep each)`,
ending with `[main] --baseline-only set; exiting.` This both confirms
the live Space is reachable AND smoke-tests the superscript itself.

If all four green: skip step 2 and go to step 3.

---

## 2. (Only if step 1 reported stale Space) Push the latest code

```powershell
python .scripts/push_bypass.py
```

Wait 2 min. The Space rebuilds. Re-run all four pre-flight commands.

If `push_bypass.py` errors with "rate limited", wait 10 min then try
again. If "permission denied", `huggingface-cli login` again with the
right account.

---

## 3. Running the full training pipeline (the main event)

This is what produces the result. Estimated wall clock: **65–85 min**.
You have **two equivalent paths**; pick whichever you prefer.

### 3.A Headless on Kaggle (one command)

Easiest if Kaggle's GPU is what you've got.

1. Push your repo to GitHub (`git push`).
2. New Kaggle notebook → Settings → **GPU P100** + **Internet ON**.
3. Add a single code cell:

   ```bash
   !git clone https://github.com/<your-fork>/cybersec_env.git
   %cd cybersec_env
   !python train.py
   ```

4. Click "Run All".

`train.py` self-installs all deps, probes the Space, runs the entire
pipeline, and writes everything to `/kaggle/working/cybersec_workdir/_artifacts/`
(or just `cybersec_workdir/` on Kaggle). The console prints are
self-narrating: every section starts with `[probe] / [baseline] /
[dataset] / [train] / [eval-train] / [eval-OOD] / [canary] / [main]`.

### 3.B Notebook on Kaggle / Colab

Easiest if you want to watch progress cell-by-cell.

1. Get `notebooks/cybersec_grpo.ipynb` onto Kaggle/Colab (upload, or
   `git clone`, your call).
2. **Runtime → GPU**. Internet ON.
3. Run All.

Same pipeline, same artifacts; just a different UX.

### 3.3 What you should see while it runs

Don't watch every line. Just check these milestones:

| About when    | What                                          | Expect                                                                                         |
| ------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 0 min         | install / probe                               | `[probe] space health: {'status': 'healthy'}` and the 6-action list.                           |
| 1–25 min      | `[baseline] ...` lines                        | Four `[baseline] <scenario>` rows with `random=...  heuristic=...`. Heuristic should beat random on aggregate. |
| 26 min        | `[dataset] N rows  built in ...s`             | `N` should be 1500.                                                                            |
| 27–28 min     | `Trainable parameters: ...`                   | Confirms Unsloth loaded the LoRA modules.                                                      |
| 28–55 min     | TRL training progress bars                    | Loss should trend down; KL should stay below ~10. NOT a stack trace.                            |
| 55 min        | `[train] saved adapter -> ...`                | Adapter persisted. After this is just eval.                                                    |
| 56–80 min     | `[eval-train]` and `[eval-OOD]` lines         | Each row shows `mean_return  std=N.NNN  monitor_fallback_rate=N.NNN`. **At least 2 of the 3 train rows must have `std > 0.1`** (this is the iter-4 canary). |
| 80 min        | `[canary] all sanity checks passed`           | If this asserts, copy the assertion message + every `std_return=...` line and paste into the cheap model. |

### 3.4 If a step fails

1. Find the section name (`[probe]`, `[baseline]`, `[train]`, ...).
2. Copy the **last 40 lines** of console output.
3. Paste this into the cheap model:
   ```
   I'm running cybersec_env/train.py on Kaggle (or notebooks/cybersec_grpo.ipynb).
   The [SECTION] step failed with this error:
   <paste last 40 lines>

   The script source is at https://github.com/<your-fork>/cybersec_env/blob/main/train.py
   (or notebooks/cybersec_grpo.ipynb).
   What's the smallest fix? Give me the exact line(s) to change.
   ```
4. Apply the fix. **Don't restart from zero** — `_artifacts/` is
   checkpointed; rerun only the failing step (or for `train.py`, just
   re-run the whole script — early steps are cheap).

### 3.5 If the run finishes successfully

Copy these out of `_artifacts/` (or `/kaggle/working/cybersec_workdir/_artifacts/`):

* `summary_table.md`
* `training_diagnostics.png`
* `before_after_curves.png`
* `baseline_metrics.json`
* `post_train_metrics.json`

These are your **submission evidence**. Keep them safe.

### 3.6 If a canary asserts (mode collapse, fallback-heavy, etc.)

The iter-4 canaries fail loud on purpose. The most common ones:

| Canary message                            | Means                                                       | Fix to try                                                                       |
|-------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `REWARD HACK CANARY: ... < 0.1 ...`        | ≥2 train scenarios still have `std_return ≈ 0`. The policy memorised canned plans. | Bump exploration: `--grpo-num-generations 8 --grpo-temperature 1.4 --grpo-beta 0.06`. |
| `monitor_fallback_rate ... 0.5+`          | Trained model is mostly emitting unparseable text.          | Use a longer adapter run (`--grpo-max-steps 200`) or check the chat template hasn't broken. |
| `valid_rate < 80%`                         | Trained model is sending env-rejected actions.              | Same as above + verify SYSTEM_PROMPT lists the action enum correctly.            |
| `trained-llm ... >5pts below random`       | Training diverged catastrophically.                         | Lower lr (`--grpo-learning-rate 1e-6`), reduce `--grpo-max-steps`.                |

Re-run with the suggested override:

```powershell
python train.py --grpo-num-generations 8 --grpo-temperature 1.4 --grpo-beta 0.06
```

---

## 4. After a successful run — the things you should do

### 4.1 Update the README with results

Open `cybersec/README.md` and find the "Baselines (30 episodes per
scenario, against the live HF Space)" section. Update the table with
new numbers if any. Then add or update a small "Iter-4 results"
paragraph linking the two PNGs.

Use the cheap model: paste `summary_table.md` and ask "write me 2
sentences on the headline finding."

### 4.2 (Optional) Push artifacts back to the Space

In `train.py` we don't auto-upload (it's a CLI), but the notebook has
a final cell with `PUSH_ARTIFACTS = True`. Either: re-run that cell,
or do it manually:

```powershell
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='_artifacts', path_in_repo='_artifacts', repo_id='Lonelyguyse1/cybersec', repo_type='space', commit_message='iter-4 artifacts')"
```

### 4.3 Commit + push to GitHub

```powershell
git add cybersec/README.md
git status     # CHECK before committing
git commit -m "iter-4 results"
git push
```

`_artifacts/` is gitignored on purpose; only commit it if a judge asks
you to (the Space already holds it, see 4.2).

---

## 5. Final submission scope

Your submission is **only the `cybersec/` directory + the training
script**. Specifically:

* `cybersec/` (the entire env package — pyproject, Dockerfile, openenv.yaml, `cybersec/` source, `cybersec/training/`, `cybersec/server/`).
* `train.py` (the headless superscript).
* `notebooks/cybersec_grpo.ipynb` (the notebook UX).

Everything else (`tests/`, `.scripts/`, `_artifacts/`, `NEXT_STEPS.md`,
`USER_GUIDE.md`, `*.pdf`, `idea.txt`, `api.txt`) is developer
scaffolding the judge does NOT need.

When you're ready, from the repo root:

```powershell
mkdir _submission
Copy-Item -Recurse cybersec _submission\cybersec
Copy-Item train.py _submission\train.py
Copy-Item notebooks\cybersec_grpo.ipynb _submission\cybersec_grpo.ipynb
Compress-Archive -Path _submission\* -DestinationPath cybersec_submission.zip
```

Final deliverable: `cybersec_submission.zip` containing the env
package + `train.py` + the notebook.

---

## 6. Quick reference (cheat sheet)

| What                                           | Command                                              |
| ---------------------------------------------- | ---------------------------------------------------- |
| Run all tests                                  | `python -m pytest tests/ -q`                         |
| Check Space is alive                           | `python .scripts/probe_space.py`                     |
| Smoke single remote episode                    | `python .scripts/remote_smoke.py`                    |
| Smoke baseline episodes via remote env         | `python train.py --baseline-only --n-baseline-episodes 5` |
| Push iter-N to the Space                       | `python .scripts/push_bypass.py`                     |
| Regenerate the notebook from the build script  | `python .scripts/build_unified_notebook.py`          |
| Validate the OpenEnv layout                    | `openenv validate cybersec`                          |
| Run the full training pipeline                 | `python train.py`                                    |
| Override any MODE dial                         | `python train.py -h`                                 |

---

## 7. When in doubt

Read `NEXT_STEPS.md`. If it says X and this guide says Y, this guide
wins (it's been tuned for your no-Pro / Kaggle-first workflow).

If you want to escalate back to Sonnet/Opus, bundle into one prompt:
1. The section name + last 40 lines of output.
2. `_artifacts/summary_table.md`.
3. `_artifacts/training_diagnostics.png` (attach the file).
4. `git log --oneline -10`.
5. `python -m pytest tests/ -q` output.
