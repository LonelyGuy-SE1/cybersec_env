# User guide: how to drive the next training cycle yourself

**Audience:** you (the human owner of this repo). This guide assumes
you'll be supervising a cheaper model (e.g. Haiku, GPT-4o-mini, a free
local model) for the next phase rather than re-engaging Sonnet/Opus.
It tells you exactly which command to run, what success looks like,
and which screenshot or log line to copy back so the cheap model can
help you decide what's next.

If you ever feel lost, the long technical playbook is `NEXT_STEPS.md`
— a cheaper model can read it and follow it, but you should run the
commands.

---

## 0. Setup that you only do once

You should already have these. Run them once if you don't.

```powershell
# from the repo root, in PowerShell
git pull
python -m pip install -e cybersec
python -m pip install pytest psutil huggingface_hub openenv-core
```

If any of those fail, paste the full error into the cheap model and
ask "what's missing?". They are simple `pip` errors; even a cheap
model can fix them.

You should already have done `huggingface-cli login` once and the
token is cached in `~/.cache/huggingface/token`. If `python
.scripts/push_bypass.py` says "no token", run
`huggingface-cli login` again.

---

## 1. Pre-flight (5 min, you on your laptop)

Run these 4 commands and read the last line of each. Paste any failures
into the cheap model.

```powershell
python -m pytest tests/ -q
```
**Expect**: `70+ passed in N.Ns` at the bottom. Anything else: stop.

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
REVOKE_IDENTITY, BLOCK_EGRESS, PATCH_ASSET). If you see 11 actions or
different names, the Space has stale code — go to step 2.

```powershell
python .scripts/remote_smoke.py
```
**Expect**: `OK  steps=10  return=0.300  latency=4-7s  done=True`.
If "Unknown scenario_id", the Space is on stale scenarios — go to step 2.

If all four green: skip step 2 and go to step 3.

---

## 2. (Only if step 1 reported stale Space) Push the latest code

```powershell
python .scripts/push_bypass.py
```

Wait 2 min. The Space rebuilds. Re-run all four pre-flight commands.

If `push_bypass.py` errors with "rate limited", wait 10 min then try
again. If it errors with "permission denied", `huggingface-cli login`
again with the right account.

---

## 3. Running the notebook on Kaggle (the main event)

This is what produces the result. Estimated wall clock: **55–75 min**.

### 3.1 Get the notebook file

You need `notebooks/cybersec_grpo.ipynb` from this repo. Either:

- **Easy**: download the file from your local repo and upload it to
  Kaggle.
- **Cleanest**: push the repo to GitHub (`git push`) and then in
  Kaggle, click "New Notebook → File → Import Notebook → from URL"
  and paste the GitHub URL of the .ipynb.

### 3.2 Open in Kaggle

1. Go to <https://kaggle.com/code> and click "+ New Notebook".
2. Click "File" → "Upload Notebook" → choose `cybersec_grpo.ipynb`.
3. Top right, click the gear (Settings):
   - Accelerator → **GPU P100** (you get 30h/week free).
   - Persistence → "Files only" is fine.
   - Internet → **must be ON** (the notebook hits the HF Space).
4. Top right, click "Run All" (or `Ctrl + F9`).

### 3.3 What you should see while it runs

Don't watch every cell. Just check these milestones:

| About when    | What                                         | Expect                                                                          |
| ------------- | -------------------------------------------- | ------------------------------------------------------------------------------- |
| 0 min         | Cell 1 ("Detect runtime")                    | Prints `runtime: kaggle` and a workdir under `/kaggle/working`.                  |
| 1 min         | Cell 2 (install)                             | Long stream of `pip install` lines. Should be quiet — no big stack trace.        |
| 3 min         | Cell 5 (probe space)                         | Prints `OK: space is healthy and on the iter-3 action surface`.                  |
| 30 min        | Cell 8 (baseline runs end)                   | Prints `baseline elapsed: NNNs (remote env)`. NNN should be ~1500–2000s.         |
| 35 min        | Cell 14 (GRPO trainer starts)                | Prints "Training" + a progress bar. **NOT** a stack trace.                       |
| 60 min        | Cell 17–18 (training ends, diagnostics plot) | A 3-panel plot appears. Watch the middle panel: reward should be trending up.    |
| 70 min        | Cell 22 (post-train eval)                    | Per-scenario `trained-llm=N.NNN std=N.NNN` lines. **`std` MUST NOT be 0.000**.   |
| 75 min        | Cell 28 (sanity asserts)                     | "all sanity checks passed". If this asserts, copy the assertion message + the per-scenario `std_return` lines and paste into the cheap model. |

### 3.4 If a cell fails

1. Click the cell that failed. Read the **last 30 lines** of the
   error.
2. Find the cell number (Kaggle shows it in the left margin).
3. Paste this into the cheap model:
   ```
   I'm running cybersec_grpo.ipynb on Kaggle.
   Cell N failed with this error:
   <paste last 30 lines>

   The notebook source is at https://github.com/<your-fork>/cybersec_env/blob/main/notebooks/cybersec_grpo.ipynb (cell N).
   What's the smallest fix? Give me the exact line(s) to change.
   ```
4. Apply the fix the cheap model suggests. **Do NOT** re-run from
   cell 0 — restart the kernel and re-run from the failing cell
   onward. Most cells are checkpointed (the dataset and adapter are
   persisted under `/kaggle/working/cybersec_workdir/`).

### 3.5 If the run finishes successfully

Copy these 4 things out of `/kaggle/working/cybersec_workdir/`:

* `summary_table.md`
* `training_diagnostics.png`
* `before_after_curves.png`
* `baseline_metrics.json`

Either: (a) right-click → download in Kaggle, or (b) bottom-right
"Output" panel → "Save Version" → the files appear there for download.

These four files together are your **submission evidence**. Keep them
safe.

### 3.6 If `std_return == 0` on every training scenario (mode collapse)

That's iter-1's exact failure pattern. The sanity asserts will catch
it. Don't panic; this is a tunable problem.

Copy this into the cheap model:

```
GRPO training finished but the trained-llm has std_return == 0 on every
training scenario. The sanity assert "REWARD HACK CANARY" fired. The
training_diagnostics.png shows reward_action_diversity is pinned at 0.25
(= 1 / num_generations).

What's the cheapest tweak to break the collapse? My options per
NEXT_STEPS.md section 4.2 are:
 (a) bump grpo_num_generations from 4 to 6
 (b) raise temperature from 1.0 to 1.2
 (c) add beta=0.05 KL coefficient
 (d) down-weight reward_step_total by 0.5

Pick one and tell me exactly which lines to change in MODE / GRPOConfig.
```

Apply the suggestion. Re-run from cell 13 (the GRPO config cell)
onward. The dataset and baselines are already cached.

---

## 4. After a successful run — the things you should do

### 4.1 Update the README with results

Open `cybersec/README.md` and add a "Results (iter-3)" section near
the top. Paste the contents of `summary_table.md` and link the two
images. Use the cheap model to phrase 1–2 sentences on the headline
finding.

### 4.2 (Optional) Push artifacts back to the Space

In the last notebook cell, set `PUSH_ARTIFACTS = True` and re-run that
cell. It uploads `_artifacts/` to the Space at `path_in_repo="_artifacts"`.
Judges who clone the Space see the same numbers you did.

### 4.3 Commit + push to GitHub

```powershell
git add cybersec/README.md _artifacts/  # _artifacts/ is normally gitignored; add explicitly only if you want it in the repo
git status                              # CHECK before committing
git commit -m "iter-3 results"
git push
```

If you decide NOT to commit `_artifacts/` (recommended; the Space
holds them), only add `cybersec/README.md`.

---

## 5. Final submission scope

Your submission to the judges is **only the `cybersec/` directory** —
the package, its `pyproject.toml`, its `Dockerfile`, its `openenv.yaml`.

Everything else (notebook, tests, `.scripts/`, `_artifacts/`,
`NEXT_STEPS.md`, `USER_GUIDE.md`, `*.pdf`, `idea.txt`) is developer
scaffolding. The judge ONLY needs to:

1. Visit the Space URL.
2. Run the notebook (which they pull from your GitHub) on their own
   Colab/Kaggle.
3. Watch episodes flow through their notebook ↔ your Space.

When you're ready, do this from the repo root:

```powershell
# Make a clean submission folder
mkdir _submission
cp -r cybersec/* _submission/
cp notebooks/cybersec_grpo.ipynb _submission/cybersec_grpo.ipynb
# zip it up if the judges want a zip
Compress-Archive -Path _submission/* -DestinationPath cybersec_submission.zip
```

The zip is your final deliverable: `cybersec/` package code + the one
training script.

---

## 6. Quick reference (cheat sheet)

| What                                           | Command                                              |
| ---------------------------------------------- | ---------------------------------------------------- |
| Run all tests                                  | `python -m pytest tests/ -q`                         |
| Check Space is alive                           | `python .scripts/probe_space.py`                     |
| Smoke a single remote episode                  | `python .scripts/remote_smoke.py`                    |
| Smoke 6 baseline episodes via remote env       | `python .scripts/smoke_remote_baselines.py`          |
| Push iter-N to the Space                       | `python .scripts/push_bypass.py`                     |
| Regenerate the notebook from the build script  | `python .scripts/build_unified_notebook.py`          |
| Measure env memory (must fit 8 GB Space)       | `python .scripts/memory_probe.py`                    |
| Validate the OpenEnv layout                    | `openenv validate cybersec`                          |
| Inspect a Colab/Kaggle .ipynb output offline   | `python .scripts/inspect_results.py path/to.ipynb`   |

---

## 7. When in doubt

Read `NEXT_STEPS.md`. If it says X and this guide says Y, this guide
wins (it's been tuned for your no-Pro / Kaggle-first workflow).

If you want to escalate back to Sonnet/Opus, bundle into one prompt:
1. The cell number + last 30 lines of output.
2. `_artifacts/summary_table.md`.
3. `_artifacts/training_diagnostics.png` (attach the file).
4. `git log --oneline -10`.
5. `python -m pytest tests/ -q` output.
