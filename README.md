# Cybersec OpenEnv — repo root

The actual OpenEnv environment lives in [`cybersec/`](./cybersec/), following
the official `openenv init` layout (env folder = Python package, with
`openenv.yaml`, `pyproject.toml`, and `server/Dockerfile` *inside* the env
folder, not at the repo root).

> **Read [`cybersec/README.md`](./cybersec/README.md)** — that file is the
> canonical spec for the environment and doubles as the Hugging Face Spaces
> card.

## What lives at this level

| Path | What it is |
|---|---|
| `cybersec/` | The OpenEnv environment package (action/observation models, scripted attacker, FastAPI server, Dockerfile, `openenv.yaml`, `training/` reward functions). |
| `train.py` | The end-to-end **training superscript**. One command: baseline eval → dataset build → GRPO QLoRA training → post-training eval against the live HF Space → sanity canaries. Mirrors the notebook 1:1. |
| `notebooks/cybersec_grpo.ipynb` | Same pipeline, click-through Colab/Kaggle UX. |
| `tests/` | Pytest contract tests for the env, reward functions, and reward-hack canaries. |

## Two ways to run the full pipeline

```bash
# 1. Install the env (run from repo root)
pip install -e ./cybersec[dev]

# A) Headless — the canonical "superscript".
#    Defaults: GRPO 120 steps, 30 baseline + 30 post-train episodes/scenario,
#    eval over the deployed HF Space.
python train.py

# B) Notebook — same pipeline, but visible step-by-step in Colab/Kaggle.
jupyter notebook notebooks/cybersec_grpo.ipynb
```

`train.py` accepts overrides for every dial (run `python train.py -h`).
For a sub-1-minute smoke against the live Space:

```bash
python train.py --baseline-only --n-baseline-episodes 2
```

## Verifying the env contract

```bash
pytest -q                           # 73 tests, includes reward-hack canaries
openenv validate cybersec           # OpenEnv layout / metadata check
cybersec-server                     # serve the env locally on port 8000
```

For everything else — design rationale, action/reward spec, baselines,
iteration history, scenarios, Docker, HF Spaces deployment — see
[`cybersec/README.md`](./cybersec/README.md).
