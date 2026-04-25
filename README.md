# Cybersec OpenEnv — repo root

The OpenEnv environment lives in [`cybersec/`](./cybersec/), following the official
`openenv init` layout (env folder = Python package, with `openenv.yaml`,
`pyproject.toml`, and `server/Dockerfile` *inside* the env folder, not at the
repo root).

> **Read [`cybersec/README.md`](./cybersec/README.md)** — canonical spec for the
> environment and Hugging Face Spaces card.

## For reviewers / judges (quick read)

- **Task:** Long-horizon, multi-agent cyber defense: scripted MITRE-aligned
  attacker vs a defender with six structured actions per tick, partial
  observations, and multi-channel rewards (see `cybersec/README.md`).
- **Train + eval:** [`notebooks/cybersec_grpo.ipynb`](./notebooks/cybersec_grpo.ipynb)
  is the **single** end-to-end path: baselines → GRPO (Unsloth QLoRA) →
  **base LLM vs trained** eval on the same seeds, plots, **strict canaries**
  (e.g. per-train-scenario return variance, MONITOR parse-fallback cap).
- **Live env:** client–server OpenEnv over WebSocket (typical deployment:
  [HF Space](https://huggingface.co/spaces/Lonelyguyse1/cybersec)). First
  request after sleep can take **30–60s**; plan for cold start when smoke-testing.
- **Auth (optional):** set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (e.g. Colab
  **User secrets**) when pulling models or the Space snapshot to avoid rate-limit
  / unauthenticated hub noise in logs.
- **Repro:** `pip install -e ./cybersec[dev]` from this repo, then open the
  notebook locally, or use **Open in Colab** from the notebook header.

## What lives at this level

| Path | What it is |
|------|------------|
| `cybersec/` | The OpenEnv environment package (action/observation models, scripted attacker, FastAPI server, Dockerfile, `openenv.yaml`, `training/` reward functions). |
| `notebooks/cybersec_grpo.ipynb` | **Canonical** training + eval pipeline (Run All in Colab/Kaggle or Jupyter). |
| `tests/` | Pytest contract tests for the env, reward functions, and reward-hack canaries. |

## Run the full pipeline

```bash
pip install -e ./cybersec[dev]
jupyter notebook notebooks/cybersec_grpo.ipynb
```

Defaults in the notebook target a **full** GRPO budget (e.g. 300 steps); reduce
`MODE["grpo_max_steps"]` and episode counts for a quick CPU/GPU smoke run.

## Verifying the env contract

```bash
pytest -q
openenv validate cybersec
cybersec-server
```

For design rationale, action/reward spec, scenarios, Docker, and HF Spaces
deployment, see [`cybersec/README.md`](./cybersec/README.md).
