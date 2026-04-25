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
| `cybersec/` | The OpenEnv environment package (action/observation models, scripted attacker, FastAPI server, Dockerfile, `openenv.yaml`). |
| `tests/` | Pytest contract tests for the env and scenarios. |
| `notebooks/` | One end-to-end Colab notebook (`cybersec_grpo.ipynb`) that runs baseline eval -> GRPO training -> post-training eval in a single Run-All. |
| `*.pdf`, `*.txt` | Hackathon reference material (themes, judging criteria, FAQ). |

## Three commands to get going

```bash
# 1. Install the env (run from repo root)
pip install -e ./cybersec[dev]

# 2. Verify the contract
pytest -q

# 3. Run the env locally
cybersec-server   # then connect with cybersec.CybersecEnv(base_url="http://localhost:8000")
```

For everything else — design rationale, action/reward spec, baselines,
training notebooks, Docker, HF Spaces deployment — see
[`cybersec/README.md`](./cybersec/README.md).
