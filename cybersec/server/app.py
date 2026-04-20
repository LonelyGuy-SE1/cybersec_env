# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for CybersecEnv."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CybersecAction, CybersecObservation
    from .cybersec_environment import CybersecEnvironment
except ModuleNotFoundError:
    from models import CybersecAction, CybersecObservation
    from server.cybersec_environment import CybersecEnvironment


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float") from exc


def _env_factory() -> CybersecEnvironment:
    scenario_id = os.getenv("CYBERSEC_SCENARIO_ID", "supply_chain_token_drift")
    horizon = _env_optional_int("CYBERSEC_HORIZON")
    false_positive_rate = _env_optional_float("CYBERSEC_FALSE_POSITIVE_RATE")
    seed = _env_optional_int("CYBERSEC_DEFAULT_SEED")
    return CybersecEnvironment(
        scenario_id=scenario_id,
        seed=seed,
        horizon=horizon,
        false_positive_rate=false_positive_rate,
    )


app = create_app(
    _env_factory,
    CybersecAction,
    CybersecObservation,
    env_name="cybersec",
    max_concurrent_envs=_env_int("CYBERSEC_MAX_CONCURRENT_ENVS", 4),
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m cybersec.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn cybersec.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
