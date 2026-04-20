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
except ImportError:
    from models import CybersecAction, CybersecObservation
    from server.cybersec_environment import CybersecEnvironment


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc
    return value


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc
    return value


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float") from exc
    return value


def _bounded_int(
    *,
    value: int | None,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    if value < minimum:
        raise RuntimeError(f"Environment variable {name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"Environment variable {name} must be <= {maximum}")
    return value


def _bounded_float(
    *,
    value: float | None,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if value < minimum or value > maximum:
        raise RuntimeError(
            f"Environment variable {name} must be between {minimum} and {maximum}"
        )
    return value


def _env_factory() -> CybersecEnvironment:
    scenario_id = os.getenv("CYBERSEC_SCENARIO_ID", "supply_chain_token_drift")
    horizon = _bounded_int(
        value=_env_optional_int("CYBERSEC_HORIZON"),
        name="CYBERSEC_HORIZON",
        minimum=1,
    )
    false_positive_rate = _bounded_float(
        value=_env_optional_float("CYBERSEC_FALSE_POSITIVE_RATE"),
        name="CYBERSEC_FALSE_POSITIVE_RATE",
        minimum=0.0,
        maximum=1.0,
    )
    seed = _bounded_int(
        value=_env_optional_int("CYBERSEC_DEFAULT_SEED"),
        name="CYBERSEC_DEFAULT_SEED",
        minimum=0,
    )
    return CybersecEnvironment(
        scenario_id=scenario_id,
        seed=seed,
        horizon=horizon,
        false_positive_rate=false_positive_rate,
    )


def _max_concurrent_envs() -> int:
    value = _env_int("CYBERSEC_MAX_CONCURRENT_ENVS", 4)
    if value < 1:
        raise RuntimeError(
            "Environment variable CYBERSEC_MAX_CONCURRENT_ENVS must be >= 1"
        )
    return value


app = create_app(
    _env_factory,
    CybersecAction,
    CybersecObservation,
    env_name="cybersec",
    max_concurrent_envs=_max_concurrent_envs(),
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
