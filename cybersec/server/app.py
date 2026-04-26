"""FastAPI application for the Cybersec OpenEnv environment.

The OpenEnv ``openenv.yaml`` declares ``app: server.app:app`` so this module
must expose a module-level ``app`` and a ``main`` callable for uvicorn.

Endpoints (provided by ``openenv.core.env_server.create_app``):

  * POST /reset   - one-shot reset of a fresh environment instance
  * POST /step    - one-shot step of a fresh environment instance
  * GET  /state   - current environment state
  * GET  /schema  - action / observation schemas
  * WS   /ws      - persistent WebSocket session (use this for episodes)

Configuration knobs (read at process start, not per-request):

  ``CYBERSEC_SCENARIO_ID``          - default scenario; if unset, scenario follows
                                      reset ``seed`` (see ``CybersecEnvironment.reset``;
                                      omitted seed uses a random seed per reset).
  ``CYBERSEC_ATTACKER_PERSONALITY`` - default personality; if unset, sampled by RNG.
  ``CYBERSEC_MAX_CONCURRENT_ENVS``  - max concurrent WebSocket sessions (default 8).
"""

from __future__ import annotations

import os
from typing import Optional

from openenv.core.env_server.http_server import create_app

try:
    from ..models import AttackerPersonality, CybersecAction, CybersecObservation
    from .cybersec_environment import CybersecEnvironment
except ImportError:  # pragma: no cover - hit only when CWD == cybersec/ and we're run as `server.app`
    from models import AttackerPersonality, CybersecAction, CybersecObservation  # type: ignore[no-redef]
    from server.cybersec_environment import CybersecEnvironment  # type: ignore[no-redef]


def _env_factory() -> CybersecEnvironment:
    scenario_id: Optional[str] = os.getenv("CYBERSEC_SCENARIO_ID") or None
    personality_raw: Optional[str] = os.getenv("CYBERSEC_ATTACKER_PERSONALITY") or None
    personality = AttackerPersonality(personality_raw) if personality_raw else None
    return CybersecEnvironment(
        default_scenario_id=scenario_id,
        default_personality=personality,
    )


_max_concurrent = int(os.getenv("CYBERSEC_MAX_CONCURRENT_ENVS", "8"))


app = create_app(
    env=_env_factory,
    action_cls=CybersecAction,
    observation_cls=CybersecObservation,
    env_name="cybersec",
    max_concurrent_envs=_max_concurrent,
)


def main() -> None:
    """Console-script entry point.

    Used by:

      * the ``server`` console script declared in ``pyproject.toml``
        (``openenv validate`` requires this exact name).
      * ``python -m cybersec.server.app`` and ``python server/app.py``.

    CLI flags are honored for local development; ``CYBERSEC_HOST`` /
    ``CYBERSEC_PORT`` env vars take precedence so the Docker image and
    Hugging Face Spaces deploy can override the bind address without code
    changes.
    """

    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Cybersec OpenEnv server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    host = os.getenv("CYBERSEC_HOST", args.host)
    port = int(os.getenv("CYBERSEC_PORT", str(args.port)))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
