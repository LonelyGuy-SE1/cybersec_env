"""Async + sync client for talking to the Cybersec environment server.

The class is the same shape as every other OpenEnv environment: subclass
:class:`openenv.core.env_client.EnvClient`, implement ``_step_payload`` and
``_parse_result``, and inherit transport from the base class.

The async client is the canonical entry point. Get a sync wrapper via
``CybersecEnv(...).sync()`` if you don't want to write ``await`` everywhere.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CybersecAction, CybersecObservation, CybersecState


class CybersecEnv(EnvClient[CybersecAction, CybersecObservation, CybersecState]):
    """Persistent-session client for :class:`CybersecEnvironment`."""

    def _step_payload(self, action: CybersecAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CybersecObservation]:
        # The WebSocket protocol echoes the same shape as the HTTP responses:
        # ``{"observation": {...}, "reward": ..., "done": ...}``. Pull the
        # inner observation, but fall back to ``payload`` itself for robustness
        # if a future server version returns the observation directly.
        obs_payload = payload.get("observation", payload)
        observation = CybersecObservation.model_validate(obs_payload)
        reward = payload.get("reward", observation.reward)
        done = bool(payload.get("done", observation.done))
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> CybersecState:
        # ``GET /state`` (and the WebSocket ``state`` request) ship the raw
        # state dict; it is *not* nested under another key.
        return CybersecState.model_validate(payload)


__all__ = ["CybersecEnv"]
