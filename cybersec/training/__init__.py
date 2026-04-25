"""Training-side helpers shipped with the Cybersec OpenEnv environment.

Anything in this subpackage is meant to be consumed by RL training scripts
(notebooks, headless trainers, evaluation harnesses). The env itself
(`cybersec.server.CybersecEnvironment`) does not import from here, so the
runtime stays free of training dependencies.

Public surface:

  * :func:`default_reward_funcs`       - the canonical TRL-compatible reward
                                         function list used in the unified
                                         GRPO notebook.
  * :data:`SYSTEM_PROMPT`              - chat-template system prompt used
                                         during dataset build and inference.
  * :func:`render_observation`         - turn a `CybersecObservation` into
                                         the user-message text the LLM sees.
  * :func:`snapshot_env` /
    :func:`restore_env`                - pickle a `CybersecEnvironment` to a
                                         base64 blob so a TRL reward function
                                         can clone it and apply a candidate
                                         action.
  * :func:`parse_first_json_object` /
    :func:`parsed_action`              - lenient parsers used by every reward
                                         function and by the trained policy.
"""

from __future__ import annotations

from .rewards import (  # noqa: F401
    SYSTEM_PROMPT,
    default_reward_funcs,
    parse_first_json_object,
    parsed_action,
    render_observation,
    restore_env,
    reward_action_diversity,
    reward_avoids_exfil_path,
    reward_batch_action_entropy,
    reward_json_valid,
    reward_no_redundant_containment,
    reward_observation_aware,
    reward_schema_valid,
    reward_step_total,
    reward_target_in_valid_targets,
    snapshot_env,
)

__all__ = [
    "SYSTEM_PROMPT",
    "default_reward_funcs",
    "parse_first_json_object",
    "parsed_action",
    "render_observation",
    "restore_env",
    "reward_action_diversity",
    "reward_avoids_exfil_path",
    "reward_batch_action_entropy",
    "reward_json_valid",
    "reward_no_redundant_containment",
    "reward_observation_aware",
    "reward_schema_valid",
    "reward_step_total",
    "reward_target_in_valid_targets",
    "snapshot_env",
]
