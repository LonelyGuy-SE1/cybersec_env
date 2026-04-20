"""Episode runner and aggregate metrics for CybersecEnv baselines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

try:
    from cybersec.models import CybersecAction, CybersecObservation
except ModuleNotFoundError:
    from models import CybersecAction, CybersecObservation

from .policy import DefenderPolicy

try:
    from cybersec.server.cybersec_environment import CybersecEnvironment
except ModuleNotFoundError:
    from server.cybersec_environment import CybersecEnvironment


@dataclass(frozen=True)
class EpisodeResult:
    """Episode-level evaluation output for a policy run."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    episode_reward: float
    grader_score: float
    grader_success: bool
    terminal_reason: str
    exfiltration_succeeded: bool
    first_detection_step: Optional[int]
    false_positive_actions: int
    invalid_actions: int
    step_records: List[Dict[str, Any]]

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten episode result for CSV/JSON reporting."""
        return {
            "policy_name": self.policy_name,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "grader_score": self.grader_score,
            "grader_success": self.grader_success,
            "terminal_reason": self.terminal_reason,
            "exfiltration_succeeded": self.exfiltration_succeeded,
            "first_detection_step": self.first_detection_step,
            "false_positive_actions": self.false_positive_actions,
            "invalid_actions": self.invalid_actions,
        }


def run_episode(
    *,
    policy: DefenderPolicy,
    scenario_id: str,
    seed: int,
    horizon: Optional[int] = None,
    max_steps_guard: int = 256,
) -> EpisodeResult:
    """Run one full episode with a given policy."""

    env = CybersecEnvironment(
        scenario_id=scenario_id,
        seed=seed,
        horizon=horizon,
    )
    observation = env.reset(seed=seed)
    policy.reset_episode(observation)

    done = bool(observation.done)
    steps = 0
    total_reward = 0.0
    first_detection_step: Optional[int] = None
    false_positive_actions = 0
    invalid_actions = 0
    step_records: List[Dict[str, Any]] = []

    while not done and steps < max_steps_guard:
        action = policy.act(observation)
        observation = env.step(action)
        done = bool(observation.done)
        steps += 1

        reward = float(observation.reward or 0.0)
        total_reward += reward

        info = dict(observation.info)
        invalid_action = info.get("invalid_action")
        if invalid_action is not None:
            invalid_actions += 1

        breakdown = dict(info.get("reward_breakdown", {}) or {})
        false_positive_penalty = float(breakdown.get("false_positive_penalty", 0.0))
        if false_positive_penalty > 0.0:
            false_positive_actions += 1

        if first_detection_step is None:
            has_detection = bool(observation.known_compromised_assets) or bool(
                observation.known_compromised_identities
            )
            if has_detection:
                first_detection_step = steps

        step_records.append(
            {
                "step": steps,
                "action_type": action.action_type,
                "target": action.target,
                "parameter": action.parameter,
                "urgency": action.urgency,
                "reward": reward,
                "done": done,
                "enterprise_risk_score": float(observation.enterprise_risk_score),
                "invalid_action": invalid_action,
                "false_positive_penalty": false_positive_penalty,
                "known_compromised_assets": list(observation.known_compromised_assets),
                "known_compromised_identities": list(
                    observation.known_compromised_identities
                ),
            }
        )

    terminal_info = dict(observation.info)
    terminal_reason = str(terminal_info.get("terminal_reason", "max_steps_guard"))
    grader_score = float(terminal_info.get("grader_score", 0.0))
    grader_success = bool(terminal_info.get("grader_success", False))

    return EpisodeResult(
        policy_name=policy.name,
        scenario_id=scenario_id,
        seed=seed,
        steps=steps,
        episode_reward=total_reward,
        grader_score=grader_score,
        grader_success=grader_success,
        terminal_reason=terminal_reason,
        exfiltration_succeeded=terminal_reason == "adversary_exfiltration",
        first_detection_step=first_detection_step,
        false_positive_actions=false_positive_actions,
        invalid_actions=invalid_actions,
        step_records=step_records,
    )


def aggregate_results(results: List[EpisodeResult]) -> Dict[str, Any]:
    """Aggregate episode outputs into summary statistics."""
    if not results:
        return {
            "episodes": 0,
            "mean_episode_reward": 0.0,
            "mean_steps": 0.0,
            "mean_grader_score": 0.0,
            "success_rate": 0.0,
            "exfiltration_rate": 0.0,
            "mean_first_detection_step": None,
            "invalid_action_rate_per_step": 0.0,
            "false_positive_action_rate_per_step": 0.0,
        }

    episode_count = len(results)
    total_steps = sum(result.steps for result in results)
    detection_steps = [
        result.first_detection_step
        for result in results
        if result.first_detection_step is not None
    ]

    summary = {
        "episodes": episode_count,
        "mean_episode_reward": mean(result.episode_reward for result in results),
        "mean_steps": mean(result.steps for result in results),
        "mean_grader_score": mean(result.grader_score for result in results),
        "success_rate": mean(
            1.0 if result.grader_success else 0.0 for result in results
        ),
        "exfiltration_rate": mean(
            1.0 if result.exfiltration_succeeded else 0.0 for result in results
        ),
        "mean_first_detection_step": mean(detection_steps) if detection_steps else None,
        "invalid_action_rate_per_step": (
            sum(result.invalid_actions for result in results) / total_steps
            if total_steps > 0
            else 0.0
        ),
        "false_positive_action_rate_per_step": (
            sum(result.false_positive_actions for result in results) / total_steps
            if total_steps > 0
            else 0.0
        ),
    }
    return summary
