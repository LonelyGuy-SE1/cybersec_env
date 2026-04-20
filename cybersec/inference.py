"""Inference runner for CybersecEnv benchmark trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from baselines import HeuristicPolicy, RandomPolicy, run_episode
from client import CybersecEnv
from models import CybersecAction, CybersecObservation
from server.scenario_loader import list_scenarios

BENCHMARK = "cybersec_enterprise_campaign"
DEFAULT_MAX_STEPS = 64

SYSTEM_PROMPT = (
    "You are an enterprise incident-response controller. "
    "Return JSON only with keys: action_type, target (optional), parameter (optional), urgency (optional)."
)


@dataclass
class EpisodeOutcome:
    scenario_id: str
    seed: int
    success: bool
    steps: int
    score: float
    total_reward: float
    terminal_reason: str
    invalid_actions: int
    trace: List[Dict[str, Any]]


def _env_base_url() -> Optional[str]:
    return os.getenv("ENV_BASE_URL")


def _image_name() -> Optional[str]:
    return os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")


def _model_name() -> str:
    return os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _api_base_url() -> str:
    return os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")


def _api_key() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("API_KEY")


def _parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for token in _parse_list(raw):
        values.append(int(token))
    return values


def _build_llm_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is required for --mode llm")
    key = _api_key()
    if not key:
        raise RuntimeError("HF_TOKEN or API_KEY environment variable is required.")
    return OpenAI(base_url=_api_base_url(), api_key=key)


def _probe_llm_proxy(client: OpenAI) -> None:
    try:
        client.chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            temperature=0,
            max_tokens=2,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to query API_BASE_URL with supplied HF_TOKEN/API_KEY."
        ) from exc


def _extract_last_action_error(observation: CybersecObservation) -> Optional[str]:
    info = observation.info
    if not isinstance(info, dict):
        return None
    for key in ("last_action_error", "invalid_action"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_available_actions(observation: CybersecObservation) -> List[str]:
    info = observation.info if isinstance(observation.info, dict) else {}
    raw = info.get("available_actions", [])
    if isinstance(raw, list) and raw:
        return [str(value) for value in raw]

    if observation.available_actions:
        return [str(value) for value in observation.available_actions]

    return []


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _enforce_action_contract(
    observation: CybersecObservation,
    action: CybersecAction,
) -> Optional[CybersecAction]:
    available = set(_extract_available_actions(observation))
    if available and action.action_type not in available:
        return None

    valid = observation.valid_targets

    if action.action_type == "QUERY_LOGS":
        sources = set(valid.get("query_log_sources", []))
        if action.parameter is None or (sources and action.parameter not in sources):
            return None
        return action

    if action.action_type == "OPEN_TICKET":
        ticketable = set(valid.get("ticketable_actions", []))
        target_pool = set(valid.get("assets", [])) | set(valid.get("identities", []))
        if action.parameter is None or (
            ticketable and action.parameter not in ticketable
        ):
            return None
        if action.target is None or (target_pool and action.target not in target_pool):
            return None
        return action

    if action.action_type == "TRIAGE_ALERT":
        alert_ids = set(valid.get("alert_ids", []))
        if action.target is None or (alert_ids and action.target not in alert_ids):
            return None
        return action

    if action.action_type == "EXECUTE_TICKET":
        ready = set(valid.get("ready_ticket_ids", []))
        if action.target is None or (ready and action.target not in ready):
            return None
        return action

    if action.action_type == "REQUEST_FORENSICS":
        target_pool = set(valid.get("assets", [])) | set(valid.get("identities", []))
        pending = set(valid.get("pending_forensics_targets", []))
        if action.target is None or (target_pool and action.target not in target_pool):
            return None
        if action.target in pending:
            return None
        return action

    if action.action_type in {"ISOLATE_ASSET", "BLOCK_EGRESS", "PATCH_ASSET"}:
        assets = set(valid.get("assets", []))
        if action.target is None or (assets and action.target not in assets):
            return None
        return action

    if action.action_type in {"REVOKE_IDENTITY", "ROTATE_SECRET"}:
        identities = set(valid.get("identities", []))
        if action.target is None or (identities and action.target not in identities):
            return None
        return action

    return action


def heuristic_policy(observation: CybersecObservation, _step: int) -> CybersecAction:
    ready_ids = observation.valid_targets.get("ready_ticket_ids", [])
    if ready_ids:
        return CybersecAction(action_type="EXECUTE_TICKET", target=ready_ids[0])

    pending = sorted(
        [a for a in observation.alerts if a.triage_status == "pending"],
        key=lambda a: (a.confidence, _severity_rank(a.severity)),
        reverse=True,
    )
    if pending:
        top = pending[0]
        if top.confidence >= 0.40 or top.severity in {"high", "critical"}:
            return CybersecAction(action_type="TRIAGE_ALERT", target=top.alert_id)

    if (
        observation.enterprise_risk_score >= 0.70
        and "edge-egress" in observation.valid_targets.get("assets", [])
    ):
        return CybersecAction(action_type="BLOCK_EGRESS", target="edge-egress")

    known_assets = observation.known_compromised_assets
    if known_assets:
        target = known_assets[0]
        if not _event_mentions(observation.recent_activity, f"Asset {target} isolated"):
            return CybersecAction(action_type="ISOLATE_ASSET", target=target)
        return CybersecAction(action_type="PATCH_ASSET", target=target)

    known_ids = observation.known_compromised_identities
    if known_ids:
        target = known_ids[0]
        return CybersecAction(action_type="REVOKE_IDENTITY", target=target)

    if pending:
        entity = pending[0].entity
        targets = set(observation.valid_targets.get("assets", [])) | set(
            observation.valid_targets.get("identities", [])
        )
        if entity in targets:
            return CybersecAction(action_type="REQUEST_FORENSICS", target=entity)

    return CybersecAction(
        action_type="QUERY_LOGS", parameter=_best_log_source(observation)
    )


def get_model_action(
    client: OpenAI,
    *,
    step: int,
    observation: CybersecObservation,
    history: List[str],
) -> Optional[CybersecAction]:
    available = _extract_available_actions(observation)
    prompt = {
        "step": step,
        "scenario_id": observation.scenario_id,
        "enterprise_risk_score": observation.enterprise_risk_score,
        "available_actions": available,
        "valid_targets": observation.valid_targets,
        "known_compromised_assets": observation.known_compromised_assets,
        "known_compromised_identities": observation.known_compromised_identities,
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "source": alert.source,
                "severity": alert.severity,
                "confidence": alert.confidence,
                "entity": alert.entity,
                "triage_status": alert.triage_status,
            }
            for alert in observation.alerts
        ],
        "recent_activity": observation.recent_activity[-8:],
        "history": history[-6:],
    }
    try:
        completion = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=True),
                },
            ],
            temperature=0,
            max_tokens=200,
            stream=False,
        )
    except Exception:
        return None

    text = (completion.choices[0].message.content or "").strip()
    data = _safe_json_parse(text)
    if data is None:
        return None

    action_type = str(data.get("action_type", "")).strip().upper()
    if not action_type:
        return None

    payload: Dict[str, Any] = {"action_type": action_type}
    if data.get("target") is not None:
        payload["target"] = str(data.get("target"))
    if data.get("parameter") is not None:
        payload["parameter"] = str(data.get("parameter"))
    if data.get("urgency") is not None:
        payload["urgency"] = str(data.get("urgency")).lower()

    try:
        action = CybersecAction(**payload)
    except Exception:
        return None

    return _enforce_action_contract(observation, action)


def log_start(scenario_id: str, seed: int, model: str, env: str) -> None:
    print(
        f"[START] scenario={scenario_id} seed={seed} benchmark={BENCHMARK} env={env} model={model}",
        flush=True,
    )


def log_step(
    *,
    step: int,
    action: CybersecAction,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_desc = action.action_type
    if action.target:
        action_desc += f" target={action.target}"
    if action.parameter:
        action_desc += f" parameter={action.parameter}"
    if action.action_type == "OPEN_TICKET":
        action_desc += f" urgency={action.urgency}"
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_desc} reward={reward:.4f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(outcome: EpisodeOutcome) -> None:
    print(
        "[END] "
        f"scenario={outcome.scenario_id} seed={outcome.seed} "
        f"success={str(outcome.success).lower()} steps={outcome.steps} "
        f"score={outcome.score:.4f} total_reward={outcome.total_reward:.4f} "
        f"terminal_reason={outcome.terminal_reason} invalid_actions={outcome.invalid_actions}",
        flush=True,
    )


def _severity_rank(value: str) -> int:
    if value == "critical":
        return 4
    if value == "high":
        return 3
    if value == "medium":
        return 2
    return 1


def _best_log_source(observation: CybersecObservation) -> str:
    if any(a.source == "network" for a in observation.alerts):
        return "network"
    if any(a.source == "cloud" for a in observation.alerts):
        return "cloud"
    if any(a.source == "identity" for a in observation.alerts):
        return "identity"
    return "cloud"


def _event_mentions(events: List[str], phrase: str) -> bool:
    needle = phrase.lower()
    for event in events:
        if needle in event.lower():
            return True
    return False


def _run_remote_episode(
    *,
    env: CybersecEnv,
    scenario_id: str,
    seed: int,
    max_steps: int,
    mode: str,
    client: Optional[OpenAI],
) -> EpisodeOutcome:
    sync_env = env.sync()

    history: List[str] = []
    trace: List[Dict[str, Any]] = []
    total_reward = 0.0
    invalid_actions = 0
    score = 0.0
    success = False
    terminal_reason = "max_steps"
    steps_taken = 0

    with sync_env:
        reset_result = sync_env.reset(seed=seed, scenario_id=scenario_id)
        observation = reset_result.observation
        done = bool(reset_result.done)

        for step in range(1, max_steps + 1):
            if done:
                break

            if mode == "heuristic":
                action = heuristic_policy(observation, step)
            elif mode == "llm" and client is not None:
                action = get_model_action(
                    client,
                    step=step,
                    observation=observation,
                    history=history,
                )
                if action is None:
                    action = heuristic_policy(observation, step)
            else:
                action = heuristic_policy(observation, step)

            result = sync_env.step(action)
            observation = result.observation
            done = bool(result.done)
            reward = float(result.reward or 0.0)
            total_reward += reward
            steps_taken = step

            error = _extract_last_action_error(observation)
            if error is not None:
                invalid_actions += 1

            log_step(step=step, action=action, reward=reward, done=done, error=error)
            trace.append(
                {
                    "step": step,
                    "action_type": action.action_type,
                    "target": action.target,
                    "parameter": action.parameter,
                    "urgency": action.urgency,
                    "reward": reward,
                    "done": done,
                    "enterprise_risk_score": observation.enterprise_risk_score,
                    "invalid_action": error,
                }
            )

            history.append(
                f"step={step} action={action.action_type} reward={reward:.4f} error={error or 'null'}"
            )

            if done:
                info = observation.info if isinstance(observation.info, dict) else {}
                terminal_reason = str(info.get("terminal_reason", "unknown"))
                score_raw = info.get("grader_score", 0.0)
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0
                success = bool(info.get("grader_success", False))
                break

    return EpisodeOutcome(
        scenario_id=scenario_id,
        seed=seed,
        success=success,
        steps=steps_taken,
        score=max(0.0, min(1.0, score)),
        total_reward=total_reward,
        terminal_reason=terminal_reason,
        invalid_actions=invalid_actions,
        trace=trace,
    )


def _run_remote_episode_with_sync_client(
    *,
    sync_env: Any,
    scenario_id: str,
    seed: int,
    max_steps: int,
    mode: str,
    client: Optional[OpenAI],
) -> EpisodeOutcome:
    history: List[str] = []
    trace: List[Dict[str, Any]] = []
    total_reward = 0.0
    invalid_actions = 0
    score = 0.0
    success = False
    terminal_reason = "max_steps"
    steps_taken = 0

    reset_result = sync_env.reset(seed=seed, scenario_id=scenario_id)
    observation = reset_result.observation
    done = bool(reset_result.done)

    for step in range(1, max_steps + 1):
        if done:
            break

        if mode == "heuristic":
            action = heuristic_policy(observation, step)
        elif mode == "llm" and client is not None:
            action = get_model_action(
                client,
                step=step,
                observation=observation,
                history=history,
            )
            if action is None:
                action = heuristic_policy(observation, step)
        else:
            action = heuristic_policy(observation, step)

        result = sync_env.step(action)
        observation = result.observation
        done = bool(result.done)
        reward = float(result.reward or 0.0)
        total_reward += reward
        steps_taken = step

        error = _extract_last_action_error(observation)
        if error is not None:
            invalid_actions += 1

        log_step(step=step, action=action, reward=reward, done=done, error=error)
        trace.append(
            {
                "step": step,
                "action_type": action.action_type,
                "target": action.target,
                "parameter": action.parameter,
                "urgency": action.urgency,
                "reward": reward,
                "done": done,
                "enterprise_risk_score": observation.enterprise_risk_score,
                "invalid_action": error,
            }
        )

        history.append(
            f"step={step} action={action.action_type} reward={reward:.4f} error={error or 'null'}"
        )

        if done:
            info = observation.info if isinstance(observation.info, dict) else {}
            terminal_reason = str(info.get("terminal_reason", "unknown"))
            score_raw = info.get("grader_score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            success = bool(info.get("grader_success", False))
            break

    return EpisodeOutcome(
        scenario_id=scenario_id,
        seed=seed,
        success=success,
        steps=steps_taken,
        score=max(0.0, min(1.0, score)),
        total_reward=total_reward,
        terminal_reason=terminal_reason,
        invalid_actions=invalid_actions,
        trace=trace,
    )


def _run_local_baseline_episode(
    *,
    scenario_id: str,
    seed: int,
    max_steps: int,
    mode: str,
) -> EpisodeOutcome:
    if mode == "random":
        policy = RandomPolicy(seed=seed)
    else:
        policy = HeuristicPolicy()

    result = run_episode(
        policy=policy,
        scenario_id=scenario_id,
        seed=seed,
        max_steps_guard=max_steps,
    )
    return EpisodeOutcome(
        scenario_id=result.scenario_id,
        seed=result.seed,
        success=result.grader_success,
        steps=result.steps,
        score=result.grader_score,
        total_reward=result.episode_reward,
        terminal_reason=result.terminal_reason,
        invalid_actions=result.invalid_actions,
        trace=result.step_records,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CybersecEnv inference episodes")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "random", "llm"],
        default=os.getenv("INFERENCE_MODE", "heuristic"),
    )
    parser.add_argument(
        "--env",
        choices=["local", "remote"],
        default=os.getenv("INFERENCE_ENV", "local"),
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=os.getenv("CYBERSEC_SCENARIOS", ",".join(list_scenarios())),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=os.getenv("CYBERSEC_SEEDS", "101,202,303"),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("MAX_STEPS", str(DEFAULT_MAX_STEPS))),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("INFERENCE_OUTPUT", "outputs/evals/inference_results.json"),
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()
    scenarios = _parse_list(args.scenarios)
    seeds = _parse_int_list(args.seeds)
    if not scenarios:
        raise RuntimeError("No scenarios provided")
    if not seeds:
        raise RuntimeError("No seeds provided")
    if args.max_steps <= 0:
        raise RuntimeError("--max-steps must be positive")

    client: Optional[OpenAI] = None
    if args.mode == "llm":
        client = _build_llm_client()
        _probe_llm_proxy(client)

    outcomes: List[EpisodeOutcome] = []

    if args.env == "remote":
        env_base_url = _env_base_url()
        if env_base_url:
            env = CybersecEnv(base_url=env_base_url)
            sync_env = env.sync()
            with sync_env:
                for scenario_id in scenarios:
                    for seed in seeds:
                        log_start(
                            scenario_id=scenario_id,
                            seed=seed,
                            model=_model_name() if args.mode == "llm" else args.mode,
                            env=args.env,
                        )
                        outcome = _run_remote_episode_with_sync_client(
                            sync_env=sync_env,
                            scenario_id=scenario_id,
                            seed=seed,
                            max_steps=args.max_steps,
                            mode=args.mode,
                            client=client,
                        )
                        outcomes.append(outcome)
                        log_end(outcome)
        else:
            image = _image_name()
            if not image:
                raise RuntimeError(
                    "Set ENV_BASE_URL or IMAGE_NAME/LOCAL_IMAGE_NAME for --env remote"
                )
            env = CybersecEnv.from_docker_image(image)
            for scenario_id in scenarios:
                for seed in seeds:
                    log_start(
                        scenario_id=scenario_id,
                        seed=seed,
                        model=_model_name() if args.mode == "llm" else args.mode,
                        env=args.env,
                    )
                    outcome = _run_remote_episode(
                        env=env,
                        scenario_id=scenario_id,
                        seed=seed,
                        max_steps=args.max_steps,
                        mode=args.mode,
                        client=client,
                    )
                    outcomes.append(outcome)
                    log_end(outcome)
            env.close()
    else:
        for scenario_id in scenarios:
            for seed in seeds:
                log_start(
                    scenario_id=scenario_id,
                    seed=seed,
                    model=_model_name() if args.mode == "llm" else args.mode,
                    env=args.env,
                )
                outcome = _run_local_baseline_episode(
                    scenario_id=scenario_id,
                    seed=seed,
                    max_steps=args.max_steps,
                    mode=args.mode,
                )
                outcomes.append(outcome)
                log_end(outcome)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": BENCHMARK,
        "mode": args.mode,
        "env": args.env,
        "scenarios": scenarios,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "episodes": [
            {
                "scenario_id": outcome.scenario_id,
                "seed": outcome.seed,
                "success": outcome.success,
                "steps": outcome.steps,
                "score": outcome.score,
                "total_reward": outcome.total_reward,
                "terminal_reason": outcome.terminal_reason,
                "invalid_actions": outcome.invalid_actions,
            }
            for outcome in outcomes
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RESULT] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
