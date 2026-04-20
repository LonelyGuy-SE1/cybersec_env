from __future__ import annotations

from baselines import HeuristicPolicy, RandomPolicy, aggregate_results, run_episode


def test_run_episode_returns_terminal_metrics() -> None:
    result = run_episode(
        policy=RandomPolicy(seed=1),
        scenario_id="supply_chain_token_drift",
        seed=11,
        horizon=8,
    )

    assert result.steps > 0
    assert result.terminal_reason in {"horizon_reached", "adversary_exfiltration"}
    assert 0.0 <= result.grader_score <= 1.0
    assert isinstance(result.grader_success, bool)
    assert isinstance(result.step_records, list)


def test_aggregate_results_returns_expected_fields() -> None:
    results = [
        run_episode(
            policy=HeuristicPolicy(),
            scenario_id="federated_identity_takeover",
            seed=21,
            horizon=6,
        ),
        run_episode(
            policy=RandomPolicy(seed=2),
            scenario_id="federated_identity_takeover",
            seed=22,
            horizon=6,
        ),
    ]

    summary = aggregate_results(results)

    assert summary["episodes"] == 2
    assert "mean_episode_reward" in summary
    assert "mean_grader_score" in summary
    assert "success_rate" in summary
    assert "exfiltration_rate" in summary
