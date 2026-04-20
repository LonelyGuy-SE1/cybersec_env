from __future__ import annotations

try:
    from cybersec.server.scenario_loader import get_scenario, list_scenarios
except ModuleNotFoundError:
    from server.scenario_loader import get_scenario, list_scenarios


def test_scenario_catalog_has_expected_entries() -> None:
    names = list_scenarios()
    assert "supply_chain_token_drift" in names
    assert "federated_identity_takeover" in names
    assert "insider_repo_pivot" in names


def test_each_scenario_has_multiple_paths_and_exfiltration_step() -> None:
    for scenario_id in list_scenarios():
        scenario = get_scenario(scenario_id)
        assert len(scenario.attack_paths) >= 2

        has_exfil = False
        for path in scenario.attack_paths:
            for step in path.steps:
                if step.exfiltration_step:
                    has_exfil = True
        assert has_exfil is True


def test_scenario_lookup_raises_for_unknown_id() -> None:
    try:
        get_scenario("unknown_scenario")
    except ValueError as exc:
        assert "Unknown scenario_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown scenario id")
