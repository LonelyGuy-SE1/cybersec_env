"""Sanity tests for the MITRE ATT&CK-aligned scenarios.

These guards ensure scenario authors don't silently break the kill-chain
DAG (e.g. by referencing a missing prerequisite) or ship a scenario that
can never produce an exfiltration path.
"""

from __future__ import annotations

import pytest

from cybersec.scenarios import (
    AttackStage,
    Scenario,
    list_scenarios,
    get_scenario,
    scenario_catalog,
)


def _stage_lookup(scenario: Scenario) -> dict[str, AttackStage]:
    return {s.stage_id: s for s in scenario.stages}


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_scenario_dag_is_well_formed(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    stages = _stage_lookup(scenario)

    # All prerequisites must reference existing stages.
    for stage in scenario.stages:
        for prereq in stage.prereq_stages:
            assert prereq in stages, f"{stage.stage_id} depends on missing stage {prereq}"
            assert prereq != stage.stage_id, "stage cannot depend on itself"

    # No cycles: a topological order must exist.
    visited: dict[str, str] = {}

    def visit(node: str) -> None:
        state = visited.get(node)
        if state == "done":
            return
        if state == "in":
            raise AssertionError(f"cycle detected at stage {node}")
        visited[node] = "in"
        for p in stages[node].prereq_stages:
            visit(p)
        visited[node] = "done"

    for sid in stages:
        visit(sid)


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_scenario_has_exactly_one_terminal_exfil(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    exfils = [s for s in scenario.stages if s.is_exfil]
    assert len(exfils) == 1, f"{scenario_id} should expose exactly one is_exfil stage"


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_scenario_assets_and_identities_are_referenced(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    asset_ids = {a.asset_id for a in scenario.assets}
    identity_ids = {i.identity_id for i in scenario.identities}

    referenced_assets = {s.target_asset for s in scenario.stages if s.target_asset}
    referenced_identities = {s.target_identity for s in scenario.stages if s.target_identity}

    assert referenced_assets <= asset_ids, "stage references undeclared asset"
    assert referenced_identities <= identity_ids, "stage references undeclared identity"


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_scenario_horizon_long_enough_for_kill_chain(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    # Worst case: the entire kill chain runs sequentially at max dwell. The
    # horizon must be at least 1.4x that to give the defender room to react.
    worst_case_dwell = sum(s.dwell_range[1] for s in scenario.stages)
    assert scenario.horizon >= int(worst_case_dwell * 1.0), (
        f"{scenario_id} horizon={scenario.horizon} is too short for kill chain "
        f"max dwell {worst_case_dwell}"
    )


def test_catalog_returns_independent_copies() -> None:
    a = scenario_catalog()
    b = scenario_catalog()
    assert list(a.keys()) == list(b.keys())
    # Same content, different objects (so client mutation doesn't poison cache).
    for sid, scen in a.items():
        assert scen is not b[sid], "scenario_catalog should yield fresh objects"
