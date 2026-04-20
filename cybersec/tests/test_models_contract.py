from __future__ import annotations

import pytest

try:
    from cybersec.models import CybersecAction
except ModuleNotFoundError:
    from models import CybersecAction


def test_monitor_action_disallows_target_and_parameter() -> None:
    with pytest.raises(ValueError):
        CybersecAction(action_type="MONITOR", target="anything")

    with pytest.raises(ValueError):
        CybersecAction(action_type="MONITOR", parameter="cloud")


def test_query_logs_requires_supported_source() -> None:
    with pytest.raises(ValueError):
        CybersecAction(action_type="QUERY_LOGS")

    with pytest.raises(ValueError):
        CybersecAction(action_type="QUERY_LOGS", parameter="unsupported")

    valid = CybersecAction(action_type="QUERY_LOGS", parameter="cloud")
    assert valid.parameter == "cloud"


def test_open_ticket_requires_target_and_action_parameter() -> None:
    with pytest.raises(ValueError):
        CybersecAction(action_type="OPEN_TICKET", parameter="ISOLATE_ASSET")

    with pytest.raises(ValueError):
        CybersecAction(action_type="OPEN_TICKET", target="asset-1", parameter="MONITOR")

    valid = CybersecAction(
        action_type="OPEN_TICKET",
        target="asset-1",
        parameter="ISOLATE_ASSET",
        urgency="high",
    )
    assert valid.target == "asset-1"
    assert valid.parameter == "ISOLATE_ASSET"


def test_target_bound_actions_require_target_only() -> None:
    with pytest.raises(ValueError):
        CybersecAction(action_type="TRIAGE_ALERT")

    with pytest.raises(ValueError):
        CybersecAction(
            action_type="TRIAGE_ALERT",
            target="ALT-0001",
            parameter="extra",
        )

    valid = CybersecAction(action_type="TRIAGE_ALERT", target="ALT-0001")
    assert valid.target == "ALT-0001"
