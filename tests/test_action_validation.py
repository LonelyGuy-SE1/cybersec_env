"""Action validation: Pydantic structural checks + live ``valid_targets`` gate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cybersec import ActionType, CybersecAction, CybersecEnvironment


# ---------------------------------------------------------------------------
# Pydantic-level structural validation
# ---------------------------------------------------------------------------


def test_monitor_rejects_target() -> None:
    with pytest.raises(ValidationError):
        CybersecAction(action_type=ActionType.MONITOR, target="any-asset")


@pytest.mark.parametrize(
    "atype",
    [
        ActionType.INVESTIGATE,
        ActionType.ISOLATE_ASSET,
        ActionType.REVOKE_IDENTITY,
        ActionType.BLOCK_EGRESS,
        ActionType.PATCH_ASSET,
    ],
)
def test_actions_requiring_target_reject_missing(atype: ActionType) -> None:
    with pytest.raises(ValidationError):
        CybersecAction(action_type=atype)
    with pytest.raises(ValidationError):
        CybersecAction(action_type=atype, target="")


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CybersecAction(action_type=ActionType.MONITOR, mystery="value")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Live ``valid_targets`` gate
# ---------------------------------------------------------------------------


def test_unknown_asset_target_yields_invalid_action_penalty(env: CybersecEnvironment) -> None:
    env.reset(seed=0, scenario_id="supply_chain_token_drift")
    obs = env.step(CybersecAction(action_type=ActionType.ISOLATE_ASSET, target="ghost-asset"))
    breakdown = obs.info["reward_breakdown"]
    assert breakdown["invalid_action_penalty"] < 0
    assert "ghost-asset" not in obs.isolated_assets


def test_revoke_with_asset_id_is_invalid(env: CybersecEnvironment) -> None:
    obs = env.reset(seed=0, scenario_id="supply_chain_token_drift")
    asset = obs.valid_targets["assets"][0]
    out = env.step(CybersecAction(action_type=ActionType.REVOKE_IDENTITY, target=asset))
    assert out.info["reward_breakdown"]["invalid_action_penalty"] < 0
    assert asset not in out.revoked_identities


def test_legal_action_does_not_emit_invalid_penalty(env: CybersecEnvironment) -> None:
    obs = env.reset(seed=0, scenario_id="supply_chain_token_drift")
    asset = obs.valid_targets["assets"][0]
    out = env.step(CybersecAction(action_type=ActionType.ISOLATE_ASSET, target=asset))
    assert out.info["reward_breakdown"]["invalid_action_penalty"] == 0
    assert asset in out.isolated_assets


def test_repeat_isolation_uses_set_semantics(env: CybersecEnvironment) -> None:
    """Re-issuing the same containment action must not produce duplicate state.

    We pick ``api-gateway`` because it is *not* on any attack stage path and
    therefore does not terminate the supply-chain scenario via preemption,
    letting us run two steps in a row.
    """

    env.reset(seed=0, scenario_id="supply_chain_token_drift")
    target = "api-gateway"
    env.step(CybersecAction(action_type=ActionType.ISOLATE_ASSET, target=target))
    out2 = env.step(CybersecAction(action_type=ActionType.ISOLATE_ASSET, target=target))
    assert out2.isolated_assets.count(target) == 1
    # Re-issuing on a non-attack-path target still costs another false positive.
    assert out2.info["reward_breakdown"]["false_positive_penalty"] < 0
