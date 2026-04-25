"""Unit tests for ``cybersec.training.rewards``.

These cover every public reward function exposed by
:mod:`cybersec.training.rewards`, including the two anti-collapse rewards
added in iter-2 (:func:`reward_action_diversity` and
:func:`reward_observation_aware`).
"""

from __future__ import annotations

import json
from typing import List

import pytest

from cybersec import CybersecEnvironment
from cybersec.training.rewards import (
    SYSTEM_PROMPT,
    default_reward_funcs,
    parse_first_json_object,
    parsed_action,
    render_observation,
    restore_env,
    reward_action_diversity,
    reward_avoids_exfil_path,
    reward_json_valid,
    reward_no_redundant_containment,
    reward_observation_aware,
    reward_schema_valid,
    reward_step_total,
    reward_target_in_valid_targets,
    snapshot_env,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_parse_first_json_object_handles_chatter():
    text = 'Sure! Here is your action: {"action_type": "MONITOR"}. Hope it helps.'
    parsed = parse_first_json_object(text)
    assert parsed == {"action_type": "MONITOR"}


def test_parse_first_json_object_returns_none_on_garbage():
    assert parse_first_json_object("no json here") is None
    assert parse_first_json_object("") is None
    assert parse_first_json_object("{not json") is None


def test_parsed_action_validates_schema():
    a = parsed_action('{"action_type": "ISOLATE_ASSET", "target": "ci-runner-01"}')
    assert a is not None
    assert a.action_type.value == "ISOLATE_ASSET"
    assert a.target == "ci-runner-01"


def test_parsed_action_rejects_unknown_type():
    assert parsed_action('{"action_type": "NUKE_FROM_ORBIT"}') is None


# ---------------------------------------------------------------------------
# Format / schema rewards
# ---------------------------------------------------------------------------


def test_reward_json_valid_binary():
    out = reward_json_valid(
        prompts=["p", "p", "p"],
        completions=[
            '{"action_type": "MONITOR"}',
            "no json",
            '{"action_type": "MONITOR"',  # unbalanced
        ],
    )
    assert out == [1.0, 0.0, 0.0]


def test_reward_schema_valid_binary():
    out = reward_schema_valid(
        prompts=["p", "p"],
        completions=[
            '{"action_type": "MONITOR"}',
            '{"action_type": "WHATEVER"}',
        ],
    )
    assert out == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Target-validity reward
# ---------------------------------------------------------------------------


def test_reward_target_in_valid_targets_per_action_type():
    completions = [
        '{"action_type": "MONITOR"}',                            # always 1.0
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',  # in valid_assets -> 1
        '{"action_type": "ISOLATE_ASSET", "target": "asset-z"}',  # not in       -> 0
        '{"action_type": "REVOKE_IDENTITY", "target": "u-x"}',    # in valid_ids -> 1
        '{"action_type": "INVESTIGATE", "target": "asset-a"}',    # in either   -> 1
    ]
    valid_assets = [["asset-a"]] * 5
    valid_identities = [["u-x"]] * 5
    out = reward_target_in_valid_targets(
        prompts=["p"] * 5,
        completions=completions,
        valid_assets=valid_assets,
        valid_identities=valid_identities,
    )
    assert out == [1.0, 1.0, 0.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Redundancy reward
# ---------------------------------------------------------------------------


def test_reward_no_redundant_containment_penalises_duplicate():
    completions = [
        '{"action_type": "MONITOR"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',  # already isolated -> 0
        '{"action_type": "ISOLATE_ASSET", "target": "asset-b"}',  # fresh           -> 1
    ]
    isolated_assets = [["asset-a"], ["asset-a"], ["asset-a"]]
    out = reward_no_redundant_containment(
        prompts=["p"] * 3,
        completions=completions,
        isolated_assets=isolated_assets,
    )
    assert out == [0.5, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Env-coupled reward (uses a real cloned env)
# ---------------------------------------------------------------------------


def test_reward_step_total_uses_real_env():
    env = CybersecEnvironment()
    env.reset(seed=0, scenario_id="supply_chain_token_drift")
    blob = snapshot_env(env)
    completions = [
        '{"action_type": "MONITOR"}',
        '{"action_type": "ISOLATE_ASSET", "target": "ci-runner-01"}',
        "garbage",  # unparseable -> -1.0
    ]
    out = reward_step_total(
        prompts=["p"] * 3,
        completions=completions,
        env_snapshot=[blob, blob, blob],
    )
    assert len(out) == 3
    assert all(-2.0 <= r <= 2.0 for r in out)
    assert out[2] == -1.0


def test_reward_step_total_handles_missing_snapshot():
    out = reward_step_total(
        prompts=["p"],
        completions=['{"action_type": "MONITOR"}'],
        env_snapshot=[None],
    )
    assert out == [-1.0]


# ---------------------------------------------------------------------------
# Exfil-path shaper
# ---------------------------------------------------------------------------


def test_reward_avoids_exfil_path_active_when_alerts_present():
    completions = [
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',  # alerts > 0   -> 0.5
        '{"action_type": "INVESTIGATE", "target": "asset-a"}',    # alerts > 0   -> 0.3
        '{"action_type": "MONITOR"}',                              # alerts > 0   -> 0
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',  # alerts == 0  -> 0
    ]
    out = reward_avoids_exfil_path(
        prompts=["p"] * 4,
        completions=completions,
        alert_count=[3, 3, 3, 0],
    )
    assert out == [0.5, 0.3, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Anti-collapse rewards (iter-2)
# ---------------------------------------------------------------------------


def test_reward_action_diversity_rewards_unique_within_group():
    """4 candidates for one prompt: 3 identical isolations, 1 unique monitor.

    The 3 duplicates each get 1/3, the unique one gets 1.0.
    """

    completions = [
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
        '{"action_type": "MONITOR"}',
    ]
    out = reward_action_diversity(prompts=["P"] * 4, completions=completions)
    assert pytest.approx(out[0], rel=1e-6) == 1 / 3
    assert pytest.approx(out[1], rel=1e-6) == 1 / 3
    assert pytest.approx(out[2], rel=1e-6) == 1 / 3
    assert out[3] == 1.0


def test_reward_action_diversity_groups_by_prompt():
    """Two prompts, each with two duplicate completions -> all get 0.5."""

    completions = [
        '{"action_type": "MONITOR"}',
        '{"action_type": "MONITOR"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
    ]
    prompts = ["P1", "P1", "P2", "P2"]
    out = reward_action_diversity(prompts=prompts, completions=completions)
    assert out == [0.5, 0.5, 0.5, 0.5]


def test_reward_action_diversity_unparseable_zero():
    out = reward_action_diversity(
        prompts=["P"] * 2,
        completions=["garbage", '{"action_type": "MONITOR"}'],
    )
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_reward_observation_aware_alerts_present():
    """Containment with alerts -> 1.0, MONITOR with alerts -> 0.0."""

    completions = [
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
        '{"action_type": "MONITOR"}',
    ]
    out = reward_observation_aware(
        prompts=["p"] * 2,
        completions=completions,
        valid_assets=[["asset-a"]] * 2,
        alert_count=[3, 3],
    )
    assert out == [1.0, 0.0]


def test_reward_observation_aware_no_alerts_present():
    """No alerts -> MONITOR is preferred (0.7 > 0.4)."""

    completions = [
        '{"action_type": "MONITOR"}',
        '{"action_type": "ISOLATE_ASSET", "target": "asset-a"}',
    ]
    out = reward_observation_aware(
        prompts=["p"] * 2,
        completions=completions,
        valid_assets=[["asset-a"]] * 2,
        alert_count=[0, 0],
    )
    assert out == [0.7, 0.4]


def test_reward_observation_aware_invalid_target_zero():
    out = reward_observation_aware(
        prompts=["p"],
        completions=['{"action_type": "ISOLATE_ASSET", "target": "asset-z"}'],
        valid_assets=[["asset-a"]],
        alert_count=[3],
    )
    assert out == [0.0]


# ---------------------------------------------------------------------------
# Bundle / public surface
# ---------------------------------------------------------------------------


def test_default_reward_funcs_is_stable_order():
    funcs = default_reward_funcs()
    names = [f.__name__ for f in funcs]
    assert names == [
        "reward_json_valid",
        "reward_schema_valid",
        "reward_target_in_valid_targets",
        "reward_no_redundant_containment",
        "reward_step_total",
        "reward_avoids_exfil_path",
        "reward_action_diversity",
        "reward_observation_aware",
        "reward_batch_action_entropy",
    ]


def test_render_observation_includes_telemetry_timeline(env):
    obs = env.reset(seed=0, scenario_id="supply_chain_token_drift")
    text = render_observation(obs)
    assert "tick=" in text
    assert "valid_targets=" in text
    assert "recent_alerts:" in text
    assert "recent_forensics:" in text


def test_snapshot_restore_round_trip(env):
    env.reset(seed=42, scenario_id="federated_identity_takeover")
    blob = snapshot_env(env)
    env2 = restore_env(blob)
    assert env2.state.scenario_id == env.state.scenario_id
    assert env2.state.tick == env.state.tick


def test_system_prompt_lists_all_action_types():
    for name in (
        "MONITOR",
        "INVESTIGATE",
        "ISOLATE_ASSET",
        "REVOKE_IDENTITY",
        "BLOCK_EGRESS",
        "PATCH_ASSET",
    ):
        assert name in SYSTEM_PROMPT
