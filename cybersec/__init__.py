"""Cybersec OpenEnv environment package.

A long-horizon, partially-observable, multi-agent enterprise cyber-defense
environment. The defender (the agent the user trains) faces a scripted
attacker walking a MITRE ATT&CK-aligned stage DAG with stochastic dwell
times and detection delays.

Public surface:

  * :class:`CybersecAction` / :class:`CybersecObservation` - action / observation
  * :class:`CybersecEnvironment`                            - the env subclass
  * :class:`CybersecEnv`                                    - the persistent client
  * :func:`list_scenarios` / :func:`get_scenario`           - scenario catalog
  * :class:`AttackerPersonality`                            - the three archetypes
"""

from .__version__ import __version__
from .client import CybersecEnv
from .server.cybersec_environment import CybersecEnvironment
from .models import (
    ActionType,
    AlertEvent,
    AlertSignal,
    AttackerPersonality,
    CybersecAction,
    CybersecObservation,
    CybersecState,
    ForensicResult,
    RewardBreakdown,
)
from .scenarios import (
    AssetTemplate,
    AttackStage,
    IdentityTemplate,
    Scenario,
    get_scenario,
    list_scenarios,
    scenario_catalog,
)

__all__ = [
    "__version__",
    "CybersecEnv",
    "CybersecEnvironment",
    "ActionType",
    "AlertEvent",
    "AlertSignal",
    "AttackerPersonality",
    "CybersecAction",
    "CybersecObservation",
    "CybersecState",
    "ForensicResult",
    "RewardBreakdown",
    "AssetTemplate",
    "AttackStage",
    "IdentityTemplate",
    "Scenario",
    "get_scenario",
    "list_scenarios",
    "scenario_catalog",
]
