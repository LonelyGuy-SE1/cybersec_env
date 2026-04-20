"""Scenario definitions and loading utilities for CybersecEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Tuple

AssetDomain = Literal["endpoint", "code", "cloud", "data", "network", "saas"]
Criticality = Literal["low", "medium", "high", "critical"]
Privilege = Literal["user", "service", "admin"]
TargetType = Literal["asset", "identity"]
LogSource = Literal["identity", "endpoint", "network", "code", "cloud", "ticketing"]


@dataclass(frozen=True)
class AssetTemplate:
    asset_id: str
    domain: AssetDomain
    criticality: Criticality
    description: str
    egress_point: bool = False


@dataclass(frozen=True)
class IdentityTemplate:
    identity_id: str
    privilege: Privilege
    role: str


@dataclass(frozen=True)
class AttackStepDefinition:
    step_id: str
    stage: str
    source: LogSource
    description: str
    target_type: TargetType
    target_id: str
    required_assets: Tuple[str, ...] = ()
    required_identities: Tuple[str, ...] = ()
    success_probability: float = 0.75
    patch_resistance: float = 0.45
    detection_strength: float = 0.55
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    compromise_asset: bool = False
    compromise_identity: bool = False
    exfiltration_step: bool = False
    progress_weight: float = 1.0


@dataclass(frozen=True)
class AttackPathDefinition:
    path_id: str
    label: str
    steps: Tuple[AttackStepDefinition, ...]


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    title: str
    objective: str
    horizon: int
    assets: Tuple[AssetTemplate, ...]
    identities: Tuple[IdentityTemplate, ...]
    attack_paths: Tuple[AttackPathDefinition, ...]
    false_positive_rate: float = 0.08
    max_alerts: int = 30
    evaluation_focus_assets: Tuple[str, ...] = ()


def _supply_chain_token_drift() -> ScenarioDefinition:
    assets = (
        AssetTemplate(
            asset_id="ws-dev-12",
            domain="endpoint",
            criticality="medium",
            description="Developer workstation in engineering subnet",
        ),
        AssetTemplate(
            asset_id="ci-runner-a",
            domain="code",
            criticality="high",
            description="CI runner used for container and artifact builds",
        ),
        AssetTemplate(
            asset_id="artifact-registry",
            domain="code",
            criticality="high",
            description="Signed artifact registry feeding production workloads",
        ),
        AssetTemplate(
            asset_id="staging-api",
            domain="cloud",
            criticality="high",
            description="Staging API cluster with broad deployment permissions",
        ),
        AssetTemplate(
            asset_id="prod-api",
            domain="cloud",
            criticality="critical",
            description="Production API service mesh control plane",
        ),
        AssetTemplate(
            asset_id="customer-db",
            domain="data",
            criticality="critical",
            description="Customer profile and billing records",
        ),
        AssetTemplate(
            asset_id="edge-egress",
            domain="network",
            criticality="high",
            description="Outbound edge gateway used by backend services",
            egress_point=True,
        ),
    )

    identities = (
        IdentityTemplate(
            identity_id="dev_alex",
            privilege="user",
            role="application developer",
        ),
        IdentityTemplate(
            identity_id="svc_ci_runner",
            privilege="service",
            role="CI automation service account",
        ),
        IdentityTemplate(
            identity_id="sre_prod_admin",
            privilege="admin",
            role="on-call SRE production administrator",
        ),
    )

    primary_path = AttackPathDefinition(
        path_id="path_supply_chain",
        label="Supply-chain compromise path",
        steps=(
            AttackStepDefinition(
                step_id="phish_dev",
                stage="initial_access",
                source="endpoint",
                description="Spear phishing steals session from developer workstation",
                target_type="asset",
                target_id="ws-dev-12",
                success_probability=0.78,
                detection_strength=0.48,
                severity="medium",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="steal_ci_token",
                stage="credential_access",
                source="identity",
                description="Malware extracts CI runner token from local credentials",
                target_type="identity",
                target_id="svc_ci_runner",
                required_assets=("ws-dev-12",),
                success_probability=0.72,
                detection_strength=0.52,
                severity="high",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="takeover_ci_runner",
                stage="lateral_movement",
                source="code",
                description="Compromised token pivots into CI runner host",
                target_type="asset",
                target_id="ci-runner-a",
                required_identities=("svc_ci_runner",),
                success_probability=0.66,
                detection_strength=0.62,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="poison_artifacts",
                stage="persistence",
                source="code",
                description="Malicious build pipeline writes trojanized artifacts",
                target_type="asset",
                target_id="artifact-registry",
                required_assets=("ci-runner-a",),
                success_probability=0.64,
                detection_strength=0.58,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="pivot_staging",
                stage="lateral_movement",
                source="cloud",
                description="Trojan deployment compromises staging API cluster",
                target_type="asset",
                target_id="staging-api",
                required_assets=("artifact-registry",),
                success_probability=0.62,
                detection_strength=0.64,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="assume_prod_admin",
                stage="privilege_escalation",
                source="identity",
                description="Abused deployment trust path grants SRE admin session",
                target_type="identity",
                target_id="sre_prod_admin",
                required_assets=("staging-api",),
                success_probability=0.56,
                detection_strength=0.70,
                severity="critical",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="database_collection",
                stage="collection",
                source="cloud",
                description="Admin session reads high-value customer tables",
                target_type="asset",
                target_id="customer-db",
                required_identities=("sre_prod_admin",),
                success_probability=0.60,
                detection_strength=0.74,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="exfiltrate",
                stage="exfiltration",
                source="network",
                description="Data staged in customer DB is exfiltrated through edge gateway",
                target_type="asset",
                target_id="edge-egress",
                required_assets=("customer-db",),
                success_probability=0.64,
                detection_strength=0.78,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    fallback_path = AttackPathDefinition(
        path_id="path_direct_cloud",
        label="Direct cloud control-plane abuse",
        steps=(
            AttackStepDefinition(
                step_id="phish_dev_alt",
                stage="initial_access",
                source="endpoint",
                description="Credential theft attempt on developer workstation",
                target_type="asset",
                target_id="ws-dev-12",
                success_probability=0.72,
                detection_strength=0.46,
                severity="medium",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="pivot_prod_api",
                stage="lateral_movement",
                source="cloud",
                description="Session token replay targets production API control plane",
                target_type="asset",
                target_id="prod-api",
                required_assets=("ws-dev-12",),
                success_probability=0.44,
                detection_strength=0.72,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="abuse_prod_api_for_db",
                stage="collection",
                source="cloud",
                description="Compromised control plane reaches customer database",
                target_type="asset",
                target_id="customer-db",
                required_assets=("prod-api",),
                success_probability=0.48,
                detection_strength=0.80,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="direct_exfil",
                stage="exfiltration",
                source="network",
                description="Stolen data exits through edge gateway",
                target_type="asset",
                target_id="edge-egress",
                required_assets=("customer-db",),
                success_probability=0.52,
                detection_strength=0.82,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    return ScenarioDefinition(
        scenario_id="supply_chain_token_drift",
        title="Supply-Chain Token Drift",
        objective=(
            "Detect and contain a staged software supply-chain intrusion before"
            " customer-data exfiltration."
        ),
        horizon=32,
        assets=assets,
        identities=identities,
        attack_paths=(primary_path, fallback_path),
        false_positive_rate=0.08,
        max_alerts=32,
        evaluation_focus_assets=("customer-db", "prod-api", "edge-egress"),
    )


def _federated_identity_takeover() -> ScenarioDefinition:
    assets = (
        AssetTemplate(
            asset_id="idp-core",
            domain="saas",
            criticality="critical",
            description="Identity provider tenant with SSO and MFA policies",
        ),
        AssetTemplate(
            asset_id="helpdesk-portal",
            domain="saas",
            criticality="high",
            description="Support workflow portal with password reset tooling",
        ),
        AssetTemplate(
            asset_id="finance-app",
            domain="cloud",
            criticality="high",
            description="Finance approval application",
        ),
        AssetTemplate(
            asset_id="payroll-lake",
            domain="data",
            criticality="critical",
            description="Payroll data lake with PII and salary history",
        ),
        AssetTemplate(
            asset_id="vpn-gateway",
            domain="network",
            criticality="high",
            description="Remote access VPN gateway",
            egress_point=True,
        ),
    )

    identities = (
        IdentityTemplate(
            identity_id="contractor_jules",
            privilege="user",
            role="third-party contractor",
        ),
        IdentityTemplate(
            identity_id="helpdesk_agent_7",
            privilege="user",
            role="helpdesk identity reset operator",
        ),
        IdentityTemplate(
            identity_id="cloud_admin_ops",
            privilege="admin",
            role="cloud operations administrator",
        ),
    )

    primary_path = AttackPathDefinition(
        path_id="path_helpdesk_social",
        label="Helpdesk social engineering",
        steps=(
            AttackStepDefinition(
                step_id="compromise_contractor",
                stage="initial_access",
                source="identity",
                description="Contractor account token phished via fake policy update",
                target_type="identity",
                target_id="contractor_jules",
                success_probability=0.76,
                detection_strength=0.45,
                severity="medium",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="helpdesk_session_hijack",
                stage="credential_access",
                source="identity",
                description="Compromised contractor lures helpdesk into session handoff",
                target_type="identity",
                target_id="helpdesk_agent_7",
                required_identities=("contractor_jules",),
                success_probability=0.66,
                detection_strength=0.55,
                severity="high",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="idp_policy_tamper",
                stage="privilege_escalation",
                source="cloud",
                description="Helpdesk access weakens conditional-access policy in IdP",
                target_type="asset",
                target_id="idp-core",
                required_identities=("helpdesk_agent_7",),
                success_probability=0.58,
                detection_strength=0.70,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="mint_admin_token",
                stage="privilege_escalation",
                source="identity",
                description="Tampered IdP issues unauthorized cloud admin token",
                target_type="identity",
                target_id="cloud_admin_ops",
                required_assets=("idp-core",),
                success_probability=0.54,
                detection_strength=0.74,
                severity="critical",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="payroll_collection",
                stage="collection",
                source="cloud",
                description="Cloud admin token accesses payroll analytics lake",
                target_type="asset",
                target_id="payroll-lake",
                required_identities=("cloud_admin_ops",),
                success_probability=0.58,
                detection_strength=0.76,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="payroll_exfil",
                stage="exfiltration",
                source="network",
                description="Payroll data leaves via unmanaged VPN egress",
                target_type="asset",
                target_id="vpn-gateway",
                required_assets=("payroll-lake",),
                success_probability=0.56,
                detection_strength=0.82,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    fallback_path = AttackPathDefinition(
        path_id="path_finance_session",
        label="Finance app pivot",
        steps=(
            AttackStepDefinition(
                step_id="contractor_phish_alt",
                stage="initial_access",
                source="identity",
                description="Credential replay attempt on contractor account",
                target_type="identity",
                target_id="contractor_jules",
                success_probability=0.70,
                detection_strength=0.42,
                severity="medium",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="finance_app_takeover",
                stage="lateral_movement",
                source="cloud",
                description="Compromised account accesses finance workflow app",
                target_type="asset",
                target_id="finance-app",
                required_identities=("contractor_jules",),
                success_probability=0.46,
                detection_strength=0.68,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="finance_to_payroll",
                stage="collection",
                source="cloud",
                description="Workflow integration token reaches payroll data lake",
                target_type="asset",
                target_id="payroll-lake",
                required_assets=("finance-app",),
                success_probability=0.44,
                detection_strength=0.74,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="finance_exfil",
                stage="exfiltration",
                source="network",
                description="Collected payroll records leave via VPN gateway",
                target_type="asset",
                target_id="vpn-gateway",
                required_assets=("payroll-lake",),
                success_probability=0.50,
                detection_strength=0.80,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    return ScenarioDefinition(
        scenario_id="federated_identity_takeover",
        title="Federated Identity Takeover",
        objective=(
            "Prevent identity-control-plane abuse from propagating into payroll data"
            " exfiltration."
        ),
        horizon=30,
        assets=assets,
        identities=identities,
        attack_paths=(primary_path, fallback_path),
        false_positive_rate=0.10,
        max_alerts=28,
        evaluation_focus_assets=("idp-core", "payroll-lake", "vpn-gateway"),
    )


def _insider_repo_pivot() -> ScenarioDefinition:
    assets = (
        AssetTemplate(
            asset_id="analyst-laptop",
            domain="endpoint",
            criticality="medium",
            description="Business analyst laptop with production read access",
        ),
        AssetTemplate(
            asset_id="repo-core",
            domain="code",
            criticality="high",
            description="Core monorepo with deployment workflows",
        ),
        AssetTemplate(
            asset_id="secrets-vault",
            domain="cloud",
            criticality="critical",
            description="Central secret manager for service credentials",
        ),
        AssetTemplate(
            asset_id="billing-service",
            domain="cloud",
            criticality="high",
            description="Billing microservice processing payments",
        ),
        AssetTemplate(
            asset_id="warehouse-db",
            domain="data",
            criticality="critical",
            description="Enterprise warehouse with revenue and customer exports",
        ),
        AssetTemplate(
            asset_id="outbound-proxy",
            domain="network",
            criticality="high",
            description="Controlled outbound traffic proxy",
            egress_point=True,
        ),
    )

    identities = (
        IdentityTemplate(
            identity_id="analyst_mina",
            privilege="user",
            role="business analyst",
        ),
        IdentityTemplate(
            identity_id="repo_bot",
            privilege="service",
            role="repository automation bot",
        ),
        IdentityTemplate(
            identity_id="db_admin",
            privilege="admin",
            role="database admin account",
        ),
    )

    primary_path = AttackPathDefinition(
        path_id="path_repo_abuse",
        label="Repository automation abuse",
        steps=(
            AttackStepDefinition(
                step_id="insider_laptop_access",
                stage="initial_access",
                source="endpoint",
                description="Insider leverage on analyst endpoint",
                target_type="asset",
                target_id="analyst-laptop",
                success_probability=0.80,
                detection_strength=0.40,
                severity="medium",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="repo_bot_token_theft",
                stage="credential_access",
                source="code",
                description="Analyst host captures repository bot token",
                target_type="identity",
                target_id="repo_bot",
                required_assets=("analyst-laptop",),
                success_probability=0.68,
                detection_strength=0.56,
                severity="high",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="repo_workflow_tamper",
                stage="persistence",
                source="code",
                description="Bot token edits deployment workflow in core repo",
                target_type="asset",
                target_id="repo-core",
                required_identities=("repo_bot",),
                success_probability=0.60,
                detection_strength=0.64,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="vault_pivot",
                stage="privilege_escalation",
                source="cloud",
                description="Compromised workflow retrieves vault bootstrap secret",
                target_type="asset",
                target_id="secrets-vault",
                required_assets=("repo-core",),
                success_probability=0.56,
                detection_strength=0.70,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="db_admin_token",
                stage="privilege_escalation",
                source="identity",
                description="Vault compromise issues database admin credential",
                target_type="identity",
                target_id="db_admin",
                required_assets=("secrets-vault",),
                success_probability=0.54,
                detection_strength=0.74,
                severity="critical",
                compromise_identity=True,
            ),
            AttackStepDefinition(
                step_id="warehouse_collection",
                stage="collection",
                source="cloud",
                description="Admin credential exports warehouse snapshots",
                target_type="asset",
                target_id="warehouse-db",
                required_identities=("db_admin",),
                success_probability=0.58,
                detection_strength=0.78,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="warehouse_exfil",
                stage="exfiltration",
                source="network",
                description="Warehouse export leaves through outbound proxy",
                target_type="asset",
                target_id="outbound-proxy",
                required_assets=("warehouse-db",),
                success_probability=0.60,
                detection_strength=0.82,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    fallback_path = AttackPathDefinition(
        path_id="path_billing_pivot",
        label="Billing service pivot",
        steps=(
            AttackStepDefinition(
                step_id="insider_access_alt",
                stage="initial_access",
                source="endpoint",
                description="Insider endpoint manipulation attempt",
                target_type="asset",
                target_id="analyst-laptop",
                success_probability=0.74,
                detection_strength=0.42,
                severity="medium",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="billing_service_pivot",
                stage="lateral_movement",
                source="cloud",
                description="Compromised endpoint pivots into billing service",
                target_type="asset",
                target_id="billing-service",
                required_assets=("analyst-laptop",),
                success_probability=0.46,
                detection_strength=0.68,
                severity="high",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="billing_to_warehouse",
                stage="collection",
                source="cloud",
                description="Service-to-service trust reaches warehouse database",
                target_type="asset",
                target_id="warehouse-db",
                required_assets=("billing-service",),
                success_probability=0.48,
                detection_strength=0.74,
                severity="critical",
                compromise_asset=True,
            ),
            AttackStepDefinition(
                step_id="billing_exfil",
                stage="exfiltration",
                source="network",
                description="Warehouse extracts leave via outbound proxy",
                target_type="asset",
                target_id="outbound-proxy",
                required_assets=("warehouse-db",),
                success_probability=0.52,
                detection_strength=0.80,
                severity="critical",
                exfiltration_step=True,
                progress_weight=1.5,
            ),
        ),
    )

    return ScenarioDefinition(
        scenario_id="insider_repo_pivot",
        title="Insider Repository Pivot",
        objective=(
            "Contain insider-driven repository and secret-manager abuse before"
            " warehouse data exfiltration."
        ),
        horizon=34,
        assets=assets,
        identities=identities,
        attack_paths=(primary_path, fallback_path),
        false_positive_rate=0.07,
        max_alerts=34,
        evaluation_focus_assets=("secrets-vault", "warehouse-db", "outbound-proxy"),
    )


def scenario_catalog() -> Mapping[str, ScenarioDefinition]:
    """Return all deterministic benchmark scenarios."""
    scenarios = (
        _supply_chain_token_drift(),
        _federated_identity_takeover(),
        _insider_repo_pivot(),
    )
    return {scenario.scenario_id: scenario for scenario in scenarios}


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    """Fetch one scenario by id."""
    catalog: Dict[str, ScenarioDefinition] = dict(scenario_catalog())
    if scenario_id not in catalog:
        valid = ", ".join(sorted(catalog))
        raise ValueError(
            f"Unknown scenario_id '{scenario_id}'. Valid scenarios: {valid}"
        )
    return catalog[scenario_id]


def list_scenarios() -> Tuple[str, ...]:
    """List supported scenario ids."""
    return tuple(sorted(scenario_catalog().keys()))
