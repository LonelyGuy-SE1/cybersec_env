"""MITRE ATT&CK-aligned scenario definitions.

Each scenario is a deterministic specification of:

  * the enterprise topology (assets + identities)
  * an ordered DAG of attack stages, each tagged with a MITRE tactic / technique
  * stochastic dwell times, success probabilities, and detection strengths
  * the maximum episode horizon and background false-positive rate

The scenarios are intentionally compact (5-7 assets, 4-5 identities, 5-7
stages, horizon ~70) so a small LLM can both reason about the topology and
fit it into a 2-3k token prompt.

The DAG is the source of long-horizon planning: a stage can fire only after
every stage in ``prereq_stages`` has been completed, and each stage idles for
a stochastic dwell time before completing. Detection alerts are emitted with
a separate randomized lag, so the defender never sees the attack the moment
it actually progresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Topology primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetTemplate:
    """Static definition of an asset participating in a scenario."""

    asset_id: str
    kind: str  # "service", "endpoint", "datastore", "identity_provider", "repo"
    criticality: float  # 0..1, higher = more painful to isolate / lose
    segment: str  # network/logical zone, used for blast-radius reasoning
    description: str = ""


@dataclass(frozen=True)
class IdentityTemplate:
    """Static definition of an identity (user / service principal)."""

    identity_id: str
    role: str
    privilege: float  # 0..1, scales how dangerous a compromise of this id is
    description: str = ""


@dataclass(frozen=True)
class AttackStage:
    """One stage in the attacker's DAG.

    The full life-cycle of a stage is:

        idle -> in_progress (after prereqs done) -> success/failure (after dwell)

    Fields:
        stage_id: globally-unique within the scenario.
        mitre_tactic / mitre_technique: pretty labels for the storytelling layer.
        prereq_stages: IDs of stages that must succeed before this one can start.
        target_asset / target_identity: optional, used for containment scoring -
            isolating/revoking these things mid-stage prevents the stage.
        dwell_range: (min, max) ticks the stage idles in_progress before resolving.
        success_prob: probability the stage succeeds when its dwell timer fires.
        detection_strength: 0..1 - higher means the stage is louder and more
            likely to surface alerts during its dwell.
        alert_lag_range: (min, max) extra ticks before alerts become visible.
        compromises_asset / compromises_identity: marks ground-truth compromise
            on success; used by the reward model for detection bonuses.
        is_exfil: terminal stage, episode ends in attacker-win on success.
    """

    stage_id: str
    mitre_tactic: str
    mitre_technique: str
    prereq_stages: Tuple[str, ...] = ()
    target_asset: str = ""
    target_identity: str = ""
    dwell_range: Tuple[int, int] = (3, 8)
    success_prob: float = 0.85
    detection_strength: float = 0.5
    alert_lag_range: Tuple[int, int] = (1, 4)
    compromises_asset: bool = False
    compromises_identity: bool = False
    is_exfil: bool = False
    description: str = ""


@dataclass(frozen=True)
class Scenario:
    """A complete scenario ready to be instantiated by the environment."""

    scenario_id: str
    title: str
    summary: str
    horizon: int
    background_alert_rate: float  # per-tick probability of a noise alert
    assets: Tuple[AssetTemplate, ...]
    identities: Tuple[IdentityTemplate, ...]
    stages: Tuple[AttackStage, ...]


# ---------------------------------------------------------------------------
# Scenario 1: Supply-chain CI token drift
# ---------------------------------------------------------------------------


def _supply_chain_token_drift() -> Scenario:
    assets = (
        AssetTemplate("ci-runner-01", "service", 0.6, "build", "GitHub Actions runner"),
        AssetTemplate("artifact-registry", "datastore", 0.85, "build", "Container artifact registry"),
        AssetTemplate("api-gateway", "service", 0.9, "prod", "Public-facing API gateway"),
        AssetTemplate("payments-svc", "service", 0.95, "prod", "Payments microservice"),
        AssetTemplate("warehouse-db", "datastore", 0.95, "data", "Analytics + customer data warehouse"),
        AssetTemplate("egress-proxy", "service", 0.4, "edge", "Outbound HTTPS proxy"),
    )
    identities = (
        IdentityTemplate("svc-ci-deploy", "service-principal", 0.7, "CI deploy bot"),
        IdentityTemplate("svc-payments", "service-principal", 0.9, "Payments service identity"),
        IdentityTemplate("u-platform-eng", "engineer", 0.6, "Platform on-call engineer"),
        IdentityTemplate("u-data-analyst", "analyst", 0.4, "Data analyst with read-only DW"),
    )
    stages = (
        AttackStage(
            stage_id="ta1_token_theft",
            mitre_tactic="TA0001 Initial Access",
            mitre_technique="T1552.004 Private Keys",
            target_asset="ci-runner-01",
            target_identity="svc-ci-deploy",
            dwell_range=(2, 5),
            success_prob=0.95,
            detection_strength=0.25,
            alert_lag_range=(2, 6),
            compromises_identity=True,
            description="Steal long-lived CI deploy token from a misconfigured runner",
        ),
        AttackStage(
            stage_id="ta2_artifact_poison",
            mitre_tactic="TA0002 Execution",
            mitre_technique="T1195.002 Compromise Software Supply Chain",
            prereq_stages=("ta1_token_theft",),
            target_asset="artifact-registry",
            dwell_range=(4, 9),
            success_prob=0.8,
            detection_strength=0.45,
            alert_lag_range=(2, 5),
            compromises_asset=True,
            description="Push a poisoned image to the artifact registry",
        ),
        AttackStage(
            stage_id="ta3_lateral_payments",
            mitre_tactic="TA0008 Lateral Movement",
            mitre_technique="T1210 Exploitation of Remote Services",
            prereq_stages=("ta2_artifact_poison",),
            target_asset="payments-svc",
            target_identity="svc-payments",
            dwell_range=(6, 12),
            success_prob=0.75,
            detection_strength=0.55,
            alert_lag_range=(1, 4),
            compromises_asset=True,
            compromises_identity=True,
            description="Poisoned image lands in payments-svc; pivot to its identity",
        ),
        AttackStage(
            stage_id="ta4_data_collection",
            mitre_tactic="TA0009 Collection",
            mitre_technique="T1213 Data from Information Repositories",
            prereq_stages=("ta3_lateral_payments",),
            target_asset="warehouse-db",
            dwell_range=(5, 10),
            success_prob=0.85,
            detection_strength=0.4,
            alert_lag_range=(2, 6),
            compromises_asset=True,
            description="Use payments identity to query warehouse-db for cardholder data",
        ),
        AttackStage(
            stage_id="ta5_exfil",
            mitre_tactic="TA0010 Exfiltration",
            mitre_technique="T1041 Exfiltration Over C2 Channel",
            prereq_stages=("ta4_data_collection",),
            target_asset="egress-proxy",
            dwell_range=(2, 5),
            success_prob=0.9,
            detection_strength=0.7,
            alert_lag_range=(0, 2),
            is_exfil=True,
            description="Exfiltrate the staged dataset over the egress proxy",
        ),
    )
    return Scenario(
        scenario_id="supply_chain_token_drift",
        title="Supply-chain CI token drift",
        summary=(
            "A long-lived CI deploy token is stolen from a build runner, used "
            "to publish a poisoned artifact, which lands on the payments service "
            "and pivots into the warehouse for staged exfiltration."
        ),
        horizon=70,
        background_alert_rate=0.18,
        assets=assets,
        identities=identities,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Scenario 2: Federated identity takeover
# ---------------------------------------------------------------------------


def _federated_identity_takeover() -> Scenario:
    assets = (
        AssetTemplate("idp-okta", "identity_provider", 0.95, "core", "Federated SSO IdP"),
        AssetTemplate("vpn-gw", "service", 0.85, "edge", "Corporate VPN gateway"),
        AssetTemplate("sales-crm", "service", 0.8, "saas", "Customer relationship management SaaS"),
        AssetTemplate("docs-share", "datastore", 0.7, "saas", "Document collaboration share"),
        AssetTemplate("hr-portal", "service", 0.85, "internal", "HR portal with PII"),
        AssetTemplate("egress-edge", "service", 0.4, "edge", "SaaS egress edge"),
    )
    identities = (
        IdentityTemplate("u-vp-sales", "executive", 0.85, "VP of Sales (target of phish)"),
        IdentityTemplate("u-it-helpdesk", "operator", 0.6, "IT helpdesk operator"),
        IdentityTemplate("u-hr-admin", "admin", 0.8, "HR admin with portal-write"),
        IdentityTemplate("u-finance-controller", "executive", 0.9, "Finance controller"),
        IdentityTemplate("svc-okta-sync", "service-principal", 0.7, "Okta directory sync agent"),
    )
    stages = (
        AttackStage(
            stage_id="ph1_spear_phish",
            mitre_tactic="TA0001 Initial Access",
            mitre_technique="T1566.001 Spearphishing Attachment",
            target_identity="u-vp-sales",
            dwell_range=(3, 7),
            success_prob=0.9,
            detection_strength=0.3,
            alert_lag_range=(2, 5),
            compromises_identity=True,
            description="VP of Sales clicks a credential-harvest attachment",
        ),
        AttackStage(
            stage_id="ph2_mfa_fatigue",
            mitre_tactic="TA0006 Credential Access",
            mitre_technique="T1621 Multi-Factor Authentication Request Generation",
            prereq_stages=("ph1_spear_phish",),
            target_identity="u-vp-sales",
            target_asset="idp-okta",
            dwell_range=(4, 8),
            success_prob=0.7,
            detection_strength=0.5,
            alert_lag_range=(1, 3),
            description="MFA push spam until the VP approves; session token captured",
        ),
        AttackStage(
            stage_id="ph3_helpdesk_pivot",
            mitre_tactic="TA0008 Lateral Movement",
            mitre_technique="T1078.004 Cloud Accounts",
            prereq_stages=("ph2_mfa_fatigue",),
            target_identity="u-it-helpdesk",
            dwell_range=(5, 10),
            success_prob=0.7,
            detection_strength=0.45,
            alert_lag_range=(2, 5),
            compromises_identity=True,
            description="Use VP session to coerce helpdesk into resetting another account",
        ),
        AttackStage(
            stage_id="ph4_hr_collection",
            mitre_tactic="TA0009 Collection",
            mitre_technique="T1213.003 Code Repositories",
            prereq_stages=("ph3_helpdesk_pivot",),
            target_identity="u-hr-admin",
            target_asset="hr-portal",
            dwell_range=(6, 12),
            success_prob=0.8,
            detection_strength=0.4,
            alert_lag_range=(3, 7),
            compromises_asset=True,
            compromises_identity=True,
            description="Reset HR admin password, pull employee PII out of HR portal",
        ),
        AttackStage(
            stage_id="ph5_exfil",
            mitre_tactic="TA0010 Exfiltration",
            mitre_technique="T1567.002 Exfiltration to Cloud Storage",
            prereq_stages=("ph4_hr_collection",),
            target_asset="egress-edge",
            dwell_range=(2, 6),
            success_prob=0.9,
            detection_strength=0.65,
            alert_lag_range=(0, 2),
            is_exfil=True,
            description="Upload employee dump to attacker-controlled cloud bucket",
        ),
    )
    return Scenario(
        scenario_id="federated_identity_takeover",
        title="Federated identity takeover",
        summary=(
            "A spearphish followed by MFA fatigue gives the attacker a VP's "
            "session, which is escalated through the helpdesk into HR-admin "
            "and used to exfiltrate employee PII."
        ),
        horizon=70,
        background_alert_rate=0.22,
        assets=assets,
        identities=identities,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Scenario 3: Insider repo pivot
# ---------------------------------------------------------------------------


def _insider_repo_pivot() -> Scenario:
    assets = (
        AssetTemplate("git-monorepo", "repo", 0.9, "build", "Primary engineering monorepo"),
        AssetTemplate("secrets-vault", "datastore", 0.95, "core", "Application secrets vault"),
        AssetTemplate("staging-cluster", "service", 0.7, "stage", "Staging Kubernetes cluster"),
        AssetTemplate("prod-cluster", "service", 0.95, "prod", "Production Kubernetes cluster"),
        AssetTemplate("customer-db", "datastore", 0.95, "data", "Customer transactional database"),
        AssetTemplate("egress-prod", "service", 0.4, "edge", "Production egress firewall"),
    )
    identities = (
        IdentityTemplate("u-insider-dev", "engineer", 0.7, "Disgruntled engineer with monorepo access"),
        IdentityTemplate("u-staff-eng", "staff-engineer", 0.85, "Staff engineer (write to prod)"),
        IdentityTemplate("svc-vault-reader", "service-principal", 0.8, "Vault read-only role"),
        IdentityTemplate("svc-prod-deploy", "service-principal", 0.95, "Prod deploy role"),
    )
    stages = (
        AttackStage(
            stage_id="ir1_repo_recon",
            mitre_tactic="TA0007 Discovery",
            mitre_technique="T1083 File and Directory Discovery",
            target_asset="git-monorepo",
            target_identity="u-insider-dev",
            dwell_range=(2, 6),
            success_prob=0.95,
            detection_strength=0.2,
            alert_lag_range=(3, 7),
            description="Insider scans the monorepo for hardcoded vault references",
        ),
        AttackStage(
            stage_id="ir2_secret_harvest",
            mitre_tactic="TA0006 Credential Access",
            mitre_technique="T1552.001 Credentials In Files",
            prereq_stages=("ir1_repo_recon",),
            target_asset="secrets-vault",
            target_identity="svc-vault-reader",
            dwell_range=(4, 9),
            success_prob=0.75,
            detection_strength=0.5,
            alert_lag_range=(2, 5),
            compromises_identity=True,
            description="Use leaked SDK config to mint a vault-reader token",
        ),
        AttackStage(
            stage_id="ir3_staging_pivot",
            mitre_tactic="TA0008 Lateral Movement",
            mitre_technique="T1078.004 Cloud Accounts",
            prereq_stages=("ir2_secret_harvest",),
            target_asset="staging-cluster",
            dwell_range=(5, 10),
            success_prob=0.85,
            detection_strength=0.4,
            alert_lag_range=(2, 5),
            compromises_asset=True,
            description="Deploy a malicious sidecar to staging using the vault token",
        ),
        AttackStage(
            stage_id="ir4_prod_pivot",
            mitre_tactic="TA0008 Lateral Movement",
            mitre_technique="T1098 Account Manipulation",
            prereq_stages=("ir3_staging_pivot",),
            target_asset="prod-cluster",
            target_identity="svc-prod-deploy",
            dwell_range=(7, 14),
            success_prob=0.65,
            detection_strength=0.6,
            alert_lag_range=(1, 4),
            compromises_asset=True,
            compromises_identity=True,
            description="Abuse staging-to-prod trust to land in the prod cluster",
        ),
        AttackStage(
            stage_id="ir5_db_collection",
            mitre_tactic="TA0009 Collection",
            mitre_technique="T1005 Data from Local System",
            prereq_stages=("ir4_prod_pivot",),
            target_asset="customer-db",
            dwell_range=(4, 9),
            success_prob=0.85,
            detection_strength=0.5,
            alert_lag_range=(2, 5),
            compromises_asset=True,
            description="Snapshot the customer database to an attacker-controlled volume",
        ),
        AttackStage(
            stage_id="ir6_exfil",
            mitre_tactic="TA0010 Exfiltration",
            mitre_technique="T1048.003 Exfiltration Over Unencrypted Non-C2 Protocol",
            prereq_stages=("ir5_db_collection",),
            target_asset="egress-prod",
            dwell_range=(2, 5),
            success_prob=0.9,
            detection_strength=0.7,
            alert_lag_range=(0, 2),
            is_exfil=True,
            description="Push the customer-db snapshot through the prod egress firewall",
        ),
    )
    return Scenario(
        scenario_id="insider_repo_pivot",
        title="Insider repo pivot",
        summary=(
            "An insider engineer scans the monorepo for hardcoded credentials, "
            "mints a vault token, pivots staging -> prod, and exfiltrates the "
            "customer database."
        ),
        horizon=80,
        background_alert_rate=0.2,
        assets=assets,
        identities=identities,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Scenario 4 (held-out / OOD eval): Cloud metadata SSRF + ransom prep
# ---------------------------------------------------------------------------
#
# This scenario is intentionally *not* in the GRPO training set. Its DAG is
# a different shape from the three training scenarios (no human identity
# pivot at all -- the whole chain is service principals + cloud metadata),
# so a policy that has only memorised "isolate the secrets vault" cannot
# transfer here without actually reading the observation.


def _cloud_metadata_ssrf() -> Scenario:
    assets = (
        AssetTemplate("public-webapp", "service", 0.7, "edge", "Internet-facing web app"),
        AssetTemplate("metadata-svc", "service", 0.95, "core", "Cloud metadata endpoint (169.254.169.254)"),
        AssetTemplate("kms-keyring", "datastore", 0.95, "core", "Cloud KMS key ring"),
        AssetTemplate("backup-bucket", "datastore", 0.85, "data", "Production backup object store"),
        AssetTemplate("egress-cloud", "service", 0.4, "edge", "Cloud egress NAT"),
        AssetTemplate("status-page", "service", 0.3, "edge", "Public status page (decoy)"),
    )
    identities = (
        IdentityTemplate("svc-webapp-role", "service-principal", 0.6, "Public-webapp instance role"),
        IdentityTemplate("svc-backup-role", "service-principal", 0.85, "Backup-job instance role"),
        IdentityTemplate("svc-kms-encryptor", "service-principal", 0.9, "KMS encrypt-only role"),
        IdentityTemplate("svc-statuspage", "service-principal", 0.3, "Status-page deploy role"),
    )
    stages = (
        AttackStage(
            stage_id="cm1_ssrf",
            mitre_tactic="TA0001 Initial Access",
            mitre_technique="T1190 Exploit Public-Facing Application",
            target_asset="public-webapp",
            dwell_range=(2, 5),
            success_prob=0.95,
            detection_strength=0.3,
            alert_lag_range=(2, 5),
            description="SSRF in the public web app pivots to cloud metadata service",
        ),
        AttackStage(
            stage_id="cm2_metadata_creds",
            mitre_tactic="TA0006 Credential Access",
            mitre_technique="T1552.005 Cloud Instance Metadata API",
            prereq_stages=("cm1_ssrf",),
            target_asset="metadata-svc",
            target_identity="svc-webapp-role",
            dwell_range=(3, 7),
            success_prob=0.85,
            detection_strength=0.45,
            alert_lag_range=(2, 5),
            compromises_identity=True,
            description="Mint short-lived AWS/GCP creds for the webapp instance role",
        ),
        AttackStage(
            stage_id="cm3_role_chain",
            mitre_tactic="TA0008 Lateral Movement",
            mitre_technique="T1078.004 Cloud Accounts",
            prereq_stages=("cm2_metadata_creds",),
            target_identity="svc-backup-role",
            dwell_range=(5, 10),
            success_prob=0.65,
            detection_strength=0.5,
            alert_lag_range=(2, 5),
            compromises_identity=True,
            description="AssumeRole-chain from webapp -> backup role",
        ),
        AttackStage(
            stage_id="cm4_kms_replicate",
            mitre_tactic="TA0009 Collection",
            mitre_technique="T1486 Data Encrypted for Impact",
            prereq_stages=("cm3_role_chain",),
            target_asset="kms-keyring",
            target_identity="svc-kms-encryptor",
            dwell_range=(6, 12),
            success_prob=0.7,
            detection_strength=0.55,
            alert_lag_range=(2, 6),
            compromises_asset=True,
            description="Replicate KMS keys to attacker tenant for ransom-note encryption",
        ),
        AttackStage(
            stage_id="cm5_exfil",
            mitre_tactic="TA0010 Exfiltration",
            mitre_technique="T1567.002 Exfiltration to Cloud Storage",
            prereq_stages=("cm4_kms_replicate",),
            target_asset="egress-cloud",
            dwell_range=(2, 5),
            success_prob=0.9,
            detection_strength=0.7,
            alert_lag_range=(0, 2),
            is_exfil=True,
            description="Exfiltrate snapshot of backup-bucket through cloud egress",
        ),
    )
    return Scenario(
        scenario_id="cloud_metadata_ssrf",
        title="Cloud metadata SSRF + ransom prep",
        summary=(
            "An SSRF in a public web app reaches the cloud metadata service, "
            "mints short-lived instance-role credentials, role-chains to the "
            "backup service principal, replicates KMS keys, and exfiltrates "
            "the production backup bucket."
        ),
        horizon=70,
        background_alert_rate=0.15,
        assets=assets,
        identities=identities,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
#
# Three training scenarios + one held-out OOD scenario. Train code paths
# should iterate ``list_train_scenarios()``; eval / generalisation reports
# should iterate ``list_scenarios()`` (which includes the held-out one) or
# ``list_eval_scenarios()`` (only the held-out one).


_TRAIN_FACTORIES: Dict[str, Callable[[], Scenario]] = {
    "supply_chain_token_drift": _supply_chain_token_drift,
    "federated_identity_takeover": _federated_identity_takeover,
    "insider_repo_pivot": _insider_repo_pivot,
}

_EVAL_ONLY_FACTORIES: Dict[str, Callable[[], Scenario]] = {
    "cloud_metadata_ssrf": _cloud_metadata_ssrf,
}

_FACTORIES: Dict[str, Callable[[], Scenario]] = {
    **_TRAIN_FACTORIES,
    **_EVAL_ONLY_FACTORIES,
}


def list_scenarios() -> List[str]:
    """Return every scenario ID (training + held-out) in canonical order."""

    return list(_FACTORIES.keys())


def list_train_scenarios() -> List[str]:
    """Scenario IDs that may appear in the GRPO training dataset."""

    return list(_TRAIN_FACTORIES.keys())


def list_eval_scenarios() -> List[str]:
    """Held-out scenario IDs used only for OOD generalisation reports."""

    return list(_EVAL_ONLY_FACTORIES.keys())


def get_scenario(scenario_id: str) -> Scenario:
    """Look up a scenario by id; raises ``KeyError`` if unknown."""

    if scenario_id not in _FACTORIES:
        raise KeyError(
            f"Unknown scenario_id={scenario_id!r}; choose from {list_scenarios()}"
        )
    return _FACTORIES[scenario_id]()


def scenario_catalog() -> Dict[str, Scenario]:
    """Return a fresh dict of scenario_id -> Scenario for every known scenario."""

    return {sid: factory() for sid, factory in _FACTORIES.items()}


__all__ = [
    "AssetTemplate",
    "IdentityTemplate",
    "AttackStage",
    "Scenario",
    "list_scenarios",
    "list_train_scenarios",
    "list_eval_scenarios",
    "get_scenario",
    "scenario_catalog",
]
