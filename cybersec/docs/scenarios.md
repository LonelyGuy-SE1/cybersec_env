# Scenario Catalog

## `supply_chain_token_drift`

- Domain: CI/CD + cloud + data exfiltration
- Focus assets: `customer-db`, `prod-api`, `edge-egress`
- Horizon: 32

## `federated_identity_takeover`

- Domain: identity provider abuse + payroll data risk
- Focus assets: `idp-core`, `payroll-lake`, `vpn-gateway`
- Horizon: 30

## `insider_repo_pivot`

- Domain: insider endpoint + repo automation + secret manager abuse
- Focus assets: `secrets-vault`, `warehouse-db`, `outbound-proxy`
- Horizon: 34

## Common Properties

Each scenario provides:

- multiple attack paths (primary + fallback)
- mandatory exfiltration step(s)
- deterministic transition behavior under fixed seed
