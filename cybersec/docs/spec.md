# CybersecEnv Specification

## Design Goal

CybersecEnv is a partially observable, long-horizon enterprise defense environment that trains an agent to
contain a hidden attacker campaign under operational and governance constraints.

## POMDP Framing

- **Hidden state**: true compromised entities, attacker path/step, exfiltration readiness.
- **Observation**: noisy alerts, tickets, forensics updates, and limited evidence scores.
- **Action space**: monitoring, evidence gathering, triage, workflow ticketing, and direct controls.
- **Transition**: defender actions, workflow actor updates, attacker progression, telemetry generation.
- **Reward**: decomposed dense reward plus terminal grading score.

## Multi-Agent Dynamics

### Defender (controllable)

The learning policy issues `CybersecAction` operations each step.

### Attacker (simulated)

- Maintains an active attack path and step index.
- Evaluates preconditions and defender control pressure.
- Can switch attack paths when blocked repeatedly.

### Enterprise workflow actor (simulated)

- Reviews and approves/rejects requested control tickets.
- Enforces delayed execution windows.
- Produces ticket updates that influence defender timing decisions.

## Deterministic Scenario Contract

Scenarios are deterministic given `(scenario_id, seed)` and define:

- enterprise assets and identities
- attack paths (>=2)
- step-wise success probabilities and detection strengths
- horizon and telemetry noise level

## Terminal Conditions

Episode terminates when either:

- horizon reached, or
- adversary exfiltration succeeds

## Output Contract

`CybersecObservation.info` includes machine-readable evaluation fields:

- `reward_breakdown`
- `attacker_event`
- `attacker_summary`
- `events`
- terminal-only: `grader_score`, `grader_success`, `terminal_reason`
