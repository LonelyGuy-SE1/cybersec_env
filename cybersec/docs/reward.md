# Reward Model

CybersecEnv uses dense step reward plus terminal grading.

## Step Signals

`StepSignals` include:

- `detection_gain`
- `containment_gain`
- `disruption_cost`
- `governance_cost`
- `efficiency_cost`
- `adversary_progress_delta`
- `false_positive_penalty`
- `invalid_action_penalty`

## Dense Reward

At each step:

- Positive components:
  - detection
  - containment
  - terminal outcome (only when done)
- Negative components:
  - adversary pressure
  - disruption/governance/efficiency penalties
  - false positives
  - invalid actions

Non-terminal steps include a small time penalty to discourage passive stalling.

## Terminal Grading

`terminal_score` in `[0,1]` combines:

- exfiltration status
- attacker progress
- enterprise risk
- protection of focus assets
- blocked attempts/path switches bonuses

`grader_success = terminal_score >= 0.72`

This score is designed for benchmark plots and comparative evaluation reports.
