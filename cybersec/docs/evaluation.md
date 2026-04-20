# Evaluation Protocol

## Deterministic Benchmarking

Use fixed seeds and scenario ids to compare policies:

1. Random baseline
2. Rule-based heuristic baseline
3. Trained policy

Collect:

- average episode reward
- terminal `grader_score`
- exfiltration rate
- mean time-to-first-detection
- false-positive actions per episode

## Recommended Reporting Structure

1. Describe scenario and hidden attacker campaign.
2. Compare baseline and trained policy trajectories on fixed seeds.
3. Report terminal scores, exfiltration rates, and containment efficiency.
4. Present learning curves and error analysis.

## Reproducibility

- Pin scenario id and seed in each reported run.
- Report environment config overrides (`horizon`, false positive rate, etc.).
- Keep run logs with observation `info` payloads for forensic traceability.
