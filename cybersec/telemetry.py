"""Telemetry: noise alerts and INVESTIGATE responses.

The attacker module produces alerts that genuinely correlate with adversary
activity (with stochastic delay). The telemetry module's job is to muddy
that signal with two kinds of noise that any real SOC also has to contend
with:

  * Background false-positive alerts (per-tick Bernoulli draw, pulled from a
    pool of plausible-looking signals on random topology entries).
  * Investigation responses with bounded confidence: confirming a true
    compromise comes back high-confidence, but not 1.0; a clean target comes
    back low-confidence "looks fine".

Both pieces are deterministic given the env RNG, so two runs with the same
seed produce the same alert/forensic stream.
"""

from __future__ import annotations

import random
from typing import List, Optional, Set

from .models import AlertEvent, AlertSignal, ForensicResult
from .scenarios import Scenario


_NOISE_DESCRIPTIONS = {
    AlertSignal.AUTH_ANOMALY: "Unusual login from new device",
    AlertSignal.LATERAL_MOVEMENT: "RDP session between non-paired hosts",
    AlertSignal.DATA_STAGING: "Large query against rarely-touched table",
    AlertSignal.EGRESS_ANOMALY: "Outbound TLS to a low-reputation CDN",
    AlertSignal.BACKGROUND_NOISE: "Heuristic-only signal, no IoC match",
}


class TelemetryEngine:
    """Noise generator + investigation oracle.

    Lives next to the environment instance and shares its ``random.Random``.
    """

    def __init__(self, scenario: Scenario, rng: random.Random):
        self.scenario = scenario
        self.rng = rng

    # -------------------------------------------------------------- background

    def background_alerts(self, tick: int) -> List[AlertEvent]:
        """Emit zero-or-one noise alert this tick.

        We cap at one to keep the alert log small. Severity is sampled low
        so the policy can - in principle - learn to discount it.
        """

        if self.rng.random() >= self.scenario.background_alert_rate:
            return []
        signal = self.rng.choice(
            [
                AlertSignal.AUTH_ANOMALY,
                AlertSignal.LATERAL_MOVEMENT,
                AlertSignal.DATA_STAGING,
                AlertSignal.EGRESS_ANOMALY,
                AlertSignal.BACKGROUND_NOISE,
            ]
        )
        # 70% of the time the noise alert ties to a real topology entry,
        # 30% it is generic. Both shapes occur in the wild.
        asset_id: Optional[str] = None
        identity_id: Optional[str] = None
        if self.rng.random() < 0.7:
            if self.rng.random() < 0.5 and self.scenario.assets:
                asset_id = self.rng.choice(self.scenario.assets).asset_id
            elif self.scenario.identities:
                identity_id = self.rng.choice(self.scenario.identities).identity_id

        severity = round(self.rng.uniform(0.1, 0.45), 3)
        return [
            AlertEvent(
                tick=tick,
                signal=signal,
                asset=asset_id,
                identity=identity_id,
                severity=severity,
                description=_NOISE_DESCRIPTIONS[signal],
            )
        ]

    # -------------------------------------------------------------- investigations

    def investigate(
        self,
        tick: int,
        target: str,
        target_kind: str,
        truly_compromised_assets: Set[str],
        truly_compromised_identities: Set[str],
    ) -> ForensicResult:
        """Return a ForensicResult for an INVESTIGATE action.

        The verdict is based on attacker ground truth, but with calibrated
        noise: 15% false-negative on truly compromised, 12% false-positive on
        clean targets. Confidence is sampled in [0.6, 0.95] for true positives,
        [0.55, 0.85] for true negatives, and lower bands for false verdicts.
        """

        if target_kind == "asset":
            truly = target in truly_compromised_assets
        elif target_kind == "identity":
            truly = target in truly_compromised_identities
        else:
            raise ValueError(f"target_kind must be 'asset' or 'identity', got {target_kind!r}")

        if truly:
            if self.rng.random() < 0.15:
                # False negative
                return ForensicResult(
                    tick=tick,
                    target=target,
                    target_kind=target_kind,
                    is_compromised=False,
                    confidence=round(self.rng.uniform(0.4, 0.65), 3),
                )
            return ForensicResult(
                tick=tick,
                target=target,
                target_kind=target_kind,
                is_compromised=True,
                confidence=round(self.rng.uniform(0.6, 0.95), 3),
            )

        if self.rng.random() < 0.12:
            # False positive
            return ForensicResult(
                tick=tick,
                target=target,
                target_kind=target_kind,
                is_compromised=True,
                confidence=round(self.rng.uniform(0.4, 0.6), 3),
            )
        return ForensicResult(
            tick=tick,
            target=target,
            target_kind=target_kind,
            is_compromised=False,
            confidence=round(self.rng.uniform(0.55, 0.85), 3),
        )


__all__ = ["TelemetryEngine"]
