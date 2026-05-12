"""Policy gatekeeper — turns a numeric risk score into an actionable decision.

The risk score is *not* the product. The action is. This module mirrors the
Risk → Action mapping documented in the README and is the post-aggregation
step the agent harness invokes after the sub-agent reports are combined.

Status (v0.1): REAL. Deterministic threshold-based policy with per-deployment
overridable thresholds. v0.2 layers learned thresholds (calibrated on the
adopting team's historical PR outcomes) on top of these defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.harness import HarnessResult


@dataclass(frozen=True)
class PolicyDecision:
    """The output of the policy gatekeeper."""

    action: str  # one of: fast_track | owner_review | sme_review | block_merge
    risk_level: str  # one of: Low | Medium | High | Critical
    rationale: str
    recommended_steps: list[str] = field(default_factory=list)


# Default risk-band → action mapping. Matches the table in README "Risk → Action
# mapping." Half-open intervals: [lo, hi).
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "fast_track": (0.00, 0.20),
    "owner_review": (0.20, 0.50),
    "sme_review": (0.50, 0.80),
    "block_merge": (0.80, 1.01),  # 1.01 so 1.0 is inclusive in the top band
}

RISK_LEVELS: dict[str, str] = {
    "fast_track": "Low",
    "owner_review": "Medium",
    "sme_review": "High",
    "block_merge": "Critical",
}

DEFAULT_RECOMMENDED_STEPS: dict[str, list[str]] = {
    "fast_track": [
        "Standard review queue",
        "Default CI suite",
    ],
    "owner_review": [
        "Assign code owner from CODEOWNERS",
        "Run targeted test suite for modified packages",
    ],
    "sme_review": [
        "Require SME reviewer from the affected domain",
        "Run extended integration test suite",
        "Surface to the team's PR-risk dashboard",
    ],
    "block_merge": [
        "Block auto-merge until manual approver disposition",
        "Require sign-off from CODEOWNERS + SME",
        "Run full regression suite including flake-flagged tests",
        "Page on-call if the change touches a critical path",
    ],
}


class PolicyGatekeeper:
    """Converts a HarnessResult into an explicit PolicyDecision.

    Parameters
    ----------
    thresholds:
        Optional per-deployment threshold overrides. Same structure as
        DEFAULT_THRESHOLDS. Strict orgs lower the block_merge floor; permissive
        orgs widen fast_track.
    """

    def __init__(self, thresholds: dict[str, tuple[float, float]] | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        # Sanity: bands must be ordered and not overlap (allow touching).
        bands = sorted(self.thresholds.values(), key=lambda b: b[0])
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(bands, bands[1:]):
            if a_hi > b_lo:
                raise ValueError(f"overlapping policy bands: {a_hi} > {b_lo}")

    def decide(self, result: HarnessResult) -> PolicyDecision:
        score = max(0.0, min(1.0, result.risk_score))
        action = self._band_for(score)
        risk_level = RISK_LEVELS[action]

        # Build a short rationale that names the score and the dominant sub-agent
        # risk factors (top 3) — so reviewers see *why* the action was chosen.
        risk_factor_summary = self._summarize_risk_factors(result)
        rationale = (
            f"Risk score {score:.2f} → {risk_level} band → action: {action.replace('_', ' ')}. "
            f"{risk_factor_summary}".strip()
        )

        return PolicyDecision(
            action=action,
            risk_level=risk_level,
            rationale=rationale,
            recommended_steps=list(DEFAULT_RECOMMENDED_STEPS[action]),
        )

    def _band_for(self, score: float) -> str:
        for action, (lo, hi) in self.thresholds.items():
            if lo <= score < hi:
                return action
        # Fallback — should not happen given the 1.01 ceiling on block_merge.
        return "block_merge"

    @staticmethod
    def _summarize_risk_factors(result: HarnessResult) -> str:
        # Pull the top 3 risk factors across all sub-agents (any order from
        # confident reports first). Keeps the rationale readable.
        ordered = sorted(
            result.sub_agent_reports, key=lambda r: -r.confidence
        )
        flat: list[str] = []
        for report in ordered:
            for rf in report.risk_factors:
                if rf not in flat:
                    flat.append(rf)
            if len(flat) >= 3:
                break
        if not flat:
            return "No specific risk factors surfaced by the sub-agents."
        top = flat[:3]
        return "Top risk factors: " + "; ".join(top) + "."
