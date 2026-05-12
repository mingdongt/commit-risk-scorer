"""Multi-agent harness for commit-risk-scorer.

Coordinates 3 specialized sub-agents (diff-analyzer, test-impact-scout,
historical-context) and aggregates their reports into a final risk score. The
LLM-judge step (v0.2) consumes the structured reports through
`src.models.gateway.ModelGateway` to produce grounded natural-language reasoning.

Architecture mirrors the diagram in `docs/design-doc.md` §Architecture.

Status (v0.1):
    - Sub-agent dispatch: REAL (sequential — concurrency lands once tools become
      I/O-bound).
    - Score aggregation: simple confidence-weighted heuristic; replaced by the
      hybrid (FT classifier + LLM judge) layer in v0.2.
    - LLM judge call: STUB (gateway.predict_hybrid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.sub_agents import (
    AgentPRAuditor,
    DiffAnalyzer,
    HistoricalContext,
    OwnershipMapper,
    SubAgent,
    SubAgentReport,
    TestImpactScout,
)


@dataclass
class HarnessResult:
    """Aggregated output of the agent harness."""

    risk_score: float  # in [0.0, 1.0]
    sub_agent_reports: list[SubAgentReport] = field(default_factory=list)
    reasoning: str | None = None  # populated by LLM judge in v0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "reasoning": self.reasoning,
            "sub_agent_reports": [
                {
                    "name": r.sub_agent_name,
                    "confidence": round(r.confidence, 4),
                    "observations": r.observations,
                    "risk_factors": r.risk_factors,
                }
                for r in self.sub_agent_reports
            ],
        }


class AgentHarness:
    """Runs sub-agents and aggregates their outputs.

    The default sub-agent set matches the design-doc architecture; pass a custom
    list (e.g., a subset for unit tests) to swap in.
    """

    def __init__(self, sub_agents: list[SubAgent] | None = None):
        self.sub_agents = sub_agents or [
            DiffAnalyzer(),
            OwnershipMapper(),
            AgentPRAuditor(),
            TestImpactScout(),
            HistoricalContext(),
        ]

    def score(self, diff: str, metadata: dict[str, Any] | None = None) -> HarnessResult:
        """Score a diff/commit and return a HarnessResult."""
        metadata = metadata or {}
        reports = [agent.analyze(diff, metadata) for agent in self.sub_agents]

        # Aggregation heuristic — v0.2 replaces this with the hybrid pipeline:
        #     FT classifier score (Triton+NeMo)  + LLM judge reasoning (Claude).
        #
        # For v0.1 we use:
        #     risk_score = sum(confidence * has_risk_factors) / total_confidence
        #                  + 0.05 * total_risk_factor_count
        #     clamped to [0, 1].

        total_weight = sum(r.confidence for r in reports) or 1.0
        weighted_risk = (
            sum(r.confidence * (1.0 if r.risk_factors else 0.0) for r in reports) / total_weight
        )
        # Only count risk factors from confident sub-agents — zero-confidence
        # stubs ("pending v0.2") must not move the production risk score.
        n_factors = sum(len(r.risk_factors) for r in reports if r.confidence > 0)
        risk_score = max(0.0, min(1.0, weighted_risk + 0.05 * n_factors))

        return HarnessResult(
            risk_score=risk_score,
            sub_agent_reports=reports,
            reasoning=None,  # filled in by LLM judge step in v0.2
        )


# ---------------------------------------------------------------------------
# Demo entry point — `python -m src.agent.harness` runs a hand-crafted diff
# through the harness and pretty-prints the result.
# ---------------------------------------------------------------------------

DEMO_DIFF = """\
--- a/src/auth/session.py
+++ b/src/auth/session.py
@@ -10,7 +10,7 @@
 def validate(user, token):
-    if user is not None and user.is_active and token.is_valid:
+    if user.is_active and token.is_valid:
         return True
     return False
"""


def _demo() -> None:
    # Local imports — keep the demo as the only place that pulls in
    # policy + explainer, so the harness module stays focused.
    from src.agent.explainer import ExplanationWriter
    from src.agent.policy import PolicyGatekeeper

    harness = AgentHarness()
    demo_metadata = {
        "author": "demo",
        "repo": "demo/repo",
        # Minimal CODEOWNERS — exercises OwnershipMapper.
        "codeowners": {
            "src/auth/": ["@security-team", "@auth-owner"],
            "src/": ["@platform-team"],
        },
    }
    result = harness.score(DEMO_DIFF, metadata=demo_metadata)

    print(f"=== Sub-agent reports ===")
    print(f"Aggregated risk score: {result.risk_score:.2f}\n")
    for report in result.sub_agent_reports:
        print(f"[{report.sub_agent_name}] confidence={report.confidence:.2f}")
        if report.observations:
            for k, v in report.observations.items():
                print(f"  {k}: {v}")
        if report.risk_factors:
            print(f"  risk_factors:")
            for rf in report.risk_factors:
                print(f"    - {rf}")
        print()

    decision = PolicyGatekeeper().decide(result)
    print(f"=== Policy decision ===")
    print(f"Action:        {decision.action}")
    print(f"Risk level:    {decision.risk_level}")
    print(f"Rationale:     {decision.rationale}\n")

    print(f"=== PR comment (markdown rendered by ExplanationWriter) ===")
    print(ExplanationWriter().render(result, decision))


if __name__ == "__main__":
    _demo()
