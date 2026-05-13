"""Multi-agent harness for commit-risk-scorer.

This harness is the concrete v0.2 implementation of the **Pre-Merge Risk
Workflow** — Node #1 of the 5-agent Agentic SDLC System described in
`docs/agentic-sdlc-architecture.md`. In the system framing it is one
specialization of a top-level `SDLCWorkflowAgent` that routes events to
specialist workflows; v0.2 ships only the Pre-Merge specialization, so the
top-level orchestrator is currently implicit.

Coordinates 5 specialized sub-agents (diff-analyzer, ownership-mapper,
agent-pr-auditor, test-impact-scout, historical-context) and aggregates their
reports. When a `ClaudeJudgeBackend` is injected the harness escalates to
Tier 2 — passing the sub-agent reports and any retrieved RAG context to the
LLM judge and using its calibrated verdict instead of the heuristic
aggregation. When no judge is injected the harness behaves exactly like v0.1
(deterministic, no network calls, CPU-only).

Architecture mirrors the diagram in `docs/design-doc.md` §Architecture; system
context in `docs/agentic-sdlc-architecture.md`.

Status (v0.2):
    - Sub-agent dispatch: REAL (sequential — concurrency lands once tools become
      I/O-bound).
    - Heuristic aggregation: REAL (used when no judge backend is registered).
    - LLM judge call: REAL via ClaudeJudgeBackend (Anthropic Claude API,
      prompt-cached system prompt, adaptive thinking, structured outputs).
    - RAG dispatch: surface ready — all Layer A/B/C retrievers are still
      stubs, so the dispatch returns no documents until the first real
      retriever lands. The judge handles an empty RAG section explicitly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agent.sub_agents import (
    AgentPRAuditor,
    DiffAnalyzer,
    HistoricalContext,
    OwnershipMapper,
    SubAgent,
    SubAgentReport,
    TestImpactScout,
)


logger = logging.getLogger(__name__)


class _JudgeBackend(Protocol):
    """Structural type the harness expects of an injected judge backend.

    Defined as a Protocol rather than importing `ClaudeJudgeBackend` directly
    so the harness module stays import-light (no `anthropic` dependency at
    import time when the judge is not used). The concrete implementation is
    `src.models.gateway.ClaudeJudgeBackend`.
    """

    def predict(self, request: Any) -> Any: ...


class _RagGateway(Protocol):
    """Structural type for the RAG dispatcher. Concrete: `src.rag.gateway.RagGateway`."""

    def dispatch(self, query: Any) -> Any: ...


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

    When `judge_backend` is provided the harness escalates to Tier 2 after
    sub-agents run: the structured reports (and any retrieved RAG context, if
    `rag_gateway` is provided) are forwarded to the LLM judge, whose
    calibrated verdict overrides the heuristic aggregation. If the judge
    raises, the harness logs and falls back to the heuristic — a CI step
    must not fail because of a transient LLM outage.
    """

    def __init__(
        self,
        sub_agents: list[SubAgent] | None = None,
        judge_backend: _JudgeBackend | None = None,
        rag_gateway: _RagGateway | None = None,
    ):
        self.sub_agents = sub_agents or [
            DiffAnalyzer(),
            OwnershipMapper(),
            AgentPRAuditor(),
            TestImpactScout(),
            HistoricalContext(),
        ]
        self.judge_backend = judge_backend
        self.rag_gateway = rag_gateway

    def score(self, diff: str, metadata: dict[str, Any] | None = None) -> HarnessResult:
        """Score a diff/commit and return a HarnessResult."""
        metadata = metadata or {}
        reports = [agent.analyze(diff, metadata) for agent in self.sub_agents]
        heuristic_score = self._heuristic_aggregate(reports)

        if self.judge_backend is None:
            return HarnessResult(
                risk_score=heuristic_score,
                sub_agent_reports=reports,
                reasoning=None,
            )

        # Tier 2 — escalate to the LLM judge. Import lazily so the harness
        # module does not pull `anthropic` into projects that never use it.
        from src.models.gateway import PredictionRequest  # noqa: PLC0415

        request = PredictionRequest(
            diff=diff,
            pr_metadata=metadata,
            sub_agent_reports=tuple(_report_to_dict(r) for r in reports),
            rag_documents=self._dispatch_rag(diff, metadata),
        )

        try:
            verdict = self.judge_backend.predict(request)
        except Exception as exc:  # noqa: BLE001 — must not crash the CI step
            logger.warning(
                "LLM judge call failed (%s); falling back to heuristic score.", exc
            )
            return HarnessResult(
                risk_score=heuristic_score,
                sub_agent_reports=reports,
                reasoning=f"(judge unavailable: {exc!s}; using heuristic score)",
            )

        return HarnessResult(
            risk_score=verdict.risk_score,
            sub_agent_reports=reports,
            reasoning=verdict.reasoning,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_aggregate(reports: list[SubAgentReport]) -> float:
        """Confidence-weighted heuristic — the v0.1 baseline + Tier-2 fallback."""
        total_weight = sum(r.confidence for r in reports) or 1.0
        weighted_risk = (
            sum(r.confidence * (1.0 if r.risk_factors else 0.0) for r in reports)
            / total_weight
        )
        # Only count risk factors from confident sub-agents — zero-confidence
        # stubs must not move the production risk score.
        n_factors = sum(len(r.risk_factors) for r in reports if r.confidence > 0)
        return max(0.0, min(1.0, weighted_risk + 0.05 * n_factors))

    def _dispatch_rag(
        self, diff: str, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        """Dispatch the RAG gateway (if injected) and flatten results to dicts.

        All retrievers currently return empty results (stubs degrade gracefully
        via NotImplementedError handling in `RagGateway._call_one`). When the
        first real retriever lands this method already routes its output into
        the judge — no further wiring needed.
        """
        if self.rag_gateway is None:
            return ()
        try:
            # Local import to avoid hard-coupling the harness module to the
            # rag package — keeps the import graph thin for harness-only use.
            from src.rag.types import RetrievalQuery  # noqa: PLC0415

            query = RetrievalQuery(diff=diff, metadata=metadata, triggers=frozenset())
            response = self.rag_gateway.dispatch(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG dispatch failed (%s); proceeding without context.", exc)
            return ()

        docs: list[dict[str, Any]] = []
        for result in getattr(response, "results", ()) or ():
            for doc in getattr(result, "documents", ()) or ():
                docs.append(
                    {
                        "layer": getattr(result, "layer", "?"),
                        "retriever": getattr(result, "retriever_name", "?"),
                        "score": getattr(doc, "score", None),
                        "content": getattr(doc, "content", "") or str(doc),
                    }
                )
        return tuple(docs)


def _report_to_dict(report: SubAgentReport) -> dict[str, Any]:
    """Convert a SubAgentReport to the dict shape PredictionRequest expects."""
    return {
        "name": report.sub_agent_name,
        "confidence": report.confidence,
        "observations": dict(report.observations),
        "risk_factors": list(report.risk_factors),
    }


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
