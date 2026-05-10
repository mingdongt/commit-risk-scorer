"""Specialized sub-agents that focus on one aspect of commit risk.

The harness (see harness.py) runs these and aggregates their outputs into a final
risk signal. Each sub-agent owns its own observation -> risk-factors mapping; the
LLM judge layer (v0.2) consumes the structured reports as part of its grounded
reasoning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.agent.tools import git_diff_stats


@dataclass
class SubAgentReport:
    """Output of one sub-agent's analysis."""

    sub_agent_name: str
    observations: dict[str, Any]
    risk_factors: list[str] = field(default_factory=list)
    confidence: float = 0.5  # how much weight the harness should put on this report, in [0, 1]


class SubAgent(ABC):
    """Base class — each sub-agent answers a single question about the diff."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport: ...


class DiffAnalyzer(SubAgent):
    """Observes the structural shape of the diff: size, fanout, file-type mix.

    This is the only sub-agent fully implemented in v0.1; the others ship stubs that
    return zero-confidence reports until their backing data sources land.
    """

    # Heuristic thresholds — chosen conservatively; will be replaced by learned
    # thresholds from the calibration step in v0.2.
    LARGE_FANOUT = 10
    LARGE_DIFF_LINES = 500

    def name(self) -> str:
        return "diff-analyzer"

    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport:
        stats = git_diff_stats(diff)
        risk_factors: list[str] = []

        if stats["files_touched"] >= self.LARGE_FANOUT:
            risk_factors.append(
                f"large fanout: {stats['files_touched']} files touched (threshold: {self.LARGE_FANOUT})"
            )
        total_lines = stats["additions"] + stats["deletions"]
        if total_lines >= self.LARGE_DIFF_LINES:
            risk_factors.append(
                f"very large diff: {total_lines} lines changed (threshold: {self.LARGE_DIFF_LINES})"
            )

        # Confidence: this sub-agent has high signal when there ARE risk factors;
        # baseline 0.4 when nothing of note is found.
        confidence = 0.7 if risk_factors else 0.4

        return SubAgentReport(
            sub_agent_name=self.name(),
            observations=stats,
            risk_factors=risk_factors,
            confidence=confidence,
        )


class TestImpactScout(SubAgent):
    """Identifies which tests exercise the modified files.

    Status: STUB. v0.2 wires to coverage.py output / Bazel test-impact graphs to
    produce real test sets and surface "no tests cover this change" as a risk factor.
    """

    def name(self) -> str:
        return "test-impact-scout"

    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport:
        return SubAgentReport(
            sub_agent_name=self.name(),
            observations={
                "status": "stub",
                "tests_covering_modified_files": None,
            },
            risk_factors=["test impact analysis pending v0.2"],
            confidence=0.0,  # zero-weight until implemented
        )


class HistoricalContext(SubAgent):
    """RAG over historical PRs / past CI failures to surface analogous prior changes.

    Status: STUB. v0.2 wires to the Elasticsearch index produced by
    src/data/scrape_github_prs.py and returns top-K similar past PRs with their
    CI outcomes as grounded context for the judge.
    """

    def name(self) -> str:
        return "historical-context"

    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport:
        return SubAgentReport(
            sub_agent_name=self.name(),
            observations={
                "status": "stub",
                "similar_pr_ids": [],
                "rag_index_size": 0,
            },
            risk_factors=["RAG retrieval pending v0.2"],
            confidence=0.0,
        )
