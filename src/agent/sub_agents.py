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


class OwnershipMapper(SubAgent):
    """Maps the modified files to suggested reviewers based on code ownership
    and historical commit / review patterns.

    The agent's value here is not just "who owns this code" — it's identifying
    *ownership gaps* (files no one owns) and *bus-factor risk* (one person owns
    everything that just changed). Both are real risk factors that don't show
    up in pure diff analysis.

    Production target (v0.2): **NVIDIA Merlin** two-tower retrieval over
    (file_path, reviewer_identity) pairs, with a reranker on file-content
    similarity. Trained on the org's PR-review history (who reviewed what,
    weighted by acceptance rate). Sub-millisecond GPU inference per PR.

    v0.1 ships a CODEOWNERS-style lookup that:
        - takes a {path_prefix: [owners]} map from metadata
        - uses longest-prefix matching
        - surfaces ownership-gap and bus-factor risk factors
        - returns recommended reviewers in observations
    """

    def name(self) -> str:
        return "ownership-mapper"

    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport:
        from src.agent.tools import git_diff_stats

        stats = git_diff_stats(diff)
        files: list[str] = stats.get("files", [])  # type: ignore[assignment]
        codeowners: dict[str, list[str]] = metadata.get("codeowners", {})

        # If no CODEOWNERS was provided at all, this sub-agent has nothing to
        # say — return a zero-confidence stub. We must not flag every file as
        # "ownership gap" simply because the caller didn't pass the map; that
        # turns a missing input into a permanent false positive.
        if not codeowners:
            return SubAgentReport(
                sub_agent_name=self.name(),
                observations={
                    "status": "no codeowners provided",
                    "files_analyzed": len(files),
                },
                risk_factors=[],
                confidence=0.0,
            )

        recommended_reviewers: set[str] = set()
        files_without_owner: list[str] = []
        per_file_owners: dict[str, list[str]] = {}

        for f in files:
            owners = self._lookup_owner(f, codeowners)
            per_file_owners[f] = owners
            if owners:
                recommended_reviewers.update(owners)
            else:
                files_without_owner.append(f)

        risk_factors: list[str] = []
        if files_without_owner:
            risk_factors.append(
                f"ownership gap: {len(files_without_owner)} file(s) without a declared code owner"
            )
        if len(recommended_reviewers) == 1 and len(files) >= 3:
            risk_factors.append(
                f"bus-factor risk: a single owner covers all {len(files)} modified files"
            )

        if not files:
            confidence = 0.0
        elif recommended_reviewers or files_without_owner:
            confidence = 0.6
        else:
            confidence = 0.2

        return SubAgentReport(
            sub_agent_name=self.name(),
            observations={
                "files_analyzed": len(files),
                "recommended_reviewers": sorted(recommended_reviewers),
                "files_without_owner": files_without_owner,
                "per_file_owners": per_file_owners,
            },
            risk_factors=risk_factors,
            confidence=confidence,
        )

    @staticmethod
    def _lookup_owner(file_path: str, codeowners: dict[str, list[str]]) -> list[str]:
        """Longest-prefix match — mirrors how GitHub CODEOWNERS resolves."""
        matches = [
            (prefix, owners)
            for prefix, owners in codeowners.items()
            if file_path.startswith(prefix)
        ]
        if not matches:
            return []
        matches.sort(key=lambda x: -len(x[0]))
        return list(matches[0][1])
