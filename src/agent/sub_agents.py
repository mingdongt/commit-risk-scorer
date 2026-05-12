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

    # Path prefixes that real platform teams treat as high-incident / security-
    # sensitive areas. A change touching any of these is risk-bumped even when
    # the diff itself is small. v0.2 learns this list per adopting team from
    # their incident-postmortem history.
    SENSITIVE_PATH_PREFIXES: tuple[str, ...] = (
        "src/auth/",
        "src/security/",
        "src/billing/",
        "src/payment/",
        "src/crypto/",
        "src/secret",       # secret_store, secrets/, etc.
        "src/admin/",
        "src/migrations/",  # schema changes have outsized blast radius
        "config/",          # production config has outsized blast radius
    )

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

        files: list[str] = stats.get("files", [])  # type: ignore[assignment]
        sensitive_hits: list[str] = [
            f for f in files if any(f.startswith(p) for p in self.SENSITIVE_PATH_PREFIXES)
        ]
        if sensitive_hits:
            # Report which sensitive areas were touched, not the file list (cleaner UX).
            touched_areas = sorted(
                {p for p in self.SENSITIVE_PATH_PREFIXES if any(f.startswith(p) for f in sensitive_hits)}
            )
            risk_factors.append(
                f"sensitive area touched: {len(sensitive_hits)} file(s) under "
                f"{', '.join(touched_areas)} — historically high-incident path(s)"
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
        # No risk_factors emitted from a stub — a "pending v0.2" note should not
        # contribute to a real PR's risk score. The non-implemented status is
        # surfaced in observations for audit/debugging.
        return SubAgentReport(
            sub_agent_name=self.name(),
            observations={
                "status": "stub — pending v0.2 (coverage / test-impact-graph wiring)",
                "tests_covering_modified_files": None,
            },
            risk_factors=[],
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
        # No risk_factors emitted from a stub. Status surfaced in observations
        # for audit/debugging; not for risk-score aggregation.
        return SubAgentReport(
            sub_agent_name=self.name(),
            observations={
                "status": "stub — pending v0.2 (Elasticsearch RAG over historical PRs)",
                "similar_pr_ids": [],
                "rag_index_size": 0,
            },
            risk_factors=[],
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


class AgentPRAuditor(SubAgent):
    """Detects PRs authored by AI coding agents and surfaces agent-specific risk.

    Motivation (also documented in `docs/design-doc.md` §AI-Generated PR Risk):
    in 2026 a non-trivial share of PRs in many engineering orgs are authored by
    agents (Copilot Agent, Devin-class systems, Claude Code, etc.). The risk
    *profile* of an agent-authored diff differs from a human one — even when
    the diffs look similar — and the policy gate should reflect that.

    PR-author classification (from `metadata["author_class"]`, if provided):
        - "human-only"        — no AI signal detected
        - "ai-assisted"       — Copilot-style autocomplete used during authoring
        - "agent-generated"   — full PR authored by an agent; human is approver, not author

    The agent can also INFER the class when not explicitly set:
        - bot-shaped author logins (suffix `-bot`, `[bot]`, `actions-user`, etc.)
        - all commit timestamps within a tight burst (< 5 min span over many commits)
        - commit messages that are formulaic and lack rationale

    Agent-PR-specific risk factors (raised only when class != human-only):
        - large mechanical refactor (high LOC + few files-touched ratio)
        - tests added without assertion logic ("looks like coverage, isn't")
        - missing human rationale (no prose in commit message body)
        - PR description / spec mentions paths not present in the diff
          (interpreted as scope-drift between prompt and shipped diff)

    Status (v0.1): REAL inference for author-class detection from metadata
    (bot-login + burst-timing heuristic); risk-factor surfacing is rule-based.
    v0.2 wires to:
        - a trained classifier on PR-style features (commit-message embedding,
          inter-commit-time distribution, file-type entropy)
        - the LLM judge for scope-drift detection (compares PR description to
          diff content)
    """

    BOT_LOGIN_MARKERS: tuple[str, ...] = (
        "[bot]",
        "-bot",
        "actions-user",
        "copilot-",
        "devin-",
        "github-actions",
    )
    MECHANICAL_REFACTOR_LINES_PER_FILE = 80  # avg additions+deletions per file

    def name(self) -> str:
        return "agent-pr-auditor"

    def analyze(self, diff: str, metadata: dict[str, Any]) -> SubAgentReport:
        from src.agent.tools import git_diff_stats

        author_class = self._classify_author(metadata)
        stats = git_diff_stats(diff)

        observations: dict[str, Any] = {
            "author_class": author_class,
            "author_login": metadata.get("author", {}).get("login")
            if isinstance(metadata.get("author"), dict)
            else metadata.get("author"),
        }

        risk_factors: list[str] = []

        # Only run the agent-specific checks for AI-touched PRs.
        if author_class != "human-only":
            n_files = int(stats.get("files_touched", 0))  # type: ignore[arg-type]
            n_lines = int(stats.get("additions", 0)) + int(stats.get("deletions", 0))  # type: ignore[arg-type]
            if n_files >= 5 and n_lines / max(n_files, 1) >= self.MECHANICAL_REFACTOR_LINES_PER_FILE:
                risk_factors.append(
                    f"agent-authored mechanical refactor: {n_lines} lines across {n_files} files"
                )

            messages: list[str] = metadata.get("commit_messages") or []
            if messages and not any(len(m.strip()) > 80 for m in messages):
                # No commit message has any prose body — humans typically add at
                # least one substantive message; agents commonly emit one-liners.
                risk_factors.append(
                    "agent-authored PR missing human rationale (all commit messages are one-liners)"
                )

            # Scope-drift: PR description names paths not in the diff. The
            # description path-list is whatever the caller chose to surface in
            # metadata["pr_description_paths"]; v0.2 extracts this automatically.
            described_paths: list[str] = metadata.get("pr_description_paths") or []
            diffed_paths = set(stats.get("files", []))  # type: ignore[arg-type]
            if described_paths:
                mentioned_not_in_diff = [p for p in described_paths if p not in diffed_paths]
                if mentioned_not_in_diff:
                    risk_factors.append(
                        f"agent-authored scope drift: PR description mentions {len(mentioned_not_in_diff)} "
                        f"path(s) not in the diff"
                    )

        # Confidence: high when we have a clear author-class signal; low when
        # we had to fall back to weak heuristics with no metadata.
        if author_class == "human-only" and not metadata.get("author"):
            # Default classification with no signals — not much to say.
            confidence = 0.1
        elif author_class == "human-only":
            confidence = 0.4
        else:
            confidence = 0.6

        return SubAgentReport(
            sub_agent_name=self.name(),
            observations=observations,
            risk_factors=risk_factors,
            confidence=confidence,
        )

    def _classify_author(self, metadata: dict[str, Any]) -> str:
        """Resolve author class: explicit metadata wins; otherwise infer from signals."""
        explicit = metadata.get("author_class")
        if explicit in {"human-only", "ai-assisted", "agent-generated"}:
            return explicit  # type: ignore[return-value]

        # Infer from author login (bot accounts).
        author = metadata.get("author")
        login: str | None = None
        if isinstance(author, dict):
            login = author.get("login")
        elif isinstance(author, str):
            login = author
        if login and any(marker in login.lower() for marker in self.BOT_LOGIN_MARKERS):
            return "agent-generated"

        # Infer from commit-timing burst.
        timestamps: list[float] | None = metadata.get("commit_timestamps")
        if timestamps and len(timestamps) >= 5:
            span = max(timestamps) - min(timestamps)
            if span < 300:  # 5 minutes — humans rarely emit 5+ commits in 5 min
                return "agent-generated"

        return "human-only"
