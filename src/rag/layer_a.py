"""Layer A — code-adjacent operational data (always-on retrieval).

Every Tier-3 agent invocation queries every Layer-A retriever. These sources
are cheap to query, high signal-to-noise, and structured enough that a
code-aware embedding + BM25 hybrid does well across all of them.

Status: interfaces concrete, implementations are stubs that raise
`NotImplementedError` with the OSS proxy and enterprise backend documented
inline. The gateway dispatch (`gateway.py`) treats stubs as "no result" so
the system degrades gracefully — useful while implementations are landing
incrementally.
"""
from __future__ import annotations

from src.rag.base import Retriever, RetrieverLayer
from src.rag.types import RetrievalQuery, RetrievalResult


class _StubLayerARetriever(Retriever):
    """Shared error-message scaffolding for the Layer-A stubs.

    Each concrete subclass overrides `name`, `_oss_proxy`, and
    `_enterprise_backend` so the NotImplementedError tells you exactly
    what's missing and what to swap in.
    """

    _name: str = ""
    _oss_proxy: str = ""
    _enterprise_backend: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> RetrieverLayer:
        return RetrieverLayer.A

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError(
            f"{self._name} retriever is a stub. "
            f"OSS-mode proxy: {self._oss_proxy}. "
            f"Enterprise backend: {self._enterprise_backend}."
        )


class SimilarPRsRetriever(_StubLayerARetriever):
    """Find historical PRs whose diffs resemble the current one.

    Strongest single Layer-A signal — "the last three times someone changed
    these files, here's what went wrong" is what a senior reviewer would
    look for. Code-aware embeddings (Voyage-Code-3 / GraphCodeBERT) over a
    commit-history index.
    """

    _name = "similar_prs"
    _oss_proxy = "GitHub PR API + Voyage-Code-3 embeddings indexed in Elasticsearch"
    _enterprise_backend = "internal source-control history + internal code-embedding service"


class IncidentHistoryRetriever(_StubLayerARetriever):
    """Look up postmortems / incident reports touching the changed modules.

    "This module had three P0s in the last 90 days" is the kind of context
    a reviewer wants surfaced *before* approving a change to it.
    """

    _name = "incident_history"
    _oss_proxy = 'GitHub issues filtered by label="incident"|"postmortem" + embeddings'
    _enterprise_backend = "ICM / Jira / internal postmortem store"


class OwnershipRetriever(_StubLayerARetriever):
    """Resolve who owns the changed files and who's currently on-call.

    Pure structured lookup — KV over CODEOWNERS + an on-call directory.
    No embeddings needed.
    """

    _name = "ownership"
    _oss_proxy = "CODEOWNERS parser + GitHub team membership API"
    _enterprise_backend = "internal service-ownership graph + PagerDuty / on-call directory"


class ArchitectureDecisionRetriever(_StubLayerARetriever):
    """Surface ADRs (architecture decision records) that constrain the change.

    "ADR-042 said this module is frozen for re-architecture in Q3" — if the
    decision is in the repo, retrieve it; if it's in an internal wiki,
    the enterprise backend handles it the same way.
    """

    _name = "architecture_decision"
    _oss_proxy = "Semantic search over /docs/adr/ markdown files in the repo"
    _enterprise_backend = "Internal architecture-decision store / Confluence ADR space"


class BuildFailureRetriever(_StubLayerARetriever):
    """Retrieve historical CI build failures with root-cause attribution.

    "Last 5 PRs that touched this Bazel BUILD file all caused flaky test X"
    — high-value retrospective signal for build-failure forecasting.
    """

    _name = "build_failure_rca"
    _oss_proxy = "GitHub Actions run history + commit-to-failure linkage"
    _enterprise_backend = "Internal CI store (CloudBuild / TeamCity / Jenkins) + RCA database"


class SecurityAdvisoryRetriever(_StubLayerARetriever):
    """Check dependency changes against published security advisories.

    Trigger: diff modifies a manifest / lock file. Mostly structured
    (package@version → advisory list).
    """

    _name = "security_advisory"
    _oss_proxy = "OSV.dev API + GitHub Advisory Database"
    _enterprise_backend = "Internal SBOM + internal vulnerability database"


class CodingStandardRetriever(_StubLayerARetriever):
    """Match diff patterns against internal coding standards.

    "This raw pointer usage violates the internal unsafe-pointer policy" —
    static-analysis-adjacent, but driven by retrieval rather than hand-coded
    rules so the standard can evolve without recompiling the agent.
    """

    _name = "coding_standard"
    _oss_proxy = "Semantic search over /docs/standards/ + linter rule database"
    _enterprise_backend = "Internal coding-standard wiki + per-team style enforcement"


LAYER_A_RETRIEVERS: tuple[type[Retriever], ...] = (
    SimilarPRsRetriever,
    IncidentHistoryRetriever,
    OwnershipRetriever,
    ArchitectureDecisionRetriever,
    BuildFailureRetriever,
    SecurityAdvisoryRetriever,
    CodingStandardRetriever,
)
