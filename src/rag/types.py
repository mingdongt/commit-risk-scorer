"""Common types for the RAG layer.

Kept in a dedicated module so that retriever stubs in `layer_a/b/c.py` and
the dispatcher in `gateway.py` can share a single contract without circular
imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Citation:
    """A pointer back to the source document a retrieval result came from.

    The agent surfaces these to the reviewer so high-risk verdicts are
    auditable. Provenance is non-negotiable for the Tier-3 layer — a verdict
    that says "this looks like the auth incident from Q3" is only useful if
    the reviewer can click through to the actual postmortem.
    """

    source_type: str          # e.g. "past_pr", "postmortem", "prd", "compliance_doc"
    source_id: str            # backend-specific identifier (PR number, doc URL, etc.)
    title: str | None = None  # human-readable label for UI
    url: str | None = None    # clickable link when available
    excerpt: str | None = None  # short quoted span the agent grounded on


@dataclass(frozen=True)
class Document:
    """A retrieved chunk plus enough metadata to score / cite it."""

    content: str
    citation: Citation
    score: float = 0.0        # retriever-native relevance score (higher = better)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQuery:
    """Input to any retriever, regardless of layer.

    The query is intentionally rich:

    - `question` is the agent-formulated natural-language query (Layer C
      retrievers and re-rankers consume this).
    - `diff` and `pr_metadata` are the underlying PR context — Layer A and
      Layer B retrievers usually consult these rather than `question`,
      because they trade in structured signals (file paths, author, time).
    - `triggers` is an open set of hints the agent or router can pass to
      route retrieval (e.g. {"touches_pii", "touches_oncall_critical"}).
    """

    question: str
    diff: str | None = None
    pr_metadata: dict[str, Any] = field(default_factory=dict)
    triggers: frozenset[str] = field(default_factory=frozenset)
    top_k: int = 5


@dataclass(frozen=True)
class RetrievalResult:
    """Output of one retriever invocation.

    A list of documents plus a layer tag (so the gateway can attribute cost
    and the audit log can group evidence by retrieval layer).
    """

    layer: str                # "A" / "B" / "C"
    retriever_name: str       # e.g. "similar_prs", "compliance_check"
    documents: tuple[Document, ...] = ()
    latency_ms: float = 0.0

    def is_empty(self) -> bool:
        return len(self.documents) == 0
