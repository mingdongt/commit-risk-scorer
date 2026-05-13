"""Layer C — cross-functional enterprise knowledge (LLM-judged retrieval).

The hardest and highest-value layer. These sources are heterogeneous prose
(PRDs, compliance docs, customer support themes, vendor contracts, internal
Slack archives, NVIDIA-specific domain knowledge). They're queried rarely
but, when relevant, carry P0 signal.

Activation model: the Tier-3 agent inspects the diff and decides which
Layer-C retrievers (if any) are worth consulting. The gateway exposes them
to the agent as named tools; the agent picks. This is "agentic RAG" rather
than "always-on flat RAG" — only the agent has enough context to know
whether a payment-flow change is the kind that should trigger a compliance
check.

Retrieval characteristics:
- Long-context text embeddings (not code-specialized).
- Hierarchical / semantic chunking (titles, sections, paragraphs).
- Cross-encoder re-ranking on top of dense retrieval.
- Strict provenance — every retrieved span must cite back to a specific
  document so the reviewer can audit the claim.
"""
from __future__ import annotations

from src.rag.base import Retriever, RetrieverLayer
from src.rag.types import RetrievalQuery, RetrievalResult


class _StubLayerCRetriever(Retriever):
    """Shared error-message scaffolding for the Layer-C stubs."""

    _name: str = ""
    _when_to_invoke: str = ""
    _oss_proxy: str = ""
    _enterprise_backend: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> RetrieverLayer:
        return RetrieverLayer.C

    @property
    def when_to_invoke(self) -> str:
        """Human-readable description of when the agent should consult this
        retriever. Surfaced to the agent as the tool's description; the LLM
        uses it to decide whether to call.
        """
        return self._when_to_invoke

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError(
            f"{self._name} retriever is a stub. "
            f"When to invoke: {self._when_to_invoke}. "
            f"OSS-mode proxy: {self._oss_proxy}. "
            f"Enterprise backend: {self._enterprise_backend}."
        )


class ProductRequirementsRetriever(_StubLayerCRetriever):
    """Look up PRDs / roadmaps relevant to the changed area.

    Catches misalignment between code change and product intent — e.g. a
    refactor in an area the PRD says is being deprecated next quarter.
    """

    _name = "prd_search"
    _when_to_invoke = (
        "When the diff appears to extend or refactor user-facing functionality, "
        "to check the area's product status (deprecated / frozen / active)."
    )
    _oss_proxy = "Semantic search over /docs/prd/ markdown + repo's GitHub Discussions"
    _enterprise_backend = "Internal PRD store (Confluence / Aha! / internal product wiki)"


class ComplianceCheckRetriever(_StubLayerCRetriever):
    """Surface compliance constraints that apply to the change.

    GDPR / SOC2 / PCI / export-control — these constraints are documented in
    long-form prose, not code, and violating them is a P0 outcome. Highest-
    stakes retriever in Layer C.
    """

    _name = "compliance_check"
    _when_to_invoke = (
        "When the diff touches PII handling, data storage, cryptographic "
        "operations, or anything regulated (e.g., export-controlled code paths)."
    )
    _oss_proxy = "Semantic search over /docs/compliance/ + public regulatory text"
    _enterprise_backend = "Internal compliance / legal / privacy-engineering doc store"


class VendorContractRetriever(_StubLayerCRetriever):
    """Check the change against active vendor / partner contracts.

    "This API breaking change violates the contract with vendor X" — usually
    not visible to the engineer making the change.
    """

    _name = "vendor_contract"
    _when_to_invoke = (
        "When the diff modifies an API contract / wire format / data schema "
        "shared with an external vendor or partner integration."
    )
    _oss_proxy = "Semantic search over /docs/integrations/ + API schema diff"
    _enterprise_backend = "Internal contract management system + partner-integration registry"


class CustomerFeedbackRetriever(_StubLayerCRetriever):
    """Pull recent customer-support themes and VoC for the affected module.

    "Module under elevated complaint pressure" — any change here is
    operationally riskier than the same change to a quiet module.
    """

    _name = "customer_feedback"
    _when_to_invoke = (
        "When the diff modifies user-visible behavior in a customer-facing "
        "module, to gauge the area's current support-load pressure."
    )
    _oss_proxy = "GitHub issues + Discussions + external community forum scrape"
    _enterprise_backend = "Internal support ticket store (Zendesk / Salesforce) + VoC analytics"


class StrategicPlanningRetriever(_StubLayerCRetriever):
    """Surface OKRs / strategic plans / quarterly priorities for the area.

    Catches mismatch between PR scope and team priorities — a large refactor
    in a deprioritized area is itself a signal.
    """

    _name = "strategic_planning"
    _when_to_invoke = (
        "When the diff is large or touches multiple modules, to check whether "
        "the scope aligns with current quarterly priorities."
    )
    _oss_proxy = "Semantic search over /docs/roadmap/ + repo milestone metadata"
    _enterprise_backend = "Internal OKR system + strategy doc repository"


class InternalDecisionArchiveRetriever(_StubLayerCRetriever):
    """Search internal Slack / Teams / email archives for recent decisions
    that affect the change.

    "Last week's architecture review decided to freeze refactors in this
    area" — decisions made in chat rarely make it into ADRs immediately.
    """

    _name = "internal_decision_archive"
    _when_to_invoke = (
        "When the diff is in an area where a recent design decision could "
        "supersede the implementation (e.g., contested module, recent reorg)."
    )
    _oss_proxy = "GitHub Discussions + PR-review-comment archive"
    _enterprise_backend = "Slack / Teams export + internal email + meeting-notes corpus"


class DomainKnowledgeRetriever(_StubLayerCRetriever):
    """Query domain-specific knowledge bases.

    For NVIDIA IPP specifically: GPU driver compatibility matrices,
    hardware spec documents, automotive safety standards (ASIL-D), CUDA
    feature-level compatibility, etc. These are the documents that an
    engineer changing low-level code would consult — surfacing them
    automatically catches whole classes of regression that look fine to
    a generic LLM but violate domain invariants.
    """

    _name = "domain_kb"
    _when_to_invoke = (
        "When the diff touches domain-specific subsystems — hardware "
        "interfaces, driver code, safety-critical paths, or anything where "
        "domain spec compliance matters more than general code quality."
    )
    _oss_proxy = "Semantic search over /docs/domain/ + linked open standards"
    _enterprise_backend = (
        "NVIDIA-internal: hardware spec store + ASIL-D safety doc corpus + "
        "GPU compatibility matrix + driver release-note archive"
    )


LAYER_C_RETRIEVERS: tuple[type[Retriever], ...] = (
    ProductRequirementsRetriever,
    ComplianceCheckRetriever,
    VendorContractRetriever,
    CustomerFeedbackRetriever,
    StrategicPlanningRetriever,
    InternalDecisionArchiveRetriever,
    DomainKnowledgeRetriever,
)
