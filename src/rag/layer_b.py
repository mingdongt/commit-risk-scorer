"""Layer B — operational context (metadata-triggered retrieval).

Layer-B retrievers fire only when PR metadata signals that the operational
context matters: the change touches an on-call-critical service, modifies a
feature-flag-gated path, lands during a release-freeze window, etc.

These are usually structured queries (KV / SQL) rather than vector search.
Each retriever declares which `triggers` activate it; the gateway consults
those flags before invoking. Always-on Layer-B querying would waste cost
on the 95% of PRs where it adds nothing.
"""
from __future__ import annotations

from src.rag.base import Retriever, RetrieverLayer
from src.rag.types import RetrievalQuery, RetrievalResult


class _StubLayerBRetriever(Retriever):
    """Shared error-message scaffolding for the Layer-B stubs."""

    _name: str = ""
    _triggers: frozenset[str] = frozenset()
    _oss_proxy: str = ""
    _enterprise_backend: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> RetrieverLayer:
        return RetrieverLayer.B

    @property
    def triggers(self) -> frozenset[str]:
        """Metadata flags that activate this retriever.

        The gateway invokes a Layer-B retriever iff the query's `triggers`
        intersects this set. An empty set means "always run" (rare for
        Layer B — prefer Layer A for that semantics).
        """
        return self._triggers

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError(
            f"{self._name} retriever is a stub. "
            f"Activates on triggers: {sorted(self._triggers)}. "
            f"OSS-mode proxy: {self._oss_proxy}. "
            f"Enterprise backend: {self._enterprise_backend}."
        )


class ReleaseCalendarRetriever(_StubLayerBRetriever):
    """Check whether the changed area is currently in a freeze / RC window.

    "Mobile team cut the release branch yesterday; this PR touches mobile" —
    not a code-quality signal at all, but a critical operational one.
    """

    _name = "release_calendar"
    _triggers = frozenset({"near_release", "always_check_calendar"})
    _oss_proxy = "Static YAML calendar in repo /docs/release-calendar.yaml"
    _enterprise_backend = "Internal release-management service / release-train tooling"


class OnCallStatusRetriever(_StubLayerBRetriever):
    """Look up who's currently on-call for the affected service, and whether
    they're available.

    "Service owner is on PTO until Monday, secondary is also out" — a high-risk
    PR landing into that window is much riskier than the same PR on a normal
    day.
    """

    _name = "oncall_status"
    _triggers = frozenset({"touches_oncall_critical", "high_severity"})
    _oss_proxy = "GitHub team membership + manual JSON availability override"
    _enterprise_backend = "PagerDuty + HR leave calendar + internal on-call directory"


class FeatureFlagStateRetriever(_StubLayerBRetriever):
    """Check the rollout state of any feature flags referenced by the diff.

    "This code path is gated by a flag at 5% rollout" — the change's blast
    radius depends on the flag state, not the diff size.
    """

    _name = "feature_flag_state"
    _triggers = frozenset({"references_feature_flag"})
    _oss_proxy = "Repo-local flag config + GitHub Actions environment vars"
    _enterprise_backend = "LaunchDarkly / Split / internal flag service"


class SLAandCostRetriever(_StubLayerBRetriever):
    """Surface service SLAs and cost data relevant to the changed component.

    "This service has a 99.99% SLA and 30% of fleet cost — any change carries
    contractual and financial weight."
    """

    _name = "sla_and_cost"
    _triggers = frozenset({"touches_sla_critical", "touches_high_cost_service"})
    _oss_proxy = "Static services.yaml manifest in repo"
    _enterprise_backend = "Internal service catalog + finance cost-attribution dashboard"


class OwnershipTransitionRetriever(_StubLayerBRetriever):
    """Surface in-flight ownership transitions (re-orgs, team handoffs).

    "This module's owning team was disbanded last week; new owner not yet
    designated" — accepting a PR here may strand it.
    """

    _name = "ownership_transition"
    _triggers = frozenset({"recent_codeowners_churn"})
    _oss_proxy = "Git log of CODEOWNERS changes + GitHub team membership history"
    _enterprise_backend = "Internal HR / org-chart service + service-ownership graph"


LAYER_B_RETRIEVERS: tuple[type[Retriever], ...] = (
    ReleaseCalendarRetriever,
    OnCallStatusRetriever,
    FeatureFlagStateRetriever,
    SLAandCostRetriever,
    OwnershipTransitionRetriever,
)
