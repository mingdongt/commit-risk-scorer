"""Cost-engineered tiered routing across cheap → medium → expensive backends.

Why this exists
---------------
A naive "always call the LLM judge" pipeline cannot survive a 5000-commit/day
workload: token cost scales linearly with traffic, and 95% of commits are
low-risk where the cheap classifier's verdict is already final. The router
implements the *escalate-only-when-needed* pattern that every production
risk-scoring system converges on:

    Tier 1 — cheap ML classifier (GBDT / fine-tuned small LM). Always runs.
    Tier 2 — LLM judge (e.g. Claude Haiku). Runs only when T1 is suspicious.
    Tier 3 — agent investigation with tools. Runs only on the riskiest tail.

With default thresholds (0.6 / 0.8) on a typical PR distribution this routes
roughly 95% of requests to Tier 1, ~20% to Tier 2, and ~5% to Tier 3 — the
expensive backends pay only for the commits where their signal actually
matters. Teams tune the thresholds against their own historical score
histogram.

This also closes design-doc §Open Questions "Should the judge see the FT
classifier's score before reasoning, or operate independently and combine
downstream?" — the chosen answer is *cascade, not ensemble*: cheap first,
escalate on signal.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.gateway import (
    ModelBackend,
    PredictionRequest,
    PredictionResponse,
)


@dataclass
class TieredRouterConfig:
    """Escalation thresholds.

    A request escalates to Tier N+1 iff the score from Tier N is >= the
    corresponding threshold. The use of `>=` rather than `>` is deliberate:
    exact-threshold scores are treated as suspicious enough to warrant the
    next tier (errs on the side of safety, the cheap direction).
    """

    escalate_to_tier_2: float = 0.6
    escalate_to_tier_3: float = 0.8

    def __post_init__(self) -> None:
        for name, value in (
            ("escalate_to_tier_2", self.escalate_to_tier_2),
            ("escalate_to_tier_3", self.escalate_to_tier_3),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name}={value} must be within [0.0, 1.0]"
                )
        if self.escalate_to_tier_3 < self.escalate_to_tier_2:
            raise ValueError(
                f"escalate_to_tier_3 ({self.escalate_to_tier_3}) must be >= "
                f"escalate_to_tier_2 ({self.escalate_to_tier_2}); otherwise a "
                "request could reach T3 without passing the T2 gate."
            )


class TieredRouter:
    """Cascading router that escalates only when cheaper tiers flag risk.

    Medium and agent backends are optional — a cheap-only configuration is a
    valid MVP, and a cheap+medium configuration is a valid pre-agent rollout.
    The router degrades gracefully: missing tiers simply cap the maximum
    reachable tier_reached.
    """

    def __init__(
        self,
        cheap: ModelBackend,
        medium: ModelBackend | None = None,
        agent: ModelBackend | None = None,
        config: TieredRouterConfig | None = None,
    ):
        self.cheap = cheap
        self.medium = medium
        self.agent = agent
        self.config = config or TieredRouterConfig()

    def score(self, request: PredictionRequest) -> PredictionResponse:
        t1 = self.cheap.predict(request)
        total_latency = t1.latency_ms

        if (
            t1.risk_score < self.config.escalate_to_tier_2
            or self.medium is None
        ):
            return PredictionResponse(
                risk_score=t1.risk_score,
                backend_used=t1.backend_used,
                latency_ms=total_latency,
                reasoning=t1.reasoning,
                raw=t1.raw,
                tier_reached=1,
            )

        t2 = self.medium.predict(request)
        total_latency += t2.latency_ms

        if (
            t2.risk_score < self.config.escalate_to_tier_3
            or self.agent is None
        ):
            return PredictionResponse(
                risk_score=t2.risk_score,
                backend_used=t2.backend_used,
                latency_ms=total_latency,
                reasoning=t2.reasoning,
                raw=t2.raw,
                tier_reached=2,
            )

        t3 = self.agent.predict(request)
        total_latency += t3.latency_ms
        return PredictionResponse(
            risk_score=t3.risk_score,
            backend_used=t3.backend_used,
            latency_ms=total_latency,
            reasoning=t3.reasoning,
            raw=t3.raw,
            tier_reached=3,
        )
