"""Tests for the TieredRouter.

The router is the project's cost-engineering layer: 95% of commits should
finish at Tier 1 (cheap ML predictor), ~20% should pay for Tier 2 (LLM
judge), and only the riskiest ~5% should pay for Tier 3 (agent
investigation). These tests pin the escalation contract and the
configurability that lets a team retune for its own traffic distribution.

Backends are mocked here — the router's job is *routing*, not prediction.
End-to-end backend tests live alongside each backend implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.models.gateway import (
    Backend,
    ModelBackend,
    PredictionRequest,
    PredictionResponse,
)
from src.models.tiered_router import TieredRouter, TieredRouterConfig


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class _StubBackend(ModelBackend):
    """Returns a fixed score and tracks invocation count."""

    score: float
    backend_id: Backend = Backend.CLAUDE
    latency_ms: float = 1.0
    reasoning: str | None = None
    calls: int = 0

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        self.calls += 1
        return PredictionResponse(
            risk_score=self.score,
            backend_used=self.backend_id,
            latency_ms=self.latency_ms,
            reasoning=self.reasoning,
        )

    def name(self) -> Backend:
        return self.backend_id


def _request() -> PredictionRequest:
    return PredictionRequest(diff="--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n")


# ---------------------------------------------------------------------------
# Escalation contract — the heart of the router
# ---------------------------------------------------------------------------


def test_low_t1_score_does_not_escalate():
    """Score < escalate_to_tier_2 must short-circuit at Tier 1."""
    cheap = _StubBackend(score=0.30, backend_id=Backend.TRITON_NEMO)
    medium = _StubBackend(score=0.99, backend_id=Backend.CLAUDE)
    agent = _StubBackend(score=0.99, backend_id=Backend.CLAUDE)

    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    resp = router.score(_request())

    assert resp.tier_reached == 1
    assert resp.risk_score == 0.30
    assert resp.backend_used == Backend.TRITON_NEMO
    assert cheap.calls == 1
    assert medium.calls == 0
    assert agent.calls == 0


def test_medium_t1_score_escalates_to_t2_only():
    """T1 >= 0.6 invokes T2; if T2 < 0.8, stop there."""
    cheap = _StubBackend(score=0.70, backend_id=Backend.TRITON_NEMO)
    medium = _StubBackend(score=0.65, backend_id=Backend.CLAUDE)
    agent = _StubBackend(score=0.99, backend_id=Backend.CLAUDE)

    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    resp = router.score(_request())

    assert resp.tier_reached == 2
    assert resp.risk_score == 0.65  # T2's score is the final answer
    assert resp.backend_used == Backend.CLAUDE
    assert cheap.calls == 1
    assert medium.calls == 1
    assert agent.calls == 0


def test_high_t2_score_escalates_to_t3():
    """T1 >= 0.6 AND T2 >= 0.8 invokes the agent (Tier 3)."""
    cheap = _StubBackend(score=0.70, backend_id=Backend.TRITON_NEMO)
    medium = _StubBackend(score=0.85, backend_id=Backend.CLAUDE)
    agent = _StubBackend(score=0.92, backend_id=Backend.CLAUDE, reasoning="investigated")

    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    resp = router.score(_request())

    assert resp.tier_reached == 3
    assert resp.risk_score == 0.92
    assert resp.reasoning == "investigated"
    assert cheap.calls == 1
    assert medium.calls == 1
    assert agent.calls == 1


def test_t2_can_lower_score_below_t1():
    """A semantic re-read may decide the commit is safer than structural features
    suggested — the router must propagate T2's score even if it's lower."""
    cheap = _StubBackend(score=0.75, backend_id=Backend.TRITON_NEMO)
    medium = _StubBackend(score=0.20, backend_id=Backend.CLAUDE)
    agent = _StubBackend(score=0.99, backend_id=Backend.CLAUDE)

    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    resp = router.score(_request())

    assert resp.tier_reached == 2
    assert resp.risk_score == 0.20
    assert agent.calls == 0


# ---------------------------------------------------------------------------
# Boundary values — exact thresholds resolve up, not down
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t1_score, expected_tier",
    [
        (0.599, 1),
        (0.600, 2),  # exact threshold escalates
        (0.601, 2),
    ],
)
def test_t1_threshold_boundary(t1_score, expected_tier):
    cheap = _StubBackend(score=t1_score)
    medium = _StubBackend(score=0.0)  # never escalates further
    router = TieredRouter(cheap=cheap, medium=medium)
    assert router.score(_request()).tier_reached == expected_tier


@pytest.mark.parametrize(
    "t2_score, expected_tier",
    [
        (0.799, 2),
        (0.800, 3),  # exact threshold escalates
        (0.801, 3),
    ],
)
def test_t2_threshold_boundary(t2_score, expected_tier):
    cheap = _StubBackend(score=0.70)  # forces T2
    medium = _StubBackend(score=t2_score)
    agent = _StubBackend(score=0.99)
    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    assert router.score(_request()).tier_reached == expected_tier


# ---------------------------------------------------------------------------
# Missing-tier degradation — router still works without medium/agent
# ---------------------------------------------------------------------------


def test_no_medium_backend_caps_at_tier_1():
    """Cheap-only deployments are a valid configuration (e.g., MVP)."""
    cheap = _StubBackend(score=0.95)
    router = TieredRouter(cheap=cheap)
    resp = router.score(_request())
    assert resp.tier_reached == 1
    assert resp.risk_score == 0.95


def test_no_agent_backend_caps_at_tier_2():
    """Skipping the agent tier saves cost when investigation isn't wanted yet."""
    cheap = _StubBackend(score=0.70)
    medium = _StubBackend(score=0.95)
    router = TieredRouter(cheap=cheap, medium=medium)
    resp = router.score(_request())
    assert resp.tier_reached == 2
    assert resp.risk_score == 0.95


# ---------------------------------------------------------------------------
# Latency aggregation — sum across invoked tiers
# ---------------------------------------------------------------------------


def test_latency_sums_across_invoked_tiers():
    cheap = _StubBackend(score=0.70, latency_ms=5.0)
    medium = _StubBackend(score=0.90, latency_ms=2000.0)
    agent = _StubBackend(score=0.95, latency_ms=25000.0)
    router = TieredRouter(cheap=cheap, medium=medium, agent=agent)
    resp = router.score(_request())
    assert resp.latency_ms == pytest.approx(5.0 + 2000.0 + 25000.0)


def test_latency_excludes_uncalled_tiers():
    cheap = _StubBackend(score=0.10, latency_ms=5.0)
    medium = _StubBackend(score=0.99, latency_ms=2000.0)
    router = TieredRouter(cheap=cheap, medium=medium)
    resp = router.score(_request())
    assert resp.latency_ms == 5.0


# ---------------------------------------------------------------------------
# Configurability — strict org / lax org retuning
# ---------------------------------------------------------------------------


def test_custom_thresholds_strict_org():
    """A strict org escalates earlier (more PRs hit the agent)."""
    cfg = TieredRouterConfig(escalate_to_tier_2=0.30, escalate_to_tier_3=0.50)
    cheap = _StubBackend(score=0.35)
    medium = _StubBackend(score=0.55)
    agent = _StubBackend(score=0.60)
    router = TieredRouter(cheap=cheap, medium=medium, agent=agent, config=cfg)
    assert router.score(_request()).tier_reached == 3


def test_threshold_validation_rejects_inverted_order():
    """t3 threshold must be >= t2 threshold; otherwise a PR could escalate
    to T3 without first passing the T2 gate, which violates the contract."""
    with pytest.raises(ValueError, match="escalate_to_tier_3"):
        TieredRouterConfig(escalate_to_tier_2=0.8, escalate_to_tier_3=0.5)


@pytest.mark.parametrize("bad_value", [-0.1, 1.5])
def test_threshold_validation_rejects_out_of_range(bad_value):
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        TieredRouterConfig(escalate_to_tier_2=bad_value)
