"""Tests for the policy gatekeeper.

The policy is the agent's user-visible contract — these tests make changes
explicit (any band re-tuning shows up as a test diff).
"""
from __future__ import annotations

import pytest

from src.agent.harness import HarnessResult
from src.agent.policy import (
    DEFAULT_RECOMMENDED_STEPS,
    DEFAULT_THRESHOLDS,
    PolicyGatekeeper,
)
from src.agent.sub_agents import SubAgentReport


def _result(score: float, factors: list[str] | None = None) -> HarnessResult:
    return HarnessResult(
        risk_score=score,
        sub_agent_reports=[
            SubAgentReport(
                sub_agent_name="test",
                observations={},
                risk_factors=factors or [],
                confidence=0.5,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Risk-band -> action mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected_action, expected_level",
    [
        (0.00, "fast_track", "Low"),
        (0.10, "fast_track", "Low"),
        (0.19, "fast_track", "Low"),
        (0.20, "owner_review", "Medium"),
        (0.49, "owner_review", "Medium"),
        (0.50, "sme_review", "High"),
        (0.79, "sme_review", "High"),
        (0.80, "block_merge", "Critical"),
        (1.00, "block_merge", "Critical"),
    ],
)
def test_score_lands_in_correct_band(score, expected_action, expected_level):
    decision = PolicyGatekeeper().decide(_result(score))
    assert decision.action == expected_action
    assert decision.risk_level == expected_level


# ---------------------------------------------------------------------------
# Rationale content
# ---------------------------------------------------------------------------


def test_rationale_mentions_score_and_action():
    decision = PolicyGatekeeper().decide(_result(0.42))
    assert "0.42" in decision.rationale
    assert "owner review" in decision.rationale  # action.replace("_", " ")


def test_rationale_surfaces_top_risk_factors():
    factors = ["large fanout: 12 files", "touches auth module", "no tests modified"]
    decision = PolicyGatekeeper().decide(_result(0.65, factors=factors))
    for f in factors[:3]:
        assert f in decision.rationale


def test_rationale_handles_zero_risk_factors():
    decision = PolicyGatekeeper().decide(_result(0.10))
    assert "No specific risk factors" in decision.rationale


# ---------------------------------------------------------------------------
# Threshold overrides
# ---------------------------------------------------------------------------


def test_custom_thresholds_strict_org():
    """A strict org might block at 0.5 instead of 0.8."""
    strict = {
        "fast_track": (0.00, 0.10),
        "owner_review": (0.10, 0.30),
        "sme_review": (0.30, 0.50),
        "block_merge": (0.50, 1.01),
    }
    gate = PolicyGatekeeper(thresholds=strict)
    assert gate.decide(_result(0.55)).action == "block_merge"
    assert gate.decide(_result(0.40)).action == "sme_review"


def test_overlapping_thresholds_rejected():
    """Overlapping bands are a config bug — the gatekeeper must refuse to load."""
    bad = {
        "fast_track": (0.00, 0.30),
        "owner_review": (0.20, 0.50),  # overlaps fast_track
    }
    with pytest.raises(ValueError, match="overlapping"):
        PolicyGatekeeper(thresholds=bad)


# ---------------------------------------------------------------------------
# Recommended steps
# ---------------------------------------------------------------------------


def test_recommended_steps_match_action():
    for action, expected_steps in DEFAULT_RECOMMENDED_STEPS.items():
        # Build a score sitting inside the band for this action.
        lo, hi = DEFAULT_THRESHOLDS[action]
        mid = (lo + min(hi, 1.0)) / 2
        decision = PolicyGatekeeper().decide(_result(mid))
        assert decision.recommended_steps == expected_steps


def test_score_clamped_to_unit_interval():
    """Out-of-range scores are clamped (defensive — shouldn't happen but cheap to handle)."""
    # > 1.0 → still lands in block_merge
    assert PolicyGatekeeper().decide(_result(1.5)).action == "block_merge"
    # < 0.0 → fast_track
    assert PolicyGatekeeper().decide(_result(-0.1)).action == "fast_track"
