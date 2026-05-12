"""Tests for the explanation writer.

The deterministic renderer is part of the audit-trail surface — its output
appears verbatim on PRs. Tests pin the structure so a future refactor cannot
silently drop a section.
"""
from __future__ import annotations

from src.agent.explainer import ExplanationWriter
from src.agent.harness import HarnessResult
from src.agent.policy import PolicyDecision
from src.agent.sub_agents import SubAgentReport


def _result_with_factors(score: float) -> HarnessResult:
    return HarnessResult(
        risk_score=score,
        sub_agent_reports=[
            SubAgentReport(
                sub_agent_name="diff-analyzer",
                observations={"files_touched": 12, "additions": 340},
                risk_factors=["large fanout: 12 files touched"],
                confidence=0.7,
            ),
            SubAgentReport(
                sub_agent_name="ownership-mapper",
                observations={"recommended_reviewers": ["@security", "@auth"]},
                risk_factors=["bus-factor risk: single owner covers all files"],
                confidence=0.6,
            ),
            # A pure-stub report — should be excluded from evidence.
            SubAgentReport(
                sub_agent_name="historical-context",
                observations={"status": "stub"},
                risk_factors=[],
                confidence=0.0,
            ),
        ],
    )


def _decision(action: str = "sme_review", risk_level: str = "High") -> PolicyDecision:
    return PolicyDecision(
        action=action,
        risk_level=risk_level,
        rationale="Risk score 0.65 → High band → action: sme review.",
        recommended_steps=["Require SME reviewer", "Run extended CI"],
    )


# ---------------------------------------------------------------------------
# Structural markdown checks
# ---------------------------------------------------------------------------


def test_render_starts_with_headline():
    output = ExplanationWriter(include_emoji=False).render(_result_with_factors(0.65), _decision())
    first_line = output.splitlines()[0]
    assert first_line.startswith("## ")
    assert "**High**" in first_line
    assert "0.65" in first_line


def test_render_includes_recommended_action():
    output = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert "**sme review**" in output


def test_render_lists_top_risk_factors():
    output = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert "**Top risk factors:**" in output
    assert "large fanout" in output
    assert "bus-factor" in output


def test_render_lists_recommended_steps():
    output = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert "**Recommended next steps:**" in output
    for step in _decision().recommended_steps:
        assert step in output


def test_render_excludes_pure_stub_reports_from_evidence():
    """A zero-confidence report with no risk factors adds no value to the PR comment."""
    output = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert "diff-analyzer" in output
    assert "ownership-mapper" in output
    assert "historical-context" not in output


def test_render_includes_audit_footer():
    """The footer points at the project + safety doc — required for auditability."""
    output = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert "commit-risk-scorer" in output
    assert "enterprise-safety" in output
    assert "advisory" in output.lower()


def test_emoji_toggle():
    with_emoji = ExplanationWriter(include_emoji=True).render(_result_with_factors(0.65), _decision())
    without_emoji = ExplanationWriter(include_emoji=False).render(_result_with_factors(0.65), _decision())
    # Without emoji the headline has fewer characters before "Commit risk"
    assert "🟠" in with_emoji
    assert "🟠" not in without_emoji


def test_render_is_deterministic():
    """Same inputs → same output (required for golden-file comparison in CI)."""
    a = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    b = ExplanationWriter().render(_result_with_factors(0.65), _decision())
    assert a == b
