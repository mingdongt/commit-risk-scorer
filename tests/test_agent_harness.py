"""Tests for the agent harness.

These are the seed for the regression-gated CI described in docs/design-doc.md
§Eval Methodology — every push runs them, and any failure blocks merge.
"""
from __future__ import annotations

from src.agent.harness import AgentHarness, HarnessResult
from src.agent.sub_agents import DiffAnalyzer, HistoricalContext, OwnershipMapper, TestImpactScout
from src.agent.tools import git_diff_stats


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_git_diff_stats_basic():
    """git_diff_stats parses a simple unified diff correctly."""
    diff = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
    )
    stats = git_diff_stats(diff)
    assert stats["files_touched"] == 1
    assert stats["additions"] == 1
    assert stats["deletions"] == 1
    assert stats["files"] == ["src/foo.py"]


def test_git_diff_stats_multifile():
    """Multi-file diff → multiple files counted."""
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    stats = git_diff_stats(diff)
    assert stats["files_touched"] == 2
    assert stats["additions"] == 2
    assert stats["deletions"] == 2


def test_git_diff_stats_empty():
    """Empty diff → zero everywhere."""
    stats = git_diff_stats("")
    assert stats["files_touched"] == 0
    assert stats["additions"] == 0
    assert stats["deletions"] == 0


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------


def test_diff_analyzer_flags_large_diff():
    """DiffAnalyzer flags diffs above the LARGE_DIFF_LINES threshold."""
    analyzer = DiffAnalyzer()
    additions = "\n".join(f"+line_{i}" for i in range(600))
    diff = f"--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,600 @@\n{additions}\n"
    report = analyzer.analyze(diff, metadata={})
    assert any("very large diff" in rf for rf in report.risk_factors)
    assert report.confidence >= 0.5


def test_diff_analyzer_no_flag_for_small_diff():
    """DiffAnalyzer raises no risk factors on a small diff."""
    analyzer = DiffAnalyzer()
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    report = analyzer.analyze(diff, metadata={})
    assert report.risk_factors == []


def test_stub_subagents_return_zero_confidence():
    """Stubs report zero confidence so the harness ignores them in aggregation."""
    for stub in (TestImpactScout(), HistoricalContext()):
        report = stub.analyze("", metadata={})
        assert report.confidence == 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def test_ownership_mapper_uses_longest_prefix_match():
    """CODEOWNERS-style longest-prefix match — more specific path wins."""
    mapper = OwnershipMapper()
    diff = "--- a/src/auth/session.py\n+++ b/src/auth/session.py\n@@ -1 +1 @@\n-x\n+y\n"
    codeowners = {
        "src/": ["@platform-team"],
        "src/auth/": ["@security-team", "@auth-owner"],
    }
    report = mapper.analyze(diff, metadata={"codeowners": codeowners})
    assert report.observations["recommended_reviewers"] == ["@auth-owner", "@security-team"]


def test_ownership_mapper_flags_ownership_gap():
    """Files matching no CODEOWNERS prefix surface as an ownership-gap risk factor."""
    mapper = OwnershipMapper()
    diff = "--- a/unowned/foo.py\n+++ b/unowned/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
    report = mapper.analyze(diff, metadata={"codeowners": {"src/": ["@team"]}})
    assert any("ownership gap" in rf for rf in report.risk_factors)


def test_ownership_mapper_flags_bus_factor():
    """A single owner covering 3+ files is a bus-factor risk."""
    mapper = OwnershipMapper()
    diff = (
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/src/c.py\n+++ b/src/c.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    report = mapper.analyze(diff, metadata={"codeowners": {"src/": ["@solo-owner"]}})
    assert any("bus-factor" in rf for rf in report.risk_factors)


def test_ownership_mapper_handles_empty_diff():
    """No files → zero confidence, no false-positive risk factors."""
    report = OwnershipMapper().analyze("", metadata={"codeowners": {}})
    assert report.confidence == 0.0
    assert report.risk_factors == []


def test_harness_score_returns_valid_result():
    """Harness produces a HarnessResult with a valid risk score in [0, 1]."""
    harness = AgentHarness()
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    result = harness.score(diff)
    assert isinstance(result, HarnessResult)
    assert 0.0 <= result.risk_score <= 1.0
    # Default sub-agent set is now 4 (diff-analyzer + ownership-mapper + 2 stubs)
    assert len(result.sub_agent_reports) == 4


def test_harness_empty_diff_low_score():
    """An empty diff has no risk factors → low score from the diff-analyzer."""
    harness = AgentHarness()
    result = harness.score("")
    assert result.risk_score < 0.2


def test_harness_serializable():
    """HarnessResult.to_dict produces JSON-friendly output."""
    import json

    harness = AgentHarness()
    result = harness.score("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n")
    payload = result.to_dict()
    json.dumps(payload)  # raises if anything is non-serializable
    assert payload["risk_score"] is not None
    assert isinstance(payload["sub_agent_reports"], list)
