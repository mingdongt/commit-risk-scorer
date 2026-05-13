"""Tests for the agent harness.

These are the seed for the regression-gated CI described in docs/design-doc.md
§Eval Methodology — every push runs them, and any failure blocks merge.
"""
from __future__ import annotations

from src.agent.harness import AgentHarness, HarnessResult
from src.agent.sub_agents import (
    AgentPRAuditor,
    DiffAnalyzer,
    HistoricalContext,
    OwnershipMapper,
    TestImpactScout,
)
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


def test_agent_pr_auditor_detects_bot_login():
    """A bot-shaped author login → classified as agent-generated."""
    auditor = AgentPRAuditor()
    report = auditor.analyze(
        "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        metadata={"author": {"login": "copilot-coding[bot]"}},
    )
    assert report.observations["author_class"] == "agent-generated"


def test_agent_pr_auditor_detects_commit_burst():
    """5+ commits within 5 minutes → classified as agent-generated."""
    auditor = AgentPRAuditor()
    # 6 commits over 200 seconds — humans would take orders of magnitude longer
    timestamps = [1000.0, 1050.0, 1100.0, 1150.0, 1180.0, 1200.0]
    report = auditor.analyze(
        "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        metadata={"commit_timestamps": timestamps},
    )
    assert report.observations["author_class"] == "agent-generated"


def test_agent_pr_auditor_respects_explicit_class():
    """Explicit author_class wins over heuristic inference."""
    auditor = AgentPRAuditor()
    report = auditor.analyze(
        "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        metadata={
            "author_class": "ai-assisted",
            "author": {"login": "copilot-bot"},  # would otherwise infer agent-generated
        },
    )
    assert report.observations["author_class"] == "ai-assisted"


def test_agent_pr_auditor_flags_mechanical_refactor():
    """Agent-authored diff with high lines-per-file ratio → mechanical-refactor risk."""
    auditor = AgentPRAuditor()
    # Build a diff with 6 files × ~100 lines each = ~600 lines total
    diff_parts = []
    for i in range(6):
        diff_parts.append(f"--- a/src/m{i}.py\n+++ b/src/m{i}.py\n@@ -1,100 +1,100 @@\n")
        diff_parts.extend(f"-old_{i}_{j}\n+new_{i}_{j}\n" for j in range(50))
    diff = "".join(diff_parts)
    report = auditor.analyze(diff, metadata={"author_class": "agent-generated"})
    assert any("mechanical refactor" in rf for rf in report.risk_factors)


def test_agent_pr_auditor_flags_missing_human_rationale():
    """Agent-authored PR with only one-liner commit messages → missing-rationale risk."""
    auditor = AgentPRAuditor()
    report = auditor.analyze(
        "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        metadata={
            "author_class": "agent-generated",
            "commit_messages": ["fix typo", "wip", "update"],
        },
    )
    assert any("missing human rationale" in rf for rf in report.risk_factors)


def test_agent_pr_auditor_flags_scope_drift():
    """PR description naming paths not in the diff → scope-drift risk."""
    auditor = AgentPRAuditor()
    report = auditor.analyze(
        "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n",
        metadata={
            "author_class": "agent-generated",
            "pr_description_paths": ["src/foo.py", "src/totally_unmodified.py"],
        },
    )
    assert any("scope drift" in rf for rf in report.risk_factors)


def test_agent_pr_auditor_silent_for_human_only():
    """Agent-specific risk factors do not fire for human-authored PRs."""
    auditor = AgentPRAuditor()
    # Large mechanical-looking diff but human author → no agent risk factors
    diff_parts = []
    for i in range(6):
        diff_parts.append(f"--- a/src/m{i}.py\n+++ b/src/m{i}.py\n@@ -1,100 +1,100 @@\n")
        diff_parts.extend(f"-old\n+new\n" for _ in range(50))
    diff = "".join(diff_parts)
    report = auditor.analyze(diff, metadata={"author_class": "human-only"})
    assert not any("agent-authored" in rf for rf in report.risk_factors)


def test_harness_score_returns_valid_result():
    """Harness produces a HarnessResult with a valid risk score in [0, 1]."""
    harness = AgentHarness()
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    result = harness.score(diff)
    assert isinstance(result, HarnessResult)
    assert 0.0 <= result.risk_score <= 1.0
    # Default sub-agent set is now 5
    # (diff-analyzer + ownership-mapper + agent-pr-auditor + 2 stubs)
    assert len(result.sub_agent_reports) == 5


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


# ---------------------------------------------------------------------------
# Harness + LLM judge wiring (Tier 2)
# ---------------------------------------------------------------------------


class _StubJudge:
    """Records the PredictionRequest it received and returns a canned verdict."""

    def __init__(self, *, risk_score=0.81, reasoning="canned-judge-reasoning", raise_exc=None):
        self._risk_score = risk_score
        self._reasoning = reasoning
        self._raise = raise_exc
        self.last_request = None

    def predict(self, request):
        self.last_request = request
        if self._raise is not None:
            raise self._raise
        # Mimic the PredictionResponse surface the harness reads.
        return type(
            "R",
            (),
            {"risk_score": self._risk_score, "reasoning": self._reasoning},
        )()


def test_harness_uses_judge_score_when_injected():
    """With a judge injected, the harness returns the judge's score, not the heuristic."""
    judge = _StubJudge(risk_score=0.81, reasoning="auth-predicate weakened")
    harness = AgentHarness(judge_backend=judge)
    diff = (
        "--- a/src/auth/session.py\n+++ b/src/auth/session.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    result = harness.score(diff)

    assert result.risk_score == 0.81
    assert result.reasoning == "auth-predicate weakened"
    # sub-agent reports are still attached (harness did not skip them)
    assert len(result.sub_agent_reports) == 5


def test_harness_forwards_sub_agent_reports_to_judge():
    """The PredictionRequest the judge sees must include dict-shaped sub-agent reports."""
    judge = _StubJudge()
    harness = AgentHarness(judge_backend=judge)
    harness.score("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n")

    req = judge.last_request
    assert req is not None
    assert len(req.sub_agent_reports) == 5
    # Each report must be a dict with the four canonical keys.
    for report in req.sub_agent_reports:
        assert set(report.keys()) == {"name", "confidence", "observations", "risk_factors"}
    names = {r["name"] for r in req.sub_agent_reports}
    assert {"diff-analyzer", "ownership-mapper", "agent-pr-auditor"} <= names


def test_harness_falls_back_to_heuristic_on_judge_failure():
    """A judge that raises must not break the CI step — fall back to heuristic + log."""
    judge = _StubJudge(raise_exc=RuntimeError("transient API blip"))
    harness = AgentHarness(judge_backend=judge)
    result = harness.score("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n")

    # Heuristic score is still valid + reasoning records the fallback reason.
    assert 0.0 <= result.risk_score <= 1.0
    assert result.reasoning is not None
    assert "judge unavailable" in result.reasoning
    assert "transient API blip" in result.reasoning


def test_harness_without_judge_preserves_v0_1_behavior():
    """No judge injected → reasoning stays None, score is the heuristic. v0.1 unchanged."""
    harness = AgentHarness()
    result = harness.score("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert result.reasoning is None
