"""Tests for the FeedbackLog — closing the loop from prediction → outcome.

This is the data substrate for the JD bullet:

    "Continuously measure and report on the impact of AI interventions,
     showing progress in metrics such as cycle time, change failure rate,
     and mean time to recovery (MTTR)."

The log records two event types — predictions (when the agent scores a PR) and
outcomes (observed 7 days later: clean / reverted / hotfixed / incident) — and
joins them by `pr_id`. Tests pin down the join semantics, the MTTR / CFR
computations, and the file-roundtrip contract.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.storage.feedback_log import (
    FeedbackLog,
    Outcome,
    OutcomeLabel,
    Prediction,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _t(minutes: int = 0) -> datetime:
    """A fixed reference timestamp offset by `minutes`."""
    base = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(minutes=minutes)


def _pred(pr_id: str, score: float = 0.5, tier: int = 1, at: datetime | None = None) -> Prediction:
    return Prediction(
        pr_id=pr_id,
        risk_score=score,
        tier_reached=tier,
        predicted_at=at or _t(0),
    )


def _outcome(
    pr_id: str,
    label: OutcomeLabel = OutcomeLabel.CLEAN,
    mttr_minutes: int | None = None,
    at: datetime | None = None,
) -> Outcome:
    return Outcome(
        pr_id=pr_id,
        label=label,
        mttr_minutes=mttr_minutes,
        observed_at=at or _t(60 * 24 * 7),  # +7 days by default
    )


# ---------------------------------------------------------------------------
# Roundtrip & file persistence
# ---------------------------------------------------------------------------


def test_prediction_and_outcome_roundtrip(tmp_path: Path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-1", score=0.42))
    log.record_outcome(_outcome("PR-1", label=OutcomeLabel.CLEAN))

    pairs = list(log.iter_labeled())
    assert len(pairs) == 1
    assert pairs[0].prediction.pr_id == "PR-1"
    assert pairs[0].prediction.risk_score == 0.42
    assert pairs[0].outcome.label == OutcomeLabel.CLEAN


def test_log_survives_reopen(tmp_path: Path):
    """JSONL persistence — a fresh FeedbackLog instance over the same file
    must see entries written by a previous instance."""
    path = tmp_path / "feedback.jsonl"
    FeedbackLog(path).record_prediction(_pred("PR-1"))
    FeedbackLog(path).record_outcome(_outcome("PR-1"))

    pairs = list(FeedbackLog(path).iter_labeled())
    assert len(pairs) == 1
    assert pairs[0].prediction.pr_id == "PR-1"


# ---------------------------------------------------------------------------
# Join semantics — the core contract
# ---------------------------------------------------------------------------


def test_predictions_without_outcomes_are_skipped(tmp_path: Path):
    """Unlabeled predictions are not training data yet — must not be yielded."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-1"))
    log.record_prediction(_pred("PR-2"))
    log.record_outcome(_outcome("PR-2"))

    pairs = list(log.iter_labeled())
    assert len(pairs) == 1
    assert pairs[0].prediction.pr_id == "PR-2"


def test_orphan_outcomes_are_skipped(tmp_path: Path):
    """An outcome without a matching prediction is dropped (the agent didn't
    score this PR — it's not our training signal even if labeled)."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_outcome(_outcome("PR-ghost"))
    assert list(log.iter_labeled()) == []


def test_repeat_predictions_resolve_to_latest(tmp_path: Path):
    """If the agent scored a PR twice (e.g., after a force-push), the outcome
    is matched against the *most recent* prediction — that's the score the
    reviewer actually saw."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-1", score=0.30, at=_t(0)))
    log.record_prediction(_pred("PR-1", score=0.75, at=_t(60)))
    log.record_outcome(_outcome("PR-1", label=OutcomeLabel.REVERTED))

    pairs = list(log.iter_labeled())
    assert len(pairs) == 1
    assert pairs[0].prediction.risk_score == 0.75


# ---------------------------------------------------------------------------
# DORA metrics — the JD's literal asks
# ---------------------------------------------------------------------------


def test_change_failure_rate(tmp_path: Path):
    """CFR = (reverted + hotfixed + incident) / total_labeled. JD wording:
    'change failure rate'."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    for i, label in enumerate(
        [
            OutcomeLabel.CLEAN,
            OutcomeLabel.CLEAN,
            OutcomeLabel.CLEAN,
            OutcomeLabel.REVERTED,
            OutcomeLabel.HOTFIXED,
        ]
    ):
        pr = f"PR-{i}"
        log.record_prediction(_pred(pr))
        log.record_outcome(_outcome(pr, label=label))

    assert log.change_failure_rate() == pytest.approx(2 / 5)


def test_change_failure_rate_undefined_when_no_data(tmp_path: Path):
    """No predictions yet → CFR is None, not a division-by-zero."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    assert log.change_failure_rate() is None


def test_mttr_median_minutes(tmp_path: Path):
    """MTTR = median minutes-to-recovery across hotfixed/reverted outcomes;
    CLEAN entries do not contribute. JD wording: 'mean time to recovery'."""
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    for pr_id, label, mttr in [
        ("PR-1", OutcomeLabel.CLEAN, None),
        ("PR-2", OutcomeLabel.REVERTED, 30),
        ("PR-3", OutcomeLabel.HOTFIXED, 90),
        ("PR-4", OutcomeLabel.REVERTED, 240),
    ]:
        log.record_prediction(_pred(pr_id))
        log.record_outcome(_outcome(pr_id, label=label, mttr_minutes=mttr))

    # median of (30, 90, 240) = 90
    assert log.mttr_median_minutes() == 90.0


def test_mttr_undefined_when_only_clean_outcomes(tmp_path: Path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-1"))
    log.record_outcome(_outcome("PR-1", label=OutcomeLabel.CLEAN))
    assert log.mttr_median_minutes() is None


# ---------------------------------------------------------------------------
# Time filtering — "show me the trailing 30 days" kind of query
# ---------------------------------------------------------------------------


def test_iter_labeled_filters_by_since(tmp_path: Path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-old", at=_t(0)))
    log.record_outcome(_outcome("PR-old", at=_t(60)))
    log.record_prediction(_pred("PR-new", at=_t(1000)))
    log.record_outcome(_outcome("PR-new", at=_t(1060)))

    recent = list(log.iter_labeled(since=_t(500)))
    assert [p.prediction.pr_id for p in recent] == ["PR-new"]


def test_change_failure_rate_respects_since(tmp_path: Path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.record_prediction(_pred("PR-old", at=_t(0)))
    log.record_outcome(_outcome("PR-old", label=OutcomeLabel.CLEAN, at=_t(60)))
    log.record_prediction(_pred("PR-new", at=_t(1000)))
    log.record_outcome(_outcome("PR-new", label=OutcomeLabel.REVERTED, at=_t(1060)))

    # All time: 1/2 = 0.5; since _t(500): only PR-new (failed) → 1.0.
    assert log.change_failure_rate() == 0.5
    assert log.change_failure_rate(since=_t(500)) == 1.0


# ---------------------------------------------------------------------------
# Validation — defend against bad inputs at the boundary
# ---------------------------------------------------------------------------


def test_prediction_rejects_out_of_range_score():
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        Prediction(pr_id="PR-1", risk_score=1.5, tier_reached=1, predicted_at=_t(0))


def test_prediction_rejects_invalid_tier():
    with pytest.raises(ValueError, match="tier_reached"):
        Prediction(pr_id="PR-1", risk_score=0.5, tier_reached=4, predicted_at=_t(0))
