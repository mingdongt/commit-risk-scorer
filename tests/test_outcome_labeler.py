"""Tests for `derive_outcome` and `relabel_pending`.

The priority hierarchy is part of the public contract: a PR that triggers
an incident must NOT be silently relabeled as merely 'hotfixed' just
because a hotfix also landed.

The relabel-pending tests pin the orchestrator's three skip conditions:
already-labeled predictions, predictions still inside the observation
window, and predictions whose signals aren't yet available from the
fetcher.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.storage.feedback_log import (
    FeedbackLog,
    OutcomeLabel,
    Prediction,
)
from src.storage.outcome_labeler import (
    FetchedSignals,
    derive_outcome,
    relabel_pending,
)


_OBSERVED_AT = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def test_no_failure_signal_yields_clean():
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=False,
        hotfix_within_window=False,
        incident_linked=False,
    )
    assert o.label == OutcomeLabel.CLEAN
    assert o.mttr_minutes is None


def test_hotfix_only_yields_hotfixed():
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=False,
        hotfix_within_window=True,
        incident_linked=False,
        mttr_minutes=45,
    )
    assert o.label == OutcomeLabel.HOTFIXED
    assert o.mttr_minutes == 45


def test_revert_beats_hotfix():
    """A revert is a stronger failure signal than a follow-up hotfix; if both
    fired, the revert label is what we record (avoid double-counting)."""
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=True,
        hotfix_within_window=True,
        incident_linked=False,
        mttr_minutes=30,
    )
    assert o.label == OutcomeLabel.REVERTED


def test_incident_beats_revert():
    """A linked production incident is the most severe — it outranks revert
    even if both happened. The label drives MTTR and CFR, so the more
    consequential signal must win."""
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=True,
        hotfix_within_window=False,
        incident_linked=True,
        mttr_minutes=120,
    )
    assert o.label == OutcomeLabel.INCIDENT


def test_clean_outcome_drops_mttr_even_if_provided():
    """A clean PR has no recovery to time. If a caller passes MTTR by accident,
    we drop it rather than persist a meaningless value."""
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=False,
        hotfix_within_window=False,
        incident_linked=False,
        mttr_minutes=999,
    )
    assert o.label == OutcomeLabel.CLEAN
    assert o.mttr_minutes is None


@pytest.mark.parametrize(
    "was_reverted, hotfix, incident, expected",
    [
        (True, False, False, OutcomeLabel.REVERTED),
        (False, True, False, OutcomeLabel.HOTFIXED),
        (False, False, True, OutcomeLabel.INCIDENT),
        (True, True, False, OutcomeLabel.REVERTED),   # revert > hotfix
        (False, True, True, OutcomeLabel.INCIDENT),   # incident > hotfix
        (True, True, True, OutcomeLabel.INCIDENT),    # incident > all
    ],
)
def test_priority_hierarchy_table(was_reverted, hotfix, incident, expected):
    o = derive_outcome(
        pr_id="PR-1",
        observed_at=_OBSERVED_AT,
        was_reverted=was_reverted,
        hotfix_within_window=hotfix,
        incident_linked=incident,
    )
    assert o.label == expected


# ---------------------------------------------------------------------------
# relabel_pending — the orchestrator's skip conditions
# ---------------------------------------------------------------------------


def _at(minutes: int = 0) -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC) + timedelta(minutes=minutes)


def test_relabel_pending_labels_old_unobserved_predictions(tmp_path: Path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.record_prediction(
        Prediction(pr_id="PR-1", risk_score=0.7, tier_reached=2, predicted_at=_at(0))
    )
    # 14 days later, the nightly job runs.
    now_fn = lambda: _at(60 * 24 * 14)  # noqa: E731

    def fetcher(pr_id: str, _: datetime) -> FetchedSignals | None:
        assert pr_id == "PR-1"
        return FetchedSignals(
            was_reverted=True, hotfix_within_window=False,
            incident_linked=False, mttr_minutes=45,
        )

    written = relabel_pending(log, fetcher, now=now_fn)
    assert written == 1
    pairs = list(log.iter_labeled())
    assert pairs[0].outcome.label == OutcomeLabel.REVERTED
    assert pairs[0].outcome.mttr_minutes == 45


def test_relabel_pending_skips_already_labeled(tmp_path: Path):
    """Idempotent — running the job twice doesn't duplicate outcomes."""
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.record_prediction(
        Prediction(pr_id="PR-1", risk_score=0.7, tier_reached=2, predicted_at=_at(0))
    )

    def fetcher(_pr_id: str, _: datetime) -> FetchedSignals:
        return FetchedSignals(
            was_reverted=False, hotfix_within_window=False, incident_linked=False
        )

    now_fn = lambda: _at(60 * 24 * 14)  # noqa: E731
    assert relabel_pending(log, fetcher, now=now_fn) == 1
    assert relabel_pending(log, fetcher, now=now_fn) == 0  # second run = no-op


def test_relabel_pending_skips_inside_observation_window(tmp_path: Path):
    """A prediction younger than the window is not labeled — wait for more
    observation time first."""
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.record_prediction(
        Prediction(pr_id="PR-1", risk_score=0.7, tier_reached=2, predicted_at=_at(0))
    )
    # Only 1 day elapsed; default window is 7 days.
    now_fn = lambda: _at(60 * 24 * 1)  # noqa: E731
    fetcher_calls: list[str] = []

    def fetcher(pr_id: str, _: datetime) -> FetchedSignals | None:
        fetcher_calls.append(pr_id)
        return FetchedSignals(False, False, False)

    assert relabel_pending(log, fetcher, now=now_fn) == 0
    assert fetcher_calls == []  # fetcher must not be called for in-window PRs


def test_relabel_pending_skips_when_fetcher_returns_none(tmp_path: Path):
    """Signals not yet available (e.g., upstream API didn't return)
    leaves the prediction unlabeled — next nightly run retries."""
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.record_prediction(
        Prediction(pr_id="PR-1", risk_score=0.7, tier_reached=2, predicted_at=_at(0))
    )
    now_fn = lambda: _at(60 * 24 * 14)  # noqa: E731
    assert relabel_pending(log, lambda *_: None, now=now_fn) == 0
    assert list(log.iter_labeled()) == []
