"""Pure-function mapping from observed post-merge signals to an `Outcome`,
plus the orchestrator that nightly jobs call to relabel pending predictions.

`derive_outcome` is kept pure (no I/O, no clock) so adopters can wire it to
any signal source (GitHub API, internal CI, ICM incidents, Jira) without
touching the labeling rules. The priority hierarchy is unit-testable in
isolation, and the same labels can be replayed deterministically on
historical data when rules are tuned.

Priority hierarchy (highest first):

    INCIDENT  > REVERTED > HOTFIXED > CLEAN

A PR can fire multiple signals (revert + incident, hotfix + revert, etc.).
The most-severe label is the one persisted — silently relabeling an
incident as 'hotfixed' would understate the agent's miss in both CFR and
MTTR.

`relabel_pending` is the nightly entry point. Adopters provide a
`SignalFetcher` callable that converts a `pr_id` into the boolean signals
this module needs; the storage and clock are injected for testability.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.storage.feedback_log import FeedbackLog, Outcome, OutcomeLabel


def derive_outcome(
    pr_id: str,
    observed_at: datetime,
    *,
    was_reverted: bool,
    hotfix_within_window: bool,
    incident_linked: bool,
    mttr_minutes: int | None = None,
) -> Outcome:
    """Build an `Outcome` from the signals observed during the labeling window.

    Args:
        pr_id: PR identifier (must match the prediction's pr_id for join).
        observed_at: When this labeling decision was made (typically "now" in
            the nightly job, frozen in tests for determinism).
        was_reverted: True if a revert commit was identified for this PR.
        hotfix_within_window: True if a follow-up fix landed within the
            observation window referencing this PR/commit.
        incident_linked: True if a tracked production incident references
            this PR/commit.
        mttr_minutes: Time-to-recovery in minutes. Ignored for CLEAN
            outcomes; required for the dashboard's MTTR aggregation on
            failure outcomes.
    """
    if incident_linked:
        label = OutcomeLabel.INCIDENT
    elif was_reverted:
        label = OutcomeLabel.REVERTED
    elif hotfix_within_window:
        label = OutcomeLabel.HOTFIXED
    else:
        label = OutcomeLabel.CLEAN

    persisted_mttr = mttr_minutes if label != OutcomeLabel.CLEAN else None
    return Outcome(
        pr_id=pr_id,
        label=label,
        observed_at=observed_at,
        mttr_minutes=persisted_mttr,
    )


# ---------------------------------------------------------------------------
# Nightly relabel orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchedSignals:
    """The signal bundle a `SignalFetcher` must produce.

    Adopters' fetchers translate their source-of-truth (GitHub API,
    internal CI, ICM, Jira, etc.) into these booleans. Keeping the bundle
    flat means the same `relabel_pending` orchestrator works regardless of
    where signals come from.
    """

    was_reverted: bool
    hotfix_within_window: bool
    incident_linked: bool
    mttr_minutes: int | None = None


# (pr_id, prediction_made_at) -> signals, or None if signals aren't ready yet.
SignalFetcher = Callable[[str, datetime], FetchedSignals | None]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def relabel_pending(
    log: FeedbackLog,
    fetcher: SignalFetcher,
    *,
    observation_window: timedelta = timedelta(days=7),
    now: Callable[[], datetime] | None = None,
) -> int:
    """Label every prediction in `log` whose observation window has elapsed.

    Idempotent: predictions that already have an outcome in the log are
    skipped. Predictions whose window has not yet elapsed are skipped (the
    next nightly run will pick them up). Returns the count of new outcomes
    written, suitable for cron/log output.

    The CLI entry point that wires this to a real GitHub fetcher is the
    deployment concern of each adopting team — the orchestrator stays
    decoupled from any specific source.
    """
    clock = now or _utc_now
    cutoff = clock() - observation_window
    state = log._load()  # noqa: SLF001 — log is the friend of its labeler

    written = 0
    for pr_id, prediction in state.predictions.items():
        if pr_id in state.outcomes:
            continue
        if prediction.predicted_at > cutoff:
            continue
        signals = fetcher(pr_id, prediction.predicted_at)
        if signals is None:
            continue
        outcome = derive_outcome(
            pr_id=pr_id,
            observed_at=clock(),
            was_reverted=signals.was_reverted,
            hotfix_within_window=signals.hotfix_within_window,
            incident_linked=signals.incident_linked,
            mttr_minutes=signals.mttr_minutes,
        )
        log.record_outcome(outcome)
        written += 1
    return written
