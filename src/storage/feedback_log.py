"""Append-only feedback log — closes the loop from prediction to observed outcome.

Why this exists
---------------
The agent's predictions are only as good as the data the next training run
sees. A risk-scoring system without a feedback channel cannot improve over
time — it ossifies around its initial training distribution and silently
miscalibrates as the codebase and team evolve. The `FeedbackLog` is the data
substrate for the JD bullet:

    "Continuously measure and report on the impact of AI interventions,
     showing progress in metrics such as cycle time, change failure rate,
     and mean time to recovery (MTTR)."

It records two event types:

    - Prediction: written at score time (pr_id, risk_score, tier_reached, ts).
    - Outcome:    written after an observation window (typically 7 days),
                  labeled CLEAN | REVERTED | HOTFIXED | INCIDENT, plus the
                  observed MTTR in minutes for non-clean outcomes.

Joined by `pr_id`, the pairs are both (a) the training set for the next
classifier refresh and (b) the DORA dashboard's source of truth.

Storage choice
--------------
JSONL on local disk. Deliberately minimal: every adopting org will want to
plug this into their existing storage (Mongo / MySQL / ES — see
`audit_store.py`). The JSONL implementation here is the *reference* that
keeps the OSS project runnable without a database dependency and pins the
interface that the production-storage adapters must satisfy.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import median


class OutcomeLabel(str, Enum):
    """The post-merge label assigned to a scored PR.

    CLEAN     — merged and stayed merged through the observation window.
    REVERTED  — the PR was reverted (clear failure signal).
    HOTFIXED  — a follow-up fix landed within the observation window
                referencing this PR / commit (looser failure signal).
    INCIDENT  — the change caused a tracked production incident (strongest
                failure signal).

    Subclassing `str` makes the value JSON-serializable directly and
    round-trippable through `OutcomeLabel(value)`.
    """

    CLEAN = "clean"
    REVERTED = "reverted"
    HOTFIXED = "hotfixed"
    INCIDENT = "incident"


_FAILURE_LABELS = frozenset(
    {OutcomeLabel.REVERTED, OutcomeLabel.HOTFIXED, OutcomeLabel.INCIDENT}
)


@dataclass(frozen=True)
class Prediction:
    pr_id: str
    risk_score: float
    tier_reached: int
    predicted_at: datetime

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                f"risk_score={self.risk_score} must be within [0.0, 1.0]"
            )
        if self.tier_reached not in (1, 2, 3):
            raise ValueError(
                f"tier_reached={self.tier_reached} must be 1, 2, or 3 "
                "(matches TieredRouter tiers)"
            )


@dataclass(frozen=True)
class Outcome:
    pr_id: str
    label: OutcomeLabel
    observed_at: datetime
    mttr_minutes: int | None = None


@dataclass(frozen=True)
class LabeledPair:
    """A prediction joined with its observed outcome — one training row."""

    prediction: Prediction
    outcome: Outcome


# ---------------------------------------------------------------------------
# JSONL (de)serialization
# ---------------------------------------------------------------------------


_PREDICTION_TYPE = "prediction"
_OUTCOME_TYPE = "outcome"


def _prediction_to_jsonl(p: Prediction) -> str:
    return json.dumps(
        {
            "type": _PREDICTION_TYPE,
            "pr_id": p.pr_id,
            "risk_score": p.risk_score,
            "tier_reached": p.tier_reached,
            "predicted_at": p.predicted_at.isoformat(),
        }
    )


def _outcome_to_jsonl(o: Outcome) -> str:
    return json.dumps(
        {
            "type": _OUTCOME_TYPE,
            "pr_id": o.pr_id,
            "label": o.label.value,
            "mttr_minutes": o.mttr_minutes,
            "observed_at": o.observed_at.isoformat(),
        }
    )


def _parse_prediction(d: dict) -> Prediction:
    return Prediction(
        pr_id=d["pr_id"],
        risk_score=d["risk_score"],
        tier_reached=d["tier_reached"],
        predicted_at=datetime.fromisoformat(d["predicted_at"]),
    )


def _parse_outcome(d: dict) -> Outcome:
    return Outcome(
        pr_id=d["pr_id"],
        label=OutcomeLabel(d["label"]),
        mttr_minutes=d["mttr_minutes"],
        observed_at=datetime.fromisoformat(d["observed_at"]),
    )


# ---------------------------------------------------------------------------
# FeedbackLog
# ---------------------------------------------------------------------------


@dataclass
class _LoadedState:
    """In-memory projection of the JSONL file at read time.

    Predictions are deduped on `pr_id` keeping the most recent
    `predicted_at` — see `test_repeat_predictions_resolve_to_latest`.
    Outcomes overwrite older outcomes for the same `pr_id` (the labeler may
    upgrade CLEAN → INCIDENT if a later incident is linked).
    """

    predictions: dict[str, Prediction] = field(default_factory=dict)
    outcomes: dict[str, Outcome] = field(default_factory=dict)


class FeedbackLog:
    """Append-only prediction/outcome log with a joined `iter_labeled` view."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writes ------------------------------------------------------------

    def record_prediction(self, prediction: Prediction) -> None:
        self._append(_prediction_to_jsonl(prediction))

    def record_outcome(self, outcome: Outcome) -> None:
        self._append(_outcome_to_jsonl(outcome))

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- reads -------------------------------------------------------------

    def _load(self) -> _LoadedState:
        state = _LoadedState()
        if not self.path.exists():
            return state
        with self.path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d["type"] == _PREDICTION_TYPE:
                    p = _parse_prediction(d)
                    existing = state.predictions.get(p.pr_id)
                    if existing is None or p.predicted_at > existing.predicted_at:
                        state.predictions[p.pr_id] = p
                elif d["type"] == _OUTCOME_TYPE:
                    o = _parse_outcome(d)
                    existing_o = state.outcomes.get(o.pr_id)
                    if existing_o is None or o.observed_at > existing_o.observed_at:
                        state.outcomes[o.pr_id] = o
        return state

    def iter_labeled(
        self, since: datetime | None = None
    ) -> Iterator[LabeledPair]:
        """Yield prediction/outcome pairs for PRs that have both.

        Predictions without outcomes (not yet observed) and outcomes without
        predictions (we didn't score this PR) are silently skipped — the
        training pipeline only learns from joined rows.

        `since` filters by `prediction.predicted_at >= since` — i.e., "show
        me predictions made on or after this timestamp."
        """
        state = self._load()
        for pr_id, prediction in state.predictions.items():
            outcome = state.outcomes.get(pr_id)
            if outcome is None:
                continue
            if since is not None and prediction.predicted_at < since:
                continue
            yield LabeledPair(prediction=prediction, outcome=outcome)

    # -- DORA metrics ------------------------------------------------------

    def change_failure_rate(self, since: datetime | None = None) -> float | None:
        """Fraction of labeled predictions whose outcome was a failure.

        Returns None when no labeled data exists (avoid division-by-zero;
        the dashboard renders "—" rather than 0).
        """
        pairs = list(self.iter_labeled(since=since))
        if not pairs:
            return None
        failures = sum(1 for p in pairs if p.outcome.label in _FAILURE_LABELS)
        return failures / len(pairs)

    def mttr_median_minutes(
        self, since: datetime | None = None
    ) -> float | None:
        """Median MTTR (minutes) across failure outcomes that recorded one.

        Median (not mean) deliberately: one runaway incident shouldn't
        dominate the dashboard. The JD's literal wording is 'mean time to
        recovery' but the metric reported across the industry is typically
        the median; we expose this one and leave a true mean to a future
        method if any adopter needs it.
        """
        values = [
            p.outcome.mttr_minutes
            for p in self.iter_labeled(since=since)
            if p.outcome.label in _FAILURE_LABELS and p.outcome.mttr_minutes is not None
        ]
        if not values:
            return None
        return float(median(values))
