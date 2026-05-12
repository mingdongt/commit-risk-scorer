"""Tests for the DORA dashboard data layer.

The Streamlit UI itself is intentionally not unit-tested — that requires the
streamlit testing harness and brittle widget-state mocks. We do test the
deterministic data layer, which is the v0.2 swap point with the audit-store.
"""
from __future__ import annotations

import pandas as pd

from src.metrics.dora_dashboard import DEFAULT_TEAMS, DashboardData, load_metrics


def test_load_metrics_returns_expected_shape():
    data = load_metrics(days=60)
    assert isinstance(data, DashboardData)
    assert isinstance(data.daily, pd.DataFrame)
    assert isinstance(data.by_team, pd.DataFrame)
    assert isinstance(data.headline, dict)


def test_daily_dataframe_has_required_columns():
    data = load_metrics(days=30)
    assert list(data.daily.columns) == [
        "date",
        "cycle_time_hours",
        "change_failure_rate",
        "mttr_hours",
        "adoption_rate",
    ]
    assert len(data.daily) == 30


def test_by_team_dataframe_has_required_columns():
    data = load_metrics(days=14, teams=("auth", "platform"))
    assert set(data.by_team.columns) == {
        "team",
        "cycle_time_hours",
        "change_failure_rate",
        "mttr_hours",
        "adoption_rate",
        "prs_scored_7d",
        "fp_feedback_rate",
    }
    assert sorted(data.by_team["team"].tolist()) == ["auth", "platform"]


def test_headline_has_all_4_dora_metrics():
    data = load_metrics(days=90)
    required = {
        "cycle_time_hours_now",
        "cycle_time_hours_delta",
        "change_failure_rate_now",
        "change_failure_rate_delta",
        "mttr_hours_now",
        "mttr_hours_delta",
        "adoption_rate_now",
        "adoption_rate_delta",
    }
    assert required.issubset(data.headline.keys())


def test_load_metrics_is_deterministic_for_same_seed():
    """Same seed → identical numbers (required for the reproducibility note
    in `docs/evaluation.md` §Estimation honesty)."""
    a = load_metrics(days=30, seed=42)
    b = load_metrics(days=30, seed=42)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.by_team, b.by_team)
    assert a.headline == b.headline


def test_metrics_in_reasonable_ranges():
    """Sanity bounds — the simulator should produce realistic-looking values."""
    data = load_metrics(days=90)
    daily = data.daily
    assert daily["cycle_time_hours"].between(0, 100).all()
    assert daily["change_failure_rate"].between(0, 1).all()
    assert daily["mttr_hours"].between(0, 50).all()
    assert daily["adoption_rate"].between(0, 1).all()


def test_default_teams_constant():
    """The default cohort is what the rest of the docs refer to."""
    assert len(DEFAULT_TEAMS) == 5
    assert "platform" in DEFAULT_TEAMS
