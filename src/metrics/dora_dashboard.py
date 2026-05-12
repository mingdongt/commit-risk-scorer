"""DORA-style impact dashboard for commit-risk-scorer.

Mirrors the metrics defined in `docs/metrics.md` (cycle time, change failure
rate, MTTR, adoption) and renders them as a Streamlit app. This is the
visible-surface counterpart to the audit-log writes documented in
`docs/enterprise-safety.md` Control 6 — adopters point the dashboard at the
audit-store and watch the impact of the agent over time.

Run locally:
    streamlit run src/metrics/dora_dashboard.py

Status (v0.1):
    The data layer is *simulated* (deterministic + reproducible via a fixed
    seed). The interface contract — `load_metrics()` returning a DataFrame
    with the columns the dashboard renders — stays the same in v0.2 when it
    is rewired to a real audit-store query.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta  # noqa: F401  (timedelta reserved for v0.2)

import numpy as np
import pandas as pd

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data layer — v0.1 simulated; v0.2 reads from AuditStore.query_*().
# ---------------------------------------------------------------------------


@dataclass
class DashboardData:
    """Aggregated metrics for the dashboard."""

    daily: pd.DataFrame  # rows: per-day metrics for the trend charts
    by_team: pd.DataFrame  # rows: per-team metrics for the breakdown table
    headline: dict[str, float]  # latest week summary for the top metric cards


# Default cohort used by the v0.1 simulated data. Adopters override this with
# whatever teams their audit-store has data for.
DEFAULT_TEAMS = ("auth", "platform", "data", "frontend", "infra")


def load_metrics(
    days: int = 90,
    teams: tuple[str, ...] = DEFAULT_TEAMS,
    seed: int = 42,
) -> DashboardData:
    """Build the dashboard's data payload.

    v0.1 returns deterministic simulated data. v0.2 swaps the body for an
    audit-store query (see `src/storage/audit_store.py`) — the return shape is
    stable.
    """
    rng = np.random.default_rng(seed)
    end = datetime.now(UTC).date()
    dates = pd.date_range(end=end, periods=days, freq="D")

    # Simulate a gentle adoption ramp (logistic) over the window, plus per-day
    # noise. The trend is intentionally positive — that's the story a real
    # adopter expects to see when the agent is helping.
    t = np.linspace(-6, 6, days)
    adoption = 1 / (1 + np.exp(-t))  # 0 -> 1 logistic
    cycle_time = 36 - 14 * adoption + rng.normal(0, 1.5, size=days)
    cfr = 0.18 - 0.08 * adoption + rng.normal(0, 0.015, size=days)
    mttr = 4.5 - 0.7 * adoption + rng.normal(0, 0.3, size=days)
    adoption_pct = (adoption * 0.85 + 0.05) + rng.normal(0, 0.02, size=days)

    daily = pd.DataFrame(
        {
            "date": dates,
            "cycle_time_hours": cycle_time.clip(min=4),
            "change_failure_rate": cfr.clip(min=0.01, max=0.5),
            "mttr_hours": mttr.clip(min=0.5),
            "adoption_rate": adoption_pct.clip(min=0.0, max=1.0),
        }
    )

    # Per-team variance: each team gets a different multiplicative offset.
    team_rows = []
    for team in teams:
        offset = rng.uniform(0.85, 1.15)
        latest = daily.iloc[-1]
        team_rows.append(
            {
                "team": team,
                "cycle_time_hours": round(float(latest["cycle_time_hours"]) * offset, 1),
                "change_failure_rate": round(float(latest["change_failure_rate"]) * offset, 3),
                "mttr_hours": round(float(latest["mttr_hours"]) * offset, 2),
                "adoption_rate": round(float(latest["adoption_rate"]) * offset, 2),
                "prs_scored_7d": int(rng.integers(40, 220)),
                "fp_feedback_rate": round(float(rng.uniform(0.02, 0.12)), 3),
            }
        )
    by_team = pd.DataFrame(team_rows)

    # Headline: latest day vs. first day, signed delta.
    first, last = daily.iloc[0], daily.iloc[-1]
    headline = {
        "cycle_time_hours_now": round(float(last["cycle_time_hours"]), 1),
        "cycle_time_hours_delta": round(
            float(last["cycle_time_hours"] - first["cycle_time_hours"]), 1
        ),
        "change_failure_rate_now": round(float(last["change_failure_rate"]), 3),
        "change_failure_rate_delta": round(
            float(last["change_failure_rate"] - first["change_failure_rate"]), 3
        ),
        "mttr_hours_now": round(float(last["mttr_hours"]), 2),
        "mttr_hours_delta": round(float(last["mttr_hours"] - first["mttr_hours"]), 2),
        "adoption_rate_now": round(float(last["adoption_rate"]), 2),
        "adoption_rate_delta": round(
            float(last["adoption_rate"] - first["adoption_rate"]), 2
        ),
    }

    return DashboardData(daily=daily, by_team=by_team, headline=headline)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _require_streamlit() -> None:
    if not _STREAMLIT_AVAILABLE:
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install streamlit`, "
            "then run `streamlit run src/metrics/dora_dashboard.py`."
        )


def render() -> None:
    """Render the dashboard. Called via `streamlit run`."""
    _require_streamlit()

    st.set_page_config(page_title="commit-risk-scorer · DORA Dashboard", layout="wide")
    st.title("📊 commit-risk-scorer — DORA Impact Dashboard")
    st.caption(
        "Engineering-outcome metrics for the agent rollout. "
        "v0.1 data is simulated; v0.2 reads from the audit-store "
        "(`src/storage/audit_store.py`)."
    )

    with st.sidebar:
        st.header("Filters")
        days = st.slider("Window (days)", min_value=14, max_value=180, value=90, step=7)
        teams_all = list(DEFAULT_TEAMS)
        teams = st.multiselect("Teams", teams_all, default=teams_all)
        if st.button("Refresh"):
            st.cache_data.clear()
        st.caption("Data is deterministic given a fixed seed; refresh re-runs with the same seed.")

    data = load_metrics(days=days, teams=tuple(teams) if teams else DEFAULT_TEAMS)
    h = data.headline

    # Headline metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Cycle time (hrs)",
        f"{h['cycle_time_hours_now']:.1f}",
        delta=f"{h['cycle_time_hours_delta']:+.1f}",
        delta_color="inverse",  # lower is better
    )
    c2.metric(
        "Change failure rate",
        f"{h['change_failure_rate_now']:.1%}",
        delta=f"{h['change_failure_rate_delta']:+.1%}",
        delta_color="inverse",
    )
    c3.metric(
        "MTTR (hrs)",
        f"{h['mttr_hours_now']:.2f}",
        delta=f"{h['mttr_hours_delta']:+.2f}",
        delta_color="inverse",
    )
    c4.metric(
        "Adoption rate",
        f"{h['adoption_rate_now']:.0%}",
        delta=f"{h['adoption_rate_delta']:+.0%}",
        delta_color="normal",  # higher is better
    )

    st.divider()

    # Trend charts
    daily = data.daily.set_index("date")
    st.subheader("Trends")
    t1, t2 = st.columns(2)
    with t1:
        st.caption("Cycle time (hours, lower is better)")
        st.line_chart(daily["cycle_time_hours"])
    with t2:
        st.caption("Change failure rate (lower is better)")
        st.line_chart(daily["change_failure_rate"])

    t3, t4 = st.columns(2)
    with t3:
        st.caption("MTTR (hours, lower is better)")
        st.line_chart(daily["mttr_hours"])
    with t4:
        st.caption("Adoption rate (higher is better)")
        st.line_chart(daily["adoption_rate"])

    st.divider()

    # Per-team breakdown
    st.subheader("Per-team breakdown (latest 7-day rollup)")
    st.dataframe(data.by_team, use_container_width=True, hide_index=True)
    st.caption(
        "`fp_feedback_rate` is the share of agent-flagged PRs whose author "
        "submitted a 'this was wrong' signal. > 15% triggers a threshold "
        "re-tune per `docs/metrics.md`."
    )


if _STREAMLIT_AVAILABLE and __name__ != "__main__":
    # Streamlit's runner imports the module, then expects `render()` to have
    # been called. We don't auto-render on import to keep the module
    # importable from tests (the test imports load_metrics without spinning
    # up a Streamlit session).
    pass


if __name__ == "__main__":
    render()
