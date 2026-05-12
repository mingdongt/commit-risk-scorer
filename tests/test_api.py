"""Tests for the FastAPI surface."""
from __future__ import annotations

import pytest

# FastAPI's TestClient is the canonical way to test endpoints without binding
# to a port. Skip cleanly when FastAPI isn't installed (the CI workflow ships
# without it on purpose — agent-layer tests are pure stdlib).
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.serving.api import app  # noqa: E402

client = TestClient(app)


SMALL_DIFF = (
    # Non-sensitive path — DiffAnalyzer's SENSITIVE_PATH_PREFIXES does not match,
    # so this exercises the "ordinary small diff → low risk" path cleanly.
    "--- a/src/helpers/format.py\n"
    "+++ b/src/helpers/format.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


LARGE_DIFF = (
    "--- a/src/helpers/format.py\n+++ b/src/helpers/format.py\n"
    "@@ -1,600 +1,600 @@\n"
    + "\n".join(f"-old_{i}\n+new_{i}" for i in range(600))
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_endpoint_ok():
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert "version" in payload


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------


def test_score_small_diff_returns_low_risk():
    r = client.post("/score", json={"diff": SMALL_DIFF})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["risk_level"] == "Low"
    assert payload["action"] == "fast_track"
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert payload["pr_comment_markdown"]  # default render_markdown=True
    assert "Commit risk" in payload["pr_comment_markdown"]


def test_score_with_codeowners_returns_reviewers():
    """When metadata.codeowners is provided, ownership-mapper surfaces reviewers."""
    r = client.post(
        "/score",
        json={
            "diff": SMALL_DIFF,
            "metadata": {
                "codeowners": {
                    "src/helpers/": ["@helpers-team"],
                    "src/": ["@platform-team"],
                }
            },
        },
    )
    assert r.status_code == 200
    payload = r.json()
    reports_by_name = {r["name"]: r for r in payload["sub_agent_reports"]}
    om = reports_by_name["ownership-mapper"]
    # Longest-prefix match: src/helpers/ wins over src/
    assert "@helpers-team" in om["observations"]["recommended_reviewers"]


def test_score_large_diff_flags_fanout():
    """A diff above the LARGE_DIFF_LINES threshold raises a risk factor."""
    r = client.post("/score", json={"diff": LARGE_DIFF})
    assert r.status_code == 200
    payload = r.json()
    assert any("very large diff" in f for f in payload["top_risk_factors"])


def test_score_render_markdown_can_be_disabled():
    r = client.post("/score", json={"diff": SMALL_DIFF, "render_markdown": False})
    assert r.status_code == 200
    assert r.json()["pr_comment_markdown"] is None


def test_score_empty_diff_rejected():
    r = client.post("/score", json={"diff": ""})
    assert r.status_code == 400
    assert "diff" in r.json()["detail"].lower()


def test_score_missing_diff_field_rejected():
    """Schema validation rejects requests without the required `diff` field."""
    r = client.post("/score", json={})
    assert r.status_code == 422  # FastAPI validation error
