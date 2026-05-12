"""Tests for the GitHub PR scraper.

The HTTP layer is mocked — these tests pin the label-derivation logic, the
schema of the JSONL output, and the rate-limit response behavior. End-to-end
hits against the real GitHub API are deferred to manual / scheduled jobs
(would require a token in CI and a stable test fixture, both out of scope for
v0.1).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.scrape_github_prs import (
    MAX_DIFF_CHARS,
    ScrapedPR,
    derive_ci_outcome,
    iter_closed_prs,
    scrape,
    scrape_one_pr,
)


# ---------------------------------------------------------------------------
# derive_ci_outcome
# ---------------------------------------------------------------------------


def test_derive_outcome_passed():
    checks = [
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "success"},
    ]
    assert derive_ci_outcome(checks) == "passed"


def test_derive_outcome_failed():
    checks = [
        {"status": "completed", "conclusion": "failure"},
        {"status": "completed", "conclusion": "failure"},
    ]
    assert derive_ci_outcome(checks) == "failed"


def test_derive_outcome_mixed():
    checks = [
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "failure"},
    ]
    assert derive_ci_outcome(checks) == "mixed"


def test_derive_outcome_unknown_no_checks():
    assert derive_ci_outcome([]) == "unknown"


def test_derive_outcome_unknown_all_in_progress():
    """No completed checks — outcome is unknown, never falsely 'passed'."""
    checks = [
        {"status": "in_progress", "conclusion": None},
        {"status": "queued", "conclusion": None},
    ]
    assert derive_ci_outcome(checks) == "unknown"


def test_derive_outcome_neutral_and_skipped_are_good():
    """Workflow author intentionally signaling 'no opinion' counts as a pass."""
    checks = [
        {"status": "completed", "conclusion": "neutral"},
        {"status": "completed", "conclusion": "skipped"},
    ]
    assert derive_ci_outcome(checks) == "passed"


def test_derive_outcome_timed_out_counts_as_failed():
    checks = [{"status": "completed", "conclusion": "timed_out"}]
    assert derive_ci_outcome(checks) == "failed"


# ---------------------------------------------------------------------------
# iter_closed_prs — pagination
# ---------------------------------------------------------------------------


def _mock_session_paginated(pages: list[list[dict]]):
    """Build a session whose .get(...).json() returns successive `pages`."""
    session = MagicMock()
    responses = []
    for page in pages:
        r = MagicMock()
        r.json.return_value = page
        r.headers = {}  # no rate-limit headers to inspect
        r.raise_for_status.return_value = None
        responses.append(r)
    session.get.side_effect = responses
    return session


def test_iter_closed_prs_stops_at_max():
    """Even if the API has more PRs, we stop at max_prs."""
    page1 = [{"number": i} for i in range(100)]
    session = _mock_session_paginated([page1])
    prs = list(iter_closed_prs(session, "owner/repo", max_prs=10))
    assert len(prs) == 10
    assert prs[0]["number"] == 0


def test_iter_closed_prs_walks_pages():
    """Paginates until max_prs is hit, even across pages."""
    page1 = [{"number": i} for i in range(100)]
    page2 = [{"number": 100 + i} for i in range(50)]
    session = _mock_session_paginated([page1, page2])
    prs = list(iter_closed_prs(session, "owner/repo", max_prs=130))
    assert len(prs) == 130
    assert prs[129]["number"] == 129


def test_iter_closed_prs_handles_empty_page():
    """An empty page ends the iteration even if max_prs not reached."""
    session = _mock_session_paginated([[]])
    prs = list(iter_closed_prs(session, "owner/repo", max_prs=10))
    assert prs == []


# ---------------------------------------------------------------------------
# scrape_one_pr — full PR fetch, mocked
# ---------------------------------------------------------------------------


def test_scrape_one_pr_builds_full_record(monkeypatch):
    """Mock _get_json + _get_diff and verify schema of the ScrapedPR."""
    from src.data import scrape_github_prs as mod

    def fake_get_json(_session, url, params=None):
        if "/check-runs" in url:
            return {"check_runs": [{"status": "completed", "conclusion": "failure"}]}
        # PR details endpoint
        return {"changed_files": 7, "additions": 142, "deletions": 23}

    def fake_get_diff(_session, _url):
        return "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"

    monkeypatch.setattr(mod, "_get_json", fake_get_json)
    monkeypatch.setattr(mod, "_get_diff", fake_get_diff)

    pr_summary = {
        "number": 42,
        "title": "Fix the thing",
        "head": {"sha": "deadbeef"},
        "base": {"sha": "cafef00d"},
        "merged_at": "2026-05-01T12:00:00Z",
    }

    scraped = scrape_one_pr(session=MagicMock(), repo="owner/repo", pr_summary=pr_summary)
    assert isinstance(scraped, ScrapedPR)
    assert scraped.pr_id == "owner/repo#42"
    assert scraped.merged is True
    assert scraped.ci_outcome == "failed"
    assert scraped.files_changed_count == 7
    assert scraped.additions == 142
    assert scraped.deletions == 23
    assert "old" in scraped.diff
    assert len(scraped.diff) <= MAX_DIFF_CHARS


def test_scrape_one_pr_returns_none_on_http_error(monkeypatch):
    """A failed fetch is logged and skipped, not crashed on."""
    import requests as _r

    from src.data import scrape_github_prs as mod

    def boom(*_a, **_kw):
        response = MagicMock()
        response.status_code = 404
        raise _r.HTTPError(response=response)

    monkeypatch.setattr(mod, "_get_json", boom)

    pr_summary = {
        "number": 1,
        "title": "?",
        "head": {"sha": "x"},
        "base": {"sha": "y"},
        "merged_at": None,
    }
    assert scrape_one_pr(session=MagicMock(), repo="owner/repo", pr_summary=pr_summary) is None


# ---------------------------------------------------------------------------
# scrape — end-to-end with mocked HTTP, writes JSONL
# ---------------------------------------------------------------------------


def test_scrape_writes_jsonl(monkeypatch, tmp_path):
    """Full scrape flow: mock pagination + per-PR fetch, verify JSONL output."""
    from src.data import scrape_github_prs as mod

    def fake_iter(_session, _repo, _max):
        for i in range(3):
            yield {
                "number": i,
                "title": f"PR {i}",
                "head": {"sha": f"sha{i}"},
                "base": {"sha": "base"},
                "merged_at": "2026-05-01" if i % 2 == 0 else None,
            }

    def fake_get_json(_session, url, params=None):
        if "/check-runs" in url:
            return {"check_runs": [{"status": "completed", "conclusion": "success"}]}
        return {"changed_files": 1, "additions": 1, "deletions": 1}

    def fake_get_diff(_session, _url):
        return "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"

    monkeypatch.setattr(mod, "iter_closed_prs", fake_iter)
    monkeypatch.setattr(mod, "_get_json", fake_get_json)
    monkeypatch.setattr(mod, "_get_diff", fake_get_diff)

    out = tmp_path / "prs.jsonl"
    n = scrape(repos=["owner/repo"], max_prs_per_repo=3, output_path=str(out), token=None)
    assert n == 3

    lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert all(r["ci_outcome"] == "passed" for r in records)
    assert records[0]["merged"] is True
    assert records[1]["merged"] is False
    assert all("diff" in r for r in records)
