"""GitHub PR scraper — produces the self-labeled training dataset.

Pairs each PR diff with its CI outcome label, exactly the input format that
`src/models/finetune/train_nemo.py` consumes. Mentioned in the design-doc
under §Hypothesis as the GitHub-PR side of the training mix (the cleaner
half being CodeXGLUE Devign).

Output schema (JSONL — one PR per line):

    {
      "pr_id":              "kubernetes/kubernetes#123456",
      "repo":               "kubernetes/kubernetes",
      "pr_number":          123456,
      "title":              "...",
      "base_sha":           "...",
      "head_sha":           "...",
      "merged":             true | false,
      "ci_outcome":         "passed" | "failed" | "mixed" | "unknown",
      "files_changed_count": int,
      "additions":          int,
      "deletions":          int,
      "diff":               "..."   # truncated to MAX_DIFF_CHARS
    }

Auth & rate limits:
    Reads `$GITHUB_TOKEN` (override with --token-env). Without a token, GitHub
    permits 60 unauthenticated requests/hour — enough to smoke-test but not to
    build a useful dataset. With a fine-grained PAT (public-repo read is
    sufficient) the limit is 5000/hour. Each PR consumes ~3 requests.
    The scraper inspects `X-RateLimit-Remaining` / `X-RateLimit-Reset` after
    every call and sleeps near exhaustion.

Usage:
    export GITHUB_TOKEN=ghp_...
    python -m src.data.scrape_github_prs \\
        --repos kubernetes/kubernetes django/django \\
        --max-prs-per-repo 100 \\
        --output data/raw/github_prs.jsonl

The output is JSONL written incrementally — interruptions leave a partial
file that can be resumed by trimming the last line and re-running (a
sequence-numbered version of resume support is on the v0.2 backlog).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import requests

GITHUB_API = "https://api.github.com"
MAX_DIFF_CHARS = 50_000  # truncate huge PRs so the JSONL stays manageable


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class ScrapedPR:
    pr_id: str
    repo: str
    pr_number: int
    title: str
    base_sha: str
    head_sha: str
    merged: bool
    ci_outcome: str  # passed | failed | mixed | unknown
    files_changed_count: int
    additions: int
    deletions: int
    diff: str


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_session(token: str | None) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "commit-risk-scorer/0.1 (+https://github.com/mingdongt/commit-risk-scorer)",
        }
    )
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _respect_rate_limit(response: requests.Response, threshold: int = 5) -> None:
    """Sleep until reset if the rate limit is about to bite.

    GitHub returns `X-RateLimit-Remaining` and `X-RateLimit-Reset` (unix
    seconds) on every API response. Below `threshold` requests left, we wait.
    """
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")
    if remaining is None or reset is None:
        return
    try:
        remaining_int = int(remaining)
        reset_int = int(reset)
    except ValueError:
        return
    if remaining_int < threshold:
        wait = max(0, reset_int - int(time.time())) + 2  # +2s of safety margin
        print(
            f"[rate-limit] {remaining_int} requests left; sleeping {wait}s",
            file=sys.stderr,
        )
        time.sleep(wait)


def _get_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    r = session.get(url, params=params, timeout=30)
    _respect_rate_limit(r)
    r.raise_for_status()
    return r.json()


def _get_diff(session: requests.Session, url: str) -> str:
    headers = {"Accept": "application/vnd.github.v3.diff"}
    r = session.get(url, headers=headers, timeout=30)
    _respect_rate_limit(r)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# GitHub API wrappers
# ---------------------------------------------------------------------------


def iter_closed_prs(
    session: requests.Session, repo: str, max_prs: int
) -> Iterator[dict[str, Any]]:
    """Iterate over recent closed PRs (most recent first), up to `max_prs`."""
    url = f"{GITHUB_API}/repos/{repo}/pulls"
    page = 1
    yielded = 0
    while yielded < max_prs:
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": min(100, max_prs - yielded),
            "page": page,
        }
        batch = _get_json(session, url, params=params)
        if not batch:
            return
        for pr in batch:
            yield pr
            yielded += 1
            if yielded >= max_prs:
                return
        page += 1


def fetch_check_runs(session: requests.Session, repo: str, sha: str) -> list[dict[str, Any]]:
    """All check-runs for a given commit SHA (used to derive CI outcome)."""
    url = f"{GITHUB_API}/repos/{repo}/commits/{sha}/check-runs"
    payload = _get_json(session, url)
    return payload.get("check_runs", [])


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def derive_ci_outcome(check_runs: list[dict[str, Any]]) -> str:
    """Aggregate check-run conclusions into a single label.

    - `passed`   — at least one completed check, all completed checks succeeded
    - `failed`   — at least one completed check failed (failure / timed_out / action_required)
    - `mixed`    — both successes and failures among completed checks (unusual; treated as failed for training)
    - `unknown`  — no completed checks (workflow never ran / still queued)
    """
    if not check_runs:
        return "unknown"

    completed = [c for c in check_runs if c.get("status") == "completed"]
    if not completed:
        return "unknown"

    bad = {"failure", "timed_out", "action_required", "cancelled"}
    good = {"success", "neutral", "skipped"}

    has_bad = any(c.get("conclusion") in bad for c in completed)
    has_good = any(c.get("conclusion") in good for c in completed)

    if has_bad and has_good:
        return "mixed"
    if has_bad:
        return "failed"
    if has_good:
        return "passed"
    return "unknown"


# ---------------------------------------------------------------------------
# Scrape orchestration
# ---------------------------------------------------------------------------


def scrape_one_pr(
    session: requests.Session, repo: str, pr_summary: dict[str, Any]
) -> ScrapedPR | None:
    """Build a ScrapedPR for one PR; returns None if the PR can't be fetched."""
    pr_number = pr_summary["number"]
    head_sha = pr_summary["head"]["sha"]
    base_sha = pr_summary["base"]["sha"]
    merged = bool(pr_summary.get("merged_at"))

    try:
        # Full PR object for additions/deletions/changed_files counts
        details = _get_json(
            session, f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        )
        diff_text = _get_diff(
            session, f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        )
        check_runs = fetch_check_runs(session, repo, head_sha)
    except requests.HTTPError as e:
        print(
            f"[skip] {repo}#{pr_number}: HTTP {e.response.status_code}",
            file=sys.stderr,
        )
        return None

    return ScrapedPR(
        pr_id=f"{repo}#{pr_number}",
        repo=repo,
        pr_number=pr_number,
        title=pr_summary["title"],
        base_sha=base_sha,
        head_sha=head_sha,
        merged=merged,
        ci_outcome=derive_ci_outcome(check_runs),
        files_changed_count=int(details.get("changed_files", 0)),
        additions=int(details.get("additions", 0)),
        deletions=int(details.get("deletions", 0)),
        diff=diff_text[:MAX_DIFF_CHARS],
    )


def scrape(repos: list[str], max_prs_per_repo: int, output_path: str, token: str | None) -> int:
    """Main entry — scrape and write JSONL. Returns number of PRs written."""
    session = _make_session(token)
    if not token:
        print(
            "[warn] No GITHUB_TOKEN set — limited to 60 unauthenticated requests/hour.",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    label_counts: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out:
        for repo in repos:
            print(f"[scrape] {repo} (up to {max_prs_per_repo} PRs)", file=sys.stderr)
            for pr_summary in iter_closed_prs(session, repo, max_prs_per_repo):
                scraped = scrape_one_pr(session, repo, pr_summary)
                if scraped is None:
                    continue
                out.write(json.dumps(asdict(scraped)) + "\n")
                out.flush()
                written += 1
                label_counts[scraped.ci_outcome] = label_counts.get(scraped.ci_outcome, 0) + 1

    print(f"[done] wrote {written} PRs to {output_path}", file=sys.stderr)
    print(f"[labels] {label_counts}", file=sys.stderr)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="GitHub repos in owner/name format (one or more).",
    )
    parser.add_argument("--max-prs-per-repo", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/raw/github_prs.jsonl")
    parser.add_argument(
        "--token-env",
        type=str,
        default="GITHUB_TOKEN",
        help="Env var holding a GitHub PAT (read-only on public repos is sufficient).",
    )
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    scrape(args.repos, args.max_prs_per_repo, args.output, token)


if __name__ == "__main__":
    main()
