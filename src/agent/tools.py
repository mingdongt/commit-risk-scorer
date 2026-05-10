"""MCP-style tools surfaced to the agent harness.

Each tool is a Python callable with a typed signature; the agent harness exposes
them as MCP tools when wired through the Claude Agent SDK (see harness.py).

Status (v0.1):
    - git_diff_stats: REAL — pure-Python parse of unified-diff format.
    - lint_check: STUB.
    - test_coverage_lookup: STUB.
    - historical_failure_search: STUB (RAG retrieval lands once Elasticsearch index ships).
"""
from __future__ import annotations

from typing import Any


def git_diff_stats(diff: str) -> dict[str, int | list[str]]:
    """Compute basic statistics on a unified git diff.

    Args:
        diff: text of a unified diff (output of `git diff` / GitHub PR diff API).

    Returns:
        Dict with `files_touched`, `additions`, `deletions`, `files`.
    """
    files: set[str] = set()
    additions = 0
    deletions = 0

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
        elif line.startswith("--- a/"):
            files.add(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {
        "files_touched": len(files),
        "additions": additions,
        "deletions": deletions,
        "files": sorted(files),
    }


def lint_check(diff: str, language: str | None = None) -> dict[str, Any]:
    """Run static analysis on the changed code.

    Status: STUB. v0.2 dispatches to ESLint / Ruff / Pylint / Clang-Tidy based on
    `language` and aggregates findings (errors/warnings counts + first-N lines).
    """
    raise NotImplementedError(
        "lint_check stub — v0.2 will dispatch to language-specific linters."
    )


def test_coverage_lookup(files: list[str]) -> dict[str, Any]:
    """Look up which tests exercise the given source files.

    Status: STUB. v0.2 reads `coverage.py` JSON output (or Bazel test-impact graph
    for monorepos) and returns the set of test names that load any modified file.
    """
    raise NotImplementedError(
        "test_coverage_lookup stub — v0.2 will read coverage.py / Bazel impact graphs."
    )


def historical_failure_search(diff: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the top-K historical PRs whose diffs are most similar to the input.

    Status: STUB. v0.2 wires to the Elasticsearch index built from scrape_github_prs.py.
    Each result includes the PR number, similarity score, and the CI outcome label.
    """
    raise NotImplementedError(
        "historical_failure_search stub — v0.2 wires to the Elasticsearch RAG index."
    )


# Tool registry — what the harness exposes to the LLM judge as MCP tools.
TOOL_REGISTRY: dict[str, Any] = {
    "git_diff_stats": git_diff_stats,
    "lint_check": lint_check,
    "test_coverage_lookup": test_coverage_lookup,
    "historical_failure_search": historical_failure_search,
}
