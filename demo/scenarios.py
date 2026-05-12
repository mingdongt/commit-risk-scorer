"""Three hand-crafted PR scenarios for the demo walkthrough.

Each scenario is the kind of PR a real reviewer sees on a Tuesday — picked so
the agent's output is *different* across them in ways a human reviewer would
agree with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Scenario:
    name: str
    narrative: str  # one-sentence framing for the demo output
    diff: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Scenario A — Low risk: README typo
# ---------------------------------------------------------------------------

SCENARIO_A = Scenario(
    name="A — README typo fix",
    narrative=(
        "Single-line README change by a regular contributor. Should fast-track "
        "through with no policy gate."
    ),
    diff=(
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -3,1 +3,1 @@\n"
        "-A shift-left enginering intellgence agent.\n"
        "+A shift-left engineering intelligence agent.\n"
    ),
    metadata={
        "author": {"login": "regular-contributor"},
        # Empty-string prefix acts as a catch-all: any path startswith("")
        # is True, but longest-prefix matching still prefers more specific
        # entries when present. Mirrors GitHub CODEOWNERS' `*` semantics.
        "codeowners": {
            "docs/": ["@docs-team"],
            "": ["@platform-team"],
        },
    },
)


# ---------------------------------------------------------------------------
# Scenario B — Medium risk: auth refactor with codeowners
# ---------------------------------------------------------------------------

# Modest-sized auth refactor — touches 4 files in src/auth/. Owner is clear
# (@security-team), so ownership-mapper has high signal, but the change is in
# a high-incident area.
_AUTH_REFACTOR_FILES: list[str] = [
    "src/auth/session.py",
    "src/auth/token_validator.py",
    "src/auth/refresh.py",
    "src/auth/__init__.py",
]


def _build_auth_diff() -> str:
    parts: list[str] = []
    for f in _AUTH_REFACTOR_FILES:
        parts.append(f"--- a/{f}\n+++ b/{f}\n@@ -10,5 +10,5 @@\n")
        parts.extend(f"-old_{f}_{i}\n+new_{f}_{i}\n" for i in range(5))
    return "".join(parts)


SCENARIO_B = Scenario(
    name="B — auth-module refactor with code owners",
    narrative=(
        "Four-file refactor inside src/auth/. Codeowners map provided. "
        "Should escalate to owner / SME review based on ownership signal "
        "and module sensitivity."
    ),
    diff=_build_auth_diff(),
    metadata={
        "author": {"login": "regular-contributor"},
        "codeowners": {
            "src/auth/": ["@security-team", "@auth-owner"],
            "src/": ["@platform-team"],
        },
        "commit_messages": [
            "Refactor session lifecycle to use shared token cache",
            "Update validator to accept new claim format from refresh path",
            "Wire __init__ exports through the new path",
        ],
    },
)


# ---------------------------------------------------------------------------
# Scenario C — Critical risk: bot-authored mechanical refactor with scope drift
# ---------------------------------------------------------------------------


def _build_bot_diff() -> str:
    """8 files × ~100 lines/file = ~800-line mechanical refactor."""
    parts: list[str] = []
    files = [f"src/module_{i}.py" for i in range(8)]
    for f in files:
        parts.append(f"--- a/{f}\n+++ b/{f}\n@@ -1,100 +1,100 @@\n")
        parts.extend(f"-old_{f}_{j}\n+new_{f}_{j}\n" for j in range(50))
    return "".join(parts)


SCENARIO_C = Scenario(
    name="C — bot-authored mechanical refactor with scope drift",
    narrative=(
        "Agent-authored PR rewriting eight modules in one go. No CODEOWNERS "
        "passed for these paths. PR description claims to touch a critical "
        "config file that isn't actually in the diff (prompt-vs-shipped-diff "
        "drift). Should land in the Critical / block-merge band."
    ),
    diff=_build_bot_diff(),
    metadata={
        "author": {"login": "copilot-coding[bot]"},
        "commit_messages": ["wip", "wip 2", "wip 3", "auto-fix lint", "ok"],
        "pr_description_paths": [
            "src/module_0.py",
            "src/module_1.py",
            "config/critical_runtime.yaml",  # not in the diff → scope drift
            "src/database/migrations/0042_breaking.py",  # also not in the diff
        ],
        "codeowners": {"docs/": ["@docs-team"]},  # doesn't cover src/module_*
    },
)


ALL_SCENARIOS: list[Scenario] = [SCENARIO_A, SCENARIO_B, SCENARIO_C]
