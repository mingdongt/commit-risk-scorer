"""FastAPI surface for commit-risk-scorer.

Turns the agent harness from an in-process library into a runnable service that
CI integrations (GitHub Action, ADO webhook, Jenkins step) can POST against.

Endpoints:
    GET  /health   — liveness + version probe
    POST /score    — run the agent on a PR diff, return score + reasoning + action

Run locally:
    uvicorn src.serving.api:app --port 8000

The service is intentionally simple in v0.2 — no auth, no rate limiting, no
queueing. v0.3 layers in: Bearer-token auth (per-org), per-PR idempotency keys,
audit-store writes (see `src/storage/audit_store.py`), and OpenTelemetry traces.

LLM judge wiring
----------------
The Claude LLM judge is opt-in via two env vars to keep the service runnable
on a fresh checkout without an API key:

    CRS_ENABLE_JUDGE=1      enable the judge backend
    ANTHROPIC_API_KEY=...   required when CRS_ENABLE_JUDGE=1

When the judge is disabled, /score returns the heuristic risk score and
`judge_reasoning` is null. When enabled, the harness escalates each request
to the judge and uses its calibrated verdict; on transient judge failures
the harness falls back to the heuristic and `judge_reasoning` carries the
fallback message.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import __version__
from src.agent.explainer import ExplanationWriter
from src.agent.harness import AgentHarness
from src.agent.policy import PolicyGatekeeper


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """POST /score input."""

    diff: str = Field(..., description="Unified-diff text for the change being scored.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional context: author, repo, codeowners map "
            "(prefix -> [owners]), pr_id, commit_sha. Used by sub-agents."
        ),
    )
    render_markdown: bool = Field(
        default=True,
        description="Whether to include the rendered PR comment markdown in the response.",
    )


class SubAgentReportSchema(BaseModel):
    name: str
    confidence: float
    observations: dict[str, Any]
    risk_factors: list[str]


class ScoreResponse(BaseModel):
    """POST /score output."""

    risk_score: float
    risk_level: str  # Low | Medium | High | Critical
    action: str  # fast_track | owner_review | sme_review | block_merge
    rationale: str
    top_risk_factors: list[str]
    recommended_steps: list[str]
    sub_agent_reports: list[SubAgentReportSchema]
    pr_comment_markdown: str | None  # None when render_markdown=False
    judge_reasoning: str | None  # Populated when CRS_ENABLE_JUDGE=1; null otherwise.
    judge_enabled: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    judge_enabled: bool


# ---------------------------------------------------------------------------
# Optional Claude judge construction
# ---------------------------------------------------------------------------


def _maybe_build_judge() -> Any | None:
    """Construct ClaudeJudgeBackend only when explicitly enabled.

    Returns None — and logs a single info-level message — when the judge is
    disabled, so the service still starts cleanly on environments without an
    Anthropic API key (laptop demos, CI smoke runs).
    """
    if os.environ.get("CRS_ENABLE_JUDGE") != "1":
        logger.info("Claude judge disabled (CRS_ENABLE_JUDGE != '1').")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "CRS_ENABLE_JUDGE=1 but ANTHROPIC_API_KEY is not set; "
            "judge will not be constructed."
        )
        return None
    # Lazy import — keeps `anthropic` out of the import graph when the judge
    # is disabled, so the service is light on cold start in that mode.
    from src.models.gateway import ClaudeJudgeBackend  # noqa: PLC0415

    logger.info("Claude judge enabled.")
    return ClaudeJudgeBackend()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="commit-risk-scorer",
    description=(
        "Shift-left engineering intelligence agent — predicts PR risk, "
        "recommends reviewer/test/gate actions, closes the loop via DORA metrics."
    ),
    version=__version__,
)


# These are constructed once per process. They're stateless and thread-safe.
_JUDGE = _maybe_build_judge()
_HARNESS = AgentHarness(judge_backend=_JUDGE)
_POLICY = PolicyGatekeeper()
_EXPLAINER = ExplanationWriter()
_JUDGE_ENABLED = _JUDGE is not None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, judge_enabled=_JUDGE_ENABLED)


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not request.diff or not request.diff.strip():
        raise HTTPException(status_code=400, detail="`diff` is required and must be non-empty.")

    result = _HARNESS.score(request.diff, metadata=request.metadata)
    decision = _POLICY.decide(result)
    comment = _EXPLAINER.render(result, decision) if request.render_markdown else None

    return ScoreResponse(
        risk_score=round(result.risk_score, 4),
        risk_level=decision.risk_level,
        action=decision.action,
        rationale=decision.rationale,
        top_risk_factors=[rf for r in result.sub_agent_reports for rf in r.risk_factors],
        recommended_steps=list(decision.recommended_steps),
        sub_agent_reports=[
            SubAgentReportSchema(
                name=r.sub_agent_name,
                confidence=round(r.confidence, 4),
                observations=r.observations,
                risk_factors=list(r.risk_factors),
            )
            for r in result.sub_agent_reports
        ],
        pr_comment_markdown=comment,
        judge_reasoning=result.reasoning,
        judge_enabled=_JUDGE_ENABLED,
    )
