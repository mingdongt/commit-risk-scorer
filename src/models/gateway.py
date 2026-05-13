"""Multi-vendor model gateway: unifies Claude, NVIDIA NIM, Triton-served NeMo, and
Azure OpenAI behind a single inference API.

v0.2 status:
    - ClaudeJudgeBackend: REAL (Anthropic Claude API, prompt caching, structured
      outputs, adaptive thinking). Used as the LLM-judge half of the hybrid
      pipeline.
    - TritonNemoBackend / NIMBackend / AzureOpenAIBackend: STUB. Provisioning
      pending — see docs/design-doc.md §Architecture and src/models/finetune/
      train_nemo.py for the NeMo path.
    - ModelGateway.predict_hybrid: REAL when at least the judge backend is
      registered. Falls through to the judge alone until the classifier backend
      ships.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic

from src.models.judge_prompt import JUDGE_SYSTEM_PROMPT, JudgeVerdict


class Backend(Enum):
    CLAUDE = "claude"
    NIM = "nvidia_nim"
    TRITON_NEMO = "triton_nemo"
    AZURE_OPENAI = "azure_openai"


@dataclass
class PredictionRequest:
    """Request to score a commit / pull request.

    The LLM-judge backend additionally consumes pre-computed sub-agent reports
    and retrieved RAG documents; these are kept as separate fields rather than
    being stuffed into `pr_metadata` so the judge contract is self-documenting.
    """

    diff: str
    pr_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    # v0.2 additions — optional structured inputs for the LLM judge.
    # Sub-agent report shape: {name, confidence, observations, risk_factors}.
    # RAG document shape: {layer, retriever, content, score}.
    sub_agent_reports: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    rag_documents: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass
class PredictionResponse:
    """Risk-scoring output."""

    risk_score: float  # in [0.0, 1.0]
    backend_used: Backend
    latency_ms: float
    reasoning: str | None = None  # populated by LLM-judge backends, None for pure classifiers
    raw: dict[str, Any] | None = None
    # Tier 1 = cheap predictor only; 2 = + LLM judge; 3 = + agent investigation.
    # Set by TieredRouter; backends called directly leave the default (1).
    tier_reached: int = 1


class ModelBackend(ABC):
    """Interface every backend must implement.

    Implementations should be thread-safe — the gateway may dispatch concurrent
    requests across backends for A/B comparisons.
    """

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse: ...

    @abstractmethod
    def name(self) -> Backend: ...


# ---------------------------------------------------------------------------
# ClaudeJudgeBackend — real Anthropic API integration with prompt caching.
# ---------------------------------------------------------------------------


class ClaudeJudgeBackend(ModelBackend):
    """LLM judge over the Anthropic Claude API.

    Implements the second half of the hybrid pipeline: takes the diff, the
    structured sub-agent reports, and any retrieved RAG documents, and returns
    a calibrated `JudgeVerdict` (risk_score, level, top factors, reasoning,
    mitigations).

    **Prompt caching.** The judge persona and scoring rubric live in
    `judge_prompt.JUDGE_SYSTEM_PROMPT` — a stable, frozen byte string sized
    above the 4096-token Opus 4.7 minimum cacheable prefix. Each request marks
    the system block with `cache_control: ephemeral`, so the prefix is paid
    once per ~5 minutes and served from cache thereafter. Verify cache hits
    via `response.usage.cache_read_input_tokens` (exposed in the returned
    `PredictionResponse.raw['usage']`); a persistent zero across repeated calls
    indicates a silent invalidator (timestamp / UUID / non-deterministic
    serialization somewhere in the prefix).

    **Determinism.** Opus 4.7 removed sampling parameters; calibration is
    controlled via `effort` and the rubric in the system prompt rather than
    `temperature`. Adaptive thinking is enabled — the model decides when and
    how much to think per request — and is configured to emit summarized
    thinking content only when caller code surfaces it (default here is
    omitted, since the structured `reasoning` field is the user-facing output).
    """

    DEFAULT_MODEL = "claude-opus-4-7"
    DEFAULT_EFFORT = "high"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
    ):
        """Construct a judge backend.

        Args:
            client: Pre-configured `anthropic.Anthropic` client. If omitted,
                one is constructed from the `ANTHROPIC_API_KEY` env var.
                Injecting a client is the recommended pattern for tests.
            model: Override the default model ID. Default `claude-opus-4-7`.
            effort: Override the default effort level. One of
                `low | medium | high | xhigh | max`. Default `high`. `xhigh`
                is the recommended setting for agentic coding workloads on
                Opus 4.7; we default to `high` to balance cost against quality
                for the per-PR call pattern.
            max_tokens: Cap on the model's response size. Default 4096 — the
                JudgeVerdict schema is small, so this leaves headroom for
                adaptive thinking without forcing a stream.
        """
        self._client = client or anthropic.Anthropic()
        self._model = model or os.environ.get("CRS_JUDGE_MODEL", self.DEFAULT_MODEL)
        self._effort = effort or os.environ.get("CRS_JUDGE_EFFORT", self.DEFAULT_EFFORT)
        self._max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Call Claude and return a structured verdict.

        Raises:
            anthropic.APIError: surfaces SDK errors to the caller (the harness
                catches these and falls back to the heuristic score so a
                transient API outage does not break the CI step).
        """
        t0 = time.perf_counter()
        user_content = self._render_user_message(request)

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": JUDGE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            output_format=JudgeVerdict,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
        )

        verdict: JudgeVerdict = response.parsed_output
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResponse(
            risk_score=verdict.risk_score,
            backend_used=Backend.CLAUDE,
            latency_ms=latency_ms,
            reasoning=verdict.reasoning,
            raw={
                "risk_level": verdict.risk_level,
                "top_risk_factors": list(verdict.top_risk_factors),
                "mitigations": list(verdict.mitigations),
                "usage": self._extract_usage(response),
                "model": self._model,
                "effort": self._effort,
            },
            tier_reached=2,
        )

    def name(self) -> Backend:
        return Backend.CLAUDE

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _render_user_message(request: PredictionRequest) -> str:
        """Build the per-request user prompt.

        Layout matches the three sections promised in JUDGE_SYSTEM_PROMPT —
        diff, sub-agent observations, historical context. Empty sections are
        rendered as explicit "(no data)" markers rather than being elided, so
        the model knows the absence is real rather than a forgotten input.
        """
        parts: list[str] = []

        parts.append("# Section 1 — Diff under review")
        parts.append("")
        parts.append("```diff")
        parts.append(request.diff.rstrip())
        parts.append("```")
        parts.append("")

        parts.append("# Section 2 — Sub-agent observations")
        parts.append("")
        if not request.sub_agent_reports:
            parts.append("(no sub-agent reports provided)")
        else:
            for report in request.sub_agent_reports:
                parts.extend(_format_sub_agent_report(report))
        parts.append("")

        parts.append("# Section 3 — Historical context (RAG)")
        parts.append("")
        if not request.rag_documents:
            parts.append("(no retrieved documents)")
        else:
            for doc in request.rag_documents:
                parts.extend(_format_rag_document(doc))
        parts.append("")

        if request.pr_metadata:
            parts.append("# Section 4 — PR metadata")
            parts.append("")
            for key in sorted(request.pr_metadata):
                value = request.pr_metadata[key]
                parts.append(f"- **{key}**: {value}")
            parts.append("")

        parts.append("Return a JudgeVerdict JSON object now.")
        return "\n".join(parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Pull cache + token usage out of the Anthropic response object.

        The Anthropic SDK exposes a Pydantic-shaped `usage` object on the
        response; we down-convert to a plain dict so the PredictionResponse
        remains pickleable and JSON-serializable for the audit store.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", 0
            )
            or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
            or 0,
        }


def _format_sub_agent_report(report: dict[str, Any]) -> list[str]:
    name = report.get("name", "unknown")
    confidence = report.get("confidence", 0.0)
    out = [f"## {name} (confidence: {confidence:.2f})"]

    observations = report.get("observations") or {}
    if observations:
        out.append("")
        out.append("**Observations:**")
        for key in sorted(observations):
            out.append(f"- {key}: {observations[key]}")

    risk_factors = report.get("risk_factors") or []
    if risk_factors:
        out.append("")
        out.append("**Risk factors flagged:**")
        for factor in risk_factors:
            out.append(f"- {factor}")
    elif confidence > 0:
        out.append("")
        out.append("**Risk factors flagged:** none")

    out.append("")
    return out


def _format_rag_document(doc: dict[str, Any]) -> list[str]:
    layer = doc.get("layer", "?")
    retriever = doc.get("retriever", "?")
    score = doc.get("score")
    header = f"## Layer {layer} / {retriever}"
    if score is not None:
        header += f" (score: {score:.3f})"
    content = (doc.get("content") or "").rstrip()
    return [header, "", content, ""]


# ---------------------------------------------------------------------------
# Stubs — implementations pending real infrastructure.
# ---------------------------------------------------------------------------


class TritonNemoBackend(ModelBackend):
    """Locally-served Mistral-7B-v0.3 + LoRA fine-tune via NVIDIA Triton.

    Status: STUB — pending production fine-tune adapter from
    src/models/finetune/train_nemo.py.
    """

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError(
            "TritonNemoBackend stub — pending production fine-tune adapter."
        )

    def name(self) -> Backend:
        return Backend.TRITON_NEMO


class NIMBackend(ModelBackend):
    """NVIDIA NIM hosted inference (OpenAI-compatible API).

    Status: STUB — implementation lands when a NIM endpoint is provisioned.
    """

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError("NIMBackend stub — pending NIM endpoint provisioning.")

    def name(self) -> Backend:
        return Backend.NIM


class AzureOpenAIBackend(ModelBackend):
    """Azure OpenAI baseline (zero-shot judge) — used as a comparison anchor.

    Status: STUB.
    """

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError("AzureOpenAIBackend stub.")

    def name(self) -> Backend:
        return Backend.AZURE_OPENAI


# ---------------------------------------------------------------------------
# ModelGateway — routes requests across registered backends.
# ---------------------------------------------------------------------------


class ModelGateway:
    """Routes prediction requests across registered backends.

    Supports per-request backend selection (for A/B comparisons) and a
    `predict_hybrid` mode that combines a fine-tuned classifier's score with
    an LLM judge's reasoning. In v0.2 only the LLM judge half is live — the
    classifier path remains a stub — so `predict_hybrid` delegates to the
    judge alone when no classifier backend is registered.
    """

    def __init__(self, backends: dict[Backend, ModelBackend]):
        self.backends = backends

    def predict(self, request: PredictionRequest, backend: Backend) -> PredictionResponse:
        if backend not in self.backends:
            raise ValueError(
                f"Backend {backend} not registered. Available: {list(self.backends.keys())}"
            )
        return self.backends[backend].predict(request)

    def predict_hybrid(self, request: PredictionRequest) -> PredictionResponse:
        """Combine FT classifier score (Triton+NeMo) with LLM judge reasoning (Claude).

        v0.2 behaviour:
            - If a classifier backend (TRITON_NEMO or NIM) is registered and
              implemented, run it first to seed the cheap signal — currently
              both raise NotImplementedError, so we skip and rely on the
              judge alone.
            - The judge backend (CLAUDE) is required; we delegate to it for
              the score and reasoning.

        When the classifier backend ships in v0.3 this method will combine
        the two scores (weighted by calibration on a held-out set) rather
        than blindly trusting the judge.
        """
        if Backend.CLAUDE not in self.backends:
            raise RuntimeError(
                "predict_hybrid requires the Claude judge backend to be registered."
            )
        return self.backends[Backend.CLAUDE].predict(request)
