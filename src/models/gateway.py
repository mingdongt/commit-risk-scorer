"""Multi-vendor model gateway: unifies Claude, NVIDIA NIM, Triton-served NeMo, and
Azure OpenAI behind a single inference API.

This module defines the routing surface and provides initial backend stubs. Full
implementations land incrementally — see `docs/design-doc.md` §Architecture for
the per-backend rollout. The gateway pattern mirrors how internal platform teams
typically run mixed frontier + internal-hosted model fleets.

Status (v0.1):
    - Interfaces and request/response dataclasses: DONE
    - All backend implementations: STUB (raise NotImplementedError)
    - Hybrid scoring (FT classifier + LLM judge): STUB
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Backend(Enum):
    CLAUDE = "claude"
    NIM = "nvidia_nim"
    TRITON_NEMO = "triton_nemo"
    AZURE_OPENAI = "azure_openai"


@dataclass
class PredictionRequest:
    """Request to score a commit / pull request."""

    diff: str
    pr_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass
class PredictionResponse:
    """Risk-scoring output."""

    risk_score: float  # in [0.0, 1.0]
    backend_used: Backend
    latency_ms: float
    reasoning: str | None = None  # populated by LLM-judge backends, None for pure classifiers
    raw: dict[str, Any] | None = None


class ModelBackend(ABC):
    """Interface every backend must implement.

    Implementations should be thread-safe — the gateway may dispatch concurrent
    requests across backends for A/B comparisons.
    """

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse: ...

    @abstractmethod
    def name(self) -> Backend: ...


class ClaudeJudgeBackend(ModelBackend):
    """LLM judge using the Claude Agent SDK with RAG over historical similar commits.

    Status: STUB — implementation lands in v0.2 (see design-doc.md §Architecture).
    """

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError(
            "ClaudeJudgeBackend stub — implementation pending. "
            "See docs/design-doc.md §Architecture."
        )

    def name(self) -> Backend:
        return Backend.CLAUDE


class TritonNemoBackend(ModelBackend):
    """Locally-served Mistral-7B-v0.3 + LoRA fine-tune via NVIDIA Triton.

    Status: STUB — pending fine-tune adapter from train_nemo.py.
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


class ModelGateway:
    """Routes prediction requests across registered backends.

    Supports per-request backend selection (for A/B comparisons) and a forthcoming
    `predict_hybrid` mode that combines a fine-tuned classifier's score with an
    LLM judge's reasoning.
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

        Status: STUB — see design-doc.md §Architecture §Hybrid Decision Layer.
        """
        raise NotImplementedError("Hybrid pipeline stub — v0.2.")
