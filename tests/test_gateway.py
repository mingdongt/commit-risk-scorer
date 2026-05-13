"""Tests for ModelGateway and ClaudeJudgeBackend.

The judge backend is tested with a mocked Anthropic client — no real network
calls. The mock asserts the request shape (system prompt cached, adaptive
thinking, structured output) and returns a canned JudgeVerdict so the
PredictionResponse path is exercised end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.gateway import (
    Backend,
    ClaudeJudgeBackend,
    ModelGateway,
    PredictionRequest,
    TritonNemoBackend,
)
from src.models.judge_prompt import JUDGE_SYSTEM_PROMPT, JudgeVerdict


# ---------------------------------------------------------------------------
# Mock Anthropic client
# ---------------------------------------------------------------------------


class _FakeMessagesAPI:
    """Captures the last `parse()` call and returns a canned response.

    The returned object mimics the surface the production SDK exposes:
    `parsed_output` (a JudgeVerdict instance) and `usage` (cache + token
    fields). Anything the production code does NOT access is omitted on
    purpose — if the test starts caring about a new field, add it explicitly.
    """

    def __init__(self, verdict: JudgeVerdict, usage: dict[str, int] | None = None):
        self._verdict = verdict
        self._usage = usage or {
            "input_tokens": 200,
            "output_tokens": 120,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 4500,
        }
        self.last_call: dict | None = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            parsed_output=self._verdict,
            usage=SimpleNamespace(**self._usage),
        )


class _FakeAnthropicClient:
    """Minimal stand-in for `anthropic.Anthropic` — only the .messages.parse path."""

    def __init__(self, verdict: JudgeVerdict, usage: dict[str, int] | None = None):
        self.messages = _FakeMessagesAPI(verdict, usage)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _verdict() -> JudgeVerdict:
    return JudgeVerdict(
        risk_score=0.72,
        risk_level="High",
        top_risk_factors=[
            "Null-user check removed from session.validate",
            "No new tests covering the affected path",
        ],
        reasoning=(
            "The diff removes the `user is not None` guard from session.validate. "
            "DiffAnalyzer confirms the file is auth-sensitive and TestImpactScout "
            "confirms no new tests were added. SME review required."
        ),
        mitigations=[
            "Add a unit test exercising validate(None, token) and asserting False",
            "Request review from @security-team",
        ],
    )


def _sample_request() -> PredictionRequest:
    return PredictionRequest(
        diff=(
            "--- a/src/auth/session.py\n"
            "+++ b/src/auth/session.py\n"
            "@@ -10,7 +10,7 @@\n"
            " def validate(user, token):\n"
            "-    if user is not None and user.is_active and token.is_valid:\n"
            "+    if user.is_active and token.is_valid:\n"
            "         return True\n"
            "     return False\n"
        ),
        pr_metadata={"author": "demo", "repo": "demo/repo"},
        sub_agent_reports=(
            {
                "name": "diff-analyzer",
                "confidence": 0.85,
                "observations": {"files_touched": 1, "additions": 1, "deletions": 1},
                "risk_factors": ["touches sensitive path: src/auth/"],
            },
            {
                "name": "agent-pr-auditor",
                "confidence": 0.95,
                "observations": {"author_class": "human-only"},
                "risk_factors": ["weakening of authentication predicate"],
            },
            {
                "name": "test-impact-scout",
                "confidence": 0.0,
                "observations": {},
                "risk_factors": [],
            },
        ),
        rag_documents=(
            {
                "layer": "A",
                "retriever": "similar_past_prs",
                "score": 0.81,
                "content": "PR-#1842 (reverted): removed null-user check in same file.",
            },
        ),
    )


# ---------------------------------------------------------------------------
# ClaudeJudgeBackend
# ---------------------------------------------------------------------------


def test_judge_returns_prediction_response_from_verdict():
    """Happy path: verdict comes back, fields map cleanly to PredictionResponse."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)

    result = backend.predict(_sample_request())

    assert result.backend_used == Backend.CLAUDE
    assert result.risk_score == 0.72
    assert result.tier_reached == 2
    assert "session.validate" in (result.reasoning or "")
    assert result.raw is not None
    assert result.raw["risk_level"] == "High"
    assert "Add a unit test" in result.raw["mitigations"][0]


def test_judge_call_pins_correct_model_and_thinking():
    """Defaults must match the project contract: Opus 4.7, adaptive thinking, high effort."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)
    backend.predict(_sample_request())

    call = fake_client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-opus-4-7"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert call["output_format"] is JudgeVerdict


def test_judge_caches_system_prompt():
    """The system prompt must carry `cache_control: ephemeral` for the cache to activate."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)
    backend.predict(_sample_request())

    system = fake_client.messages.last_call["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["text"] == JUDGE_SYSTEM_PROMPT
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_judge_user_message_renders_all_sections():
    """User prompt must include the diff, the sub-agent reports, and the RAG context."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)
    backend.predict(_sample_request())

    messages = fake_client.messages.last_call["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]

    assert "Section 1 — Diff under review" in content
    assert "src/auth/session.py" in content  # diff body present
    assert "def validate(user, token):" in content
    assert "Section 2 — Sub-agent observations" in content
    assert "diff-analyzer" in content
    assert "agent-pr-auditor" in content
    assert "weakening of authentication predicate" in content
    assert "Section 3 — Historical context" in content
    assert "similar_past_prs" in content
    assert "PR-#1842" in content


def test_judge_handles_empty_optional_sections():
    """Missing sub-agent reports / RAG documents render as explicit '(no ...)' markers."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)

    bare_request = PredictionRequest(
        diff="--- a/foo.md\n+++ b/foo.md\n@@ -1 +1 @@\n-old\n+new\n",
    )
    backend.predict(bare_request)

    content = fake_client.messages.last_call["messages"][0]["content"]
    assert "(no sub-agent reports provided)" in content
    assert "(no retrieved documents)" in content


def test_judge_exposes_usage_for_cache_verification():
    """raw['usage'] must surface cache_read_input_tokens so caching is auditable."""
    fake_client = _FakeAnthropicClient(
        _verdict(),
        usage={
            "input_tokens": 50,
            "output_tokens": 120,
            "cache_creation_input_tokens": 4700,
            "cache_read_input_tokens": 0,
        },
    )
    backend = ClaudeJudgeBackend(client=fake_client)

    result = backend.predict(_sample_request())
    usage = result.raw["usage"]
    assert usage["cache_creation_input_tokens"] == 4700
    assert usage["cache_read_input_tokens"] == 0
    assert usage["input_tokens"] == 50


def test_judge_latency_is_recorded():
    """latency_ms must be non-negative — proves the timer was wired."""
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)

    result = backend.predict(_sample_request())
    assert result.latency_ms >= 0.0


def test_judge_respects_env_var_overrides(monkeypatch):
    """CRS_JUDGE_MODEL / CRS_JUDGE_EFFORT env vars override the defaults."""
    monkeypatch.setenv("CRS_JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("CRS_JUDGE_EFFORT", "medium")
    fake_client = _FakeAnthropicClient(_verdict())
    backend = ClaudeJudgeBackend(client=fake_client)

    backend.predict(_sample_request())
    call = fake_client.messages.last_call
    assert call["model"] == "claude-sonnet-4-6"
    assert call["output_config"] == {"effort": "medium"}


# ---------------------------------------------------------------------------
# ModelGateway.predict_hybrid
# ---------------------------------------------------------------------------


def test_predict_hybrid_routes_to_claude_when_classifier_absent():
    """v0.2 fallback: with only the judge registered, predict_hybrid delegates to it."""
    fake_client = _FakeAnthropicClient(_verdict())
    judge = ClaudeJudgeBackend(client=fake_client)
    gateway = ModelGateway(backends={Backend.CLAUDE: judge})

    result = gateway.predict_hybrid(_sample_request())
    assert result.backend_used == Backend.CLAUDE
    assert result.risk_score == 0.72


def test_predict_hybrid_requires_judge():
    """Calling predict_hybrid with no Claude backend registered is a programmer error."""
    gateway = ModelGateway(backends={Backend.TRITON_NEMO: TritonNemoBackend()})
    with pytest.raises(RuntimeError, match="Claude judge backend"):
        gateway.predict_hybrid(_sample_request())


def test_gateway_predict_routes_by_backend_enum():
    """gateway.predict(req, Backend.CLAUDE) dispatches to the correct backend."""
    fake_client = _FakeAnthropicClient(_verdict())
    judge = ClaudeJudgeBackend(client=fake_client)
    gateway = ModelGateway(backends={Backend.CLAUDE: judge})

    result = gateway.predict(_sample_request(), Backend.CLAUDE)
    assert result.backend_used == Backend.CLAUDE


def test_gateway_rejects_unregistered_backend():
    """Requesting a backend that wasn't registered raises with a useful message."""
    fake_client = _FakeAnthropicClient(_verdict())
    judge = ClaudeJudgeBackend(client=fake_client)
    gateway = ModelGateway(backends={Backend.CLAUDE: judge})

    with pytest.raises(ValueError, match="not registered"):
        gateway.predict(_sample_request(), Backend.NIM)
