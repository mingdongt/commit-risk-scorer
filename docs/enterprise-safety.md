# Enterprise Safety: Controls for Running This in a Real Org

Most LLM-agent demos can ignore the difference between "this works on my laptop"
and "this is safe to run against a production engineering org's source code."
This project takes that gap seriously — the controls below are the
*non-negotiable* defaults, not extras.

If you're adopting `commit-risk-scorer` inside a company, this is the checklist
your platform / security team will ask you to walk through before turning it on.

---

## Control 1 — Source code stays inside the org boundary by default

**Rule**: PR diffs and source files are **not** sent to an external LLM
(Anthropic, OpenAI, etc.) unless the deployment policy explicitly allows it for
that repo.

**Implementation**:

- The multi-vendor model gateway ([`../src/models/gateway.py`](../src/models/gateway.py))
  defaults to a **local-first** routing policy: `Backend.TRITON_NEMO` (locally
  served fine-tune) is tried first; external backends are opt-in per repo
  configuration.
- A repo flag (`external_llm_ok: false`) blocks the gateway from selecting
  external backends regardless of caller request.
- The Triton-served NeMo fine-tune is **the entire production path** for orgs
  that cannot send code outside their boundary.

---

## Control 2 — PII / secret redaction before any model call

**Rule**: Diffs are scrubbed for credentials, customer identifiers, and
high-entropy tokens **before** reaching any model — local or external.

**Implementation** *(v0.2)*:

- Standard regex / entropy filters for AWS keys, GitHub tokens, JWT-shaped
  blobs, email addresses, IP addresses.
- Optional `trufflehog` integration for deeper secret detection.
- Redacted payloads carry a content hash so audit logs can prove what the model
  *actually* saw.

---

## Control 3 — Local / offline model fallback

**Rule**: The agent must produce a *defensible degraded result* when external
APIs are unavailable. No silent failure.

**Implementation**:

- Gateway tries backends in priority order; the **fine-tuned NeMo classifier
  alone** can produce a score + minimal evidence even when the LLM judge is
  unreachable.
- The output JSON sets `"reasoning": null` and `"confidence"` is reduced
  accordingly — the caller knows the result is classifier-only.
- An air-gapped deploy with only Triton + the local FT adapter is a supported
  configuration.

---

## Control 4 — All decisions are advisory unless explicitly configured as a gate

**Rule**: The agent's recommended actions (merge block, extra reviewer, extended
CI) are **suggestions** by default. A repo opts in to enforcement explicitly.

**Why**: A flag that surprises an engineer by silently blocking their merge
destroys adoption faster than any technical defect.

**Implementation**:

- Default mode: agent posts a comment with score + evidence + recommended
  actions; merge is not blocked.
- Enforcement is per-repo and per-severity: a team can opt in to *Critical →
  block* while leaving *High → advisory*.
- Enforcement changes require a tracked PR to the agent's config, reviewed by
  the repo's code owners (no console-only toggles).

---

## Control 5 — Human approval before merge-blocking actions

**Rule**: When enforcement is on, a *human* signs off on the block — the agent
proposes, a designated approver disposes.

**Implementation**:

- Block decisions surface as a `requires_review` GitHub Check that an approver
  from the repo's `CODEOWNERS` can resolve.
- The approver sees the agent's full evidence trail and can override with one
  click + a free-text reason (logged for audit).
- No agent action is final without either author acknowledgement (advisory) or
  approver disposition (enforced).

---

## Control 6 — Audit log: who, what, when, why

**Rule**: Every score, evidence trail, model version, and prompt version is
logged. A regression six months later must be traceable.

**Implementation**:

Every agent output emits an append-only log entry containing:

| Field | Why |
| --- | --- |
| `pr_id`, `commit_sha` | What was scored |
| `risk_score`, `risk_level`, `top_risk_factors` | The output the agent produced |
| `recommended_actions` | The action surface the user saw |
| `model_version`, `adapter_version` | Reproducibility |
| `prompt_version` | The exact judge prompt — prompts version like code |
| `backend_used` | Which model path served the request |
| `latency_ms` | Performance regression detection |
| `human_override` | Approver decision + reason (when enforced) |

The log is the source of truth for the **agent-eval** (Layer 2 in
[`evaluation.md`](evaluation.md)) and for any incident postmortem
([`postmortem-template.md`](postmortem-template.md)).

---

## Why this section exists

A common interview-time question — *"how does your project work in an
environment with real source-code-confidentiality, compliance, and audit
requirements?"* — has a short answer (the six controls above) and a longer
answer (the implementation links in each section). Both should be readable in
under five minutes.

If you're a platform-team lead evaluating this for adoption: skim the headlines
of all six controls; if anything looks weak for your environment, open an issue.

---

*Last updated: 2026-05-10*
