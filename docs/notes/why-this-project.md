# Why I built commit-risk-scorer

> Draft of a public technical post — placement TBD (Medium / dev.to / personal site).
> Lives in the repo for now as transparent documentation of the project's motivation.
> If you're reading this in the repo, the post version is the same words, slightly polished.

---

## The unoccupied quadrant

If you map the engineering-productivity AI landscape today, tools cluster in three quadrants — and one corner is notably empty:

- **Static analysis** (rule-based, reactive): Google Tricorder, every linter your CI already runs.
- **Generative review** (LLM, reactive): PR-Agent, GitHub Copilot Code Review, AWS CodeGuru.
- **Traditional-ML prediction** (classical features, predictive): the internal CloudBuild failure-prediction systems at Microsoft, every commit-risk paper in ICSE proceedings.

The empty corner: **predictive + LLM/RAG/agent-driven**. A tool that *forecasts* risk before merge, *retrieves* relevant historical context, *reasons* about the change, and *recommends* an action — all in one open-source system.

That's what `commit-risk-scorer` is.

---

## Why this corner has been empty

Three honest reasons:

**1. Big-tech has done it — internally.** Google, Meta, Microsoft, and Amazon all run sophisticated commit-risk and build-failure prediction on their internal CI. Tricorder, Sapienz, TestImpact, CodeGuru — the published papers describe these systems, but the production code is closed. The gap is in *open-source artifacts*, not in *industry knowledge*.

**2. The stack only matured recently.** Code-aware embeddings (GraphCodeBERT, Voyage Code 3), long-context LLMs that fit a full diff plus retrieved history, standardized agent SDKs (Claude Agent SDK, MCP), NVIDIA's open NeMo + NIM + Triton + Garak — these all became production-ready in the last 18 months. A version of this project attempted in 2022 would have been substantially harder; one in 2027 may look obvious in retrospect.

**3. The market is narrow.** Commit-risk scoring matters most to large engineering organizations (1000+ developers) with internal platform teams and DORA-aware leadership. Not a mass-market product. Exactly an internal-platform-team brief.

---

## What I built

A hybrid pipeline, all Apache 2.0:

- A **multi-agent harness** (Claude Agent SDK) with five specialized sub-agents — `diff-analyzer`, `ownership-mapper`, `agent-pr-auditor`, `test-impact-scout`, `historical-context`.
- A **multi-vendor model gateway** unifying Claude, NVIDIA NIM, Triton-served NeMo fine-tunes, and Azure OpenAI behind one inference API — A/B-able across frontier vs. internal-hosted models.
- A **predictive classifier** — Mistral-7B-v0.3 fine-tuned via NVIDIA NeMo + LoRA on Microsoft CodeXGLUE Devign plus scraped GitHub PR/CI outcomes. Smoke-test path on DistilBERT for CPU-only dev environments.
- A **policy gatekeeper** that translates score into action (fast-track / owner-review / SME-review / block).
- A **DORA impact dashboard** measuring cycle time, change failure rate, MTTR, and adoption.
- A **multi-backend audit store** (MongoDB / MySQL / Elasticsearch) for traceability.
- An **enterprise-safety layer** — NVIDIA Garak red-team probes, NeMo Guardrails for output constraints, eval-gated CI with pytest regression gates.

Every change passes 86 tests. Every architectural decision is recorded in [`docs/design-doc.md`](../design-doc.md). Every limitation is in [`docs/limitations.md`](../limitations.md).

---

## Honest framing: integrator, not inventor

I'm not claiming to invent a new category. Every component has been independently validated:

- JIT defect prediction has a decade of academic work (Kamei 2013, DeepJIT, CC2Vec, JITLine, the PROMISE benchmark).
- Hybrid classifier-plus-judge patterns are common in search re-ranking, content moderation, and recommendation systems.
- Multi-vendor model gateways are standard enterprise inference infrastructure.
- NVIDIA's open AI stack components are production-grade.

The contribution is **the combination**: a single runnable artifact composing these pieces with platform-team operating discipline, in the open. That's the unoccupied quadrant.

---

## What's still missing

The smoke-test fine-tune ran on CPU using DistilBERT + LoRA at F1 = 0.38 — *pipeline validation, not capability benchmark*. The production target — Mistral-7B-v0.3 on NeMo + LoRA — is pending CUDA + scaled-up training data. Expected F1 from academic baselines on similar setups: 0.55–0.75.

The DORA dashboard runs on simulated data. Real impact numbers require real deployment.

A handful of edge cases in the AI-authored-PR detection sub-agent still need calibration — currently overweights "prompt-vs-diff drift" as a signal.

I document all of this in [`docs/limitations.md`](../limitations.md). Read it before forming a judgment.

---

## Why I'm doing this in public

Two reasons, both honest:

**1. The category needs to exist as open source.** Every adjacent tool is closed. Closing this loop in public is a real contribution to the developer-productivity-AI ecosystem.

**2. The artifact is the proof; the learning is half the value.** Building this requires hands-on familiarity with NVIDIA's open AI stack and the rhythm of internal platform teams (eval-gated CI, runbooks, postmortems, partner-team onboarding). I am transparent about the motivation — this is both a working tool and a public showcase of how I approach enterprise AI tooling.

---

## If you're at an engineering org with a platform team thinking about this space

[Open an issue](https://github.com/mingdongt/commit-risk-scorer/issues). I'd like to know which assumptions hold up against your reality.

---

*[Mingdong (Eric) Tan](https://github.com/mingdongt) · 2026-05-10 · Repo: [commit-risk-scorer](https://github.com/mingdongt/commit-risk-scorer)*
