# commit-risk-scorer — Design Document

> Status: **Draft v0.2** — last updated 2026-05-10.
>
> This document captures motivation, prior art, design decisions, and eval methodology for `commit-risk-scorer`. It is the constitution of the project; code references and matches it.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Why This Gap Exists](#why-this-gap-exists)
3. [Problem Statement](#problem-statement)
4. [Prior Art](#prior-art)
5. [Hypothesis & Success Criteria](#hypothesis--success-criteria)
6. [Alternatives Considered](#alternatives-considered)
7. [Architecture](#architecture)
8. [Eval Methodology](#eval-methodology)
9. [Risks & Mitigations](#risks--mitigations)
10. [Limitations & Honest Caveats](#limitations--honest-caveats)
11. [Open Questions](#open-questions)
12. [References](#references)

---

## Motivation

### Layer 1 — Strategic context

This project sits at the intersection of three converging needs in modern software engineering:

1. **AI-native developer productivity is no longer optional.** Engineering organizations measure success in DORA terms — *cycle time*, *change failure rate*, *MTTR*. Pre-merge risk signal directly affects all three.
2. **Existing tools occupy non-overlapping corners.** Generative LLM review (PR-Agent) is calibration-free; trained classifiers (CodeBERT family) lack reasoning; LLM red-teaming (Garak) is not specialized for code-review agents.
3. **The integration is the contribution.** No current open-source project combines hybrid prediction (FT classifier + LLM judge), NVIDIA-native serving, and platform-team operating discipline (eval-gated CI, runbooks, partner-team onboarding) in one runnable artifact.

### Layer 2 — Problem context

Why specifically *commit risk scoring*, not generative review, test selection, or incident triage?

Commit risk scoring is uniquely well-positioned:

- **Public-data tractable** — GitHub PR + CI history can substitute for proprietary internal data, making the project fully reproducible.
- **Skill-coverage** — a single project naturally exercises predictive ML, RAG, agent orchestration, fine-tuning, eval discipline, and multi-vendor LLM integration.
- **Enterprise-valuable independent of any one company** — a working tool, not a portfolio piece.
- **Clean Input/Output** — `(diff, metadata) → (score, reasoning, action)` is a well-defined contract.

### Layer 3 — Design philosophy

The system is built to mirror the operating discipline of a real internal platform team — not a hackathon project. Concretely:

- **Multi-vendor model gateway** — any non-trivial enterprise AI team runs both internal-hosted models (NIM, Triton-served fine-tunes) and external frontier models (Claude, GPT). The gateway is the architectural reality, not a luxury.
- **Eval-gated CI** — every change to prompts, models, or routing logic passes pytest regression gates before merge.
- **DORA impact dashboard** — success is measured in cycle-time delta, change-failure-rate delta, MTTR delta, and adoption rate — not in F1 alone.
- **Partner-team onboarding doc** — shipping a tool is half the work; *getting another team to adopt it* is the other half.
- **Postmortem template + runbook** — when (not if) the agent misfires, the team has a structured response path.

### Layer 4 — Meta motivation

Building this project is a forcing function: it requires hands-on familiarity with NVIDIA NeMo, Triton, NIM, Garak, and NeMo Guardrails. The artifact is the proof; the learning is the real product.

---

## Why This Gap Exists

The integration this project performs — hybrid predictive pipeline + LLM agent harness + commit-history RAG, served on NVIDIA's open-source AI stack — does not exist as a single open-source artifact. That gap raises a fair question: *if it's so useful, why hasn't anyone shipped it?*

Four reasons, in order of importance:

### 1. Big-tech equivalents exist — they're just not open-sourced

Internal commit-risk and build-failure-prediction tooling has existed at scale for years:

- **Google** — Tricorder (static analysis + risk-flagging on changelists)
- **Meta** — Sapienz / Getafix (test generation + automated repair)
- **Microsoft** — TestImpact, internal CloudBuild failure prediction
- **Amazon** — CodeGuru Reviewer (commercialized as paid SaaS)
- **GitHub** — Copilot Workspace / Code Review (closed product)

The gap is in *open-source* tooling, not in *industry knowledge*. Big-tech treats this class of tool as competitive-advantage infrastructure rather than an ecosystem contribution.

### 2. The data is genuinely noisy — but the field is not new

CI outcomes conflate signal (bad commits) with noise (flaky tests, infra failures, upstream dependency churn). This is a real limitation, not a fatal one — academia has worked on **Just-In-Time (JIT) defect prediction** for over a decade with established benchmarks:

- Kamei et al. 2013 — JIT-QA framework
- DeepJIT (Hoang et al., 2019) — first deep-learning approach
- CC2Vec (Hoang et al., 2020) — commit-specific embeddings
- DeepLineDP (2022) — line-level prediction
- JITLine (Kondo et al., 2023) — LLM-augmented context

Published F1 baselines on standard datasets (PROMISE, Defects4J) sit in the **0.55–0.75** range — a defensible target.

### 3. The LLM + agent + code-embedding stack only matured in 2024–2026

The dependencies this project relies on are recent:

- Code-aware embedding models (CodeBERT/GraphCodeBERT 2020–2022; Voyage Code 2024)
- Long-context LLMs that fit a full diff plus retrieved context (2023–2024)
- Standardized agent SDKs and the MCP tool protocol (2024–2025)
- NVIDIA NeMo Guardrails / NIM / Garak — production-ready open releases (2024–2025)

A version of this project attempted in 2022 would have been substantially harder; one attempted in 2027 may look obvious in retrospect.

### 4. The market is narrow — but it is exactly the brief of an internal platform team

Commit-risk scoring matters most to:

- Large engineering organizations (1000+ developers)
- Internal platform / SRE / DevOps teams
- Organizations with DORA-aware leadership

A narrow segment for general-purpose product companies. An exact fit for an internal-platform group such as NVIDIA's IPP organization.

### Confidence calibration

This project is built as an **integrator, not an inventor**. Every component is individually validated:

| Component | Maturity |
| --- | --- |
| JIT defect prediction concept | Mature — 10+ years academic research, internal big-tech tools |
| Hybrid classifier + LLM judge | Pattern proven elsewhere (search re-ranking, content moderation) |
| Code-aware embedding RAG | Mature in codebase RAG; less explored on commit history |
| NVIDIA NeMo / Triton / Garak / Guardrails | Stable production-ready open-source releases |
| Claude Agent SDK + MCP tool federation | Recent but stable |

The contribution is the **combination** — a single runnable artifact demonstrating these components composing cleanly into a platform-team-grade tool.

---

## Problem Statement

*Coming in v0.2 — quantified pain (engineer-hours lost to CI failures), why current solutions fall short, target user persona.*

---

## Prior Art

*Coming in v0.2 — comparison table covering PR-Agent, CodeRabbit, Sweep, CodeBERT, Devign, DeepJIT, Garak, NeMo Guardrails, MS Research TestImpact, Facebook Getafix, Google Tricorder.*

Confirmed key references (full survey in v0.2):

- **[PR-Agent](https://github.com/Codium-ai/pr-agent)** — generative LLM review baseline. AGPL-3.0. ~11k stars.
- **[NVIDIA Garak](https://github.com/NVIDIA/garak)** — LLM red-teaming framework. Apache 2.0. ~7.8k stars.
- **[NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)** — programmable guardrails for LLMs. Apache 2.0. ~6.1k stars.
- **[Microsoft CodeXGLUE](https://github.com/microsoft/CodeXGLUE)** — code-intelligence benchmark; Devign defect-detection dataset.

---

## Hypothesis & Success Criteria

*Coming in v0.2.* Initial hypotheses to validate:

- **H1:** A hybrid pipeline (FT classifier + LLM judge) outperforms either component alone on F1.
- **H2:** RAG over historical similar commits improves judge calibration (lower over-confident false positives).
- **H3:** Garak red-team probe failure rate stays below 10% for the production judge prompt.

---

## Alternatives Considered

*Coming in v0.2 — extended trade-off table. Initial sketch:*

| Alternative | Why not chosen |
| --- | --- |
| Just use PR-Agent | No predictive scoring layer |
| Just train a CodeBERT-class classifier | No reasoning, no contextual retrieval |
| Single fine-tuned LLM (no judge) | No grounding signal, no zero-shot fallback |
| **Hybrid (FT classifier + RAG judge)** ✓ | Combines calibration with reasoning; covers OOD via judge |

---

## Architecture

*Component diagram and data flow coming in v0.2.* Sketched in [README.md](../README.md#architecture-in-brief).

Key components:

- **Agent harness** — Claude Agent SDK with three sub-agents (`diff-analyzer`, `test-impact-scout`, `historical-context`).
- **Multi-vendor gateway** — single `predict()` API across Claude, NVIDIA NIM, Triton-served NeMo fine-tune, and Azure OpenAI.
- **Predictive classifier** — Mistral-7B-v0.3 (Apache 2.0 base) fine-tuned via NeMo + LoRA on CodeXGLUE Devign + self-labeled GitHub PR/CI outcomes.
- **RAG layer** — Elasticsearch index of historical PRs with embeddings, retrieved as judge context.
- **Safety layer** — NeMo Guardrails for output constraints; Garak probes in eval CI.

---

## Eval Methodology

*Coming in v0.2.* Initial design:

- **Holdout** — 20% of labeled commits, stratified by repo and outcome.
- **Metrics** — F1, precision, recall, AUC-ROC, calibration error.
- **Red-team** — NVIDIA Garak probe suite against the judge prompt.
- **Regression gates** — pytest CI fails the merge if F1 drops > 2 percentage points or any Garak probe regresses.

---

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| **Public PR data is noisy** — CI failures often caused by flaky tests, not commit quality. | Use CodeXGLUE Devign as cleaner pre-train signal; document limitations transparently. |
| **NVIDIA stack learning curve** — first-time NeMo/Triton setup is non-trivial. | Time-box: NeMo demo running by milestone 1, or downgrade to HuggingFace + LoRA. |
| **License compatibility** — Llama-3.1 has commercial restrictions. | Use Mistral-7B-v0.3 (Apache 2.0) as base instead. |
| **Numbers will be challenged** — "saved ~5–8 hours/week" methodology must be defensible. | Publish estimation methodology in [`docs/metrics.md`](metrics.md). |
| **Scope creep** — every NVIDIA tool feels worth integrating. | Stick to the five chosen (NeMo, NIM, Triton, Garak, Guardrails); explicitly mark others out-of-scope. |

---

## Limitations & Honest Caveats

This project's scope is deliberately bounded by what one engineer can validate end-to-end on public data with realistic compute. Where corner-cutting is unavoidable, this section names it explicitly.

### 1. Public CI data is noisy

GitHub PR + CI outcome data conflates code quality with flaky tests, infra failures, and upstream-dependency churn. Internal datasets at large engineering organizations have access to richer signals (test execution traces, author tenure, repo-specific tuning) that are not reproducible publicly.

**Practical implication:** Expected F1 on labeled public data sits in the **0.55–0.70** range, consistent with published JIT defect-prediction baselines on PROMISE / Defects4J. State-of-the-art internal systems likely outperform this; that comparison is out of scope.

### 2. Class imbalance — most commits pass CI

The positive class (CI failure / revert) is rare in any healthy repository. Both the classifier and the judge require explicit handling:

- **Classifier**: focal loss or class weighting; threshold calibration on a held-out validation set.
- **Judge**: prompt engineering and decision policy to avoid over-confident false positives on routine changes.

### 3. No production validation

The DORA impact dashboard renders simulated data on labeled holdouts. Real cycle-time / change-failure-rate / MTTR delta numbers require deployment in a real organization with controlled rollout — explicitly out of scope for this OSS artifact.

### 4. Big-tech internal equivalents likely outperform on accuracy

Tools listed in [§Why This Gap Exists §1](#1-big-tech-equivalents-exist--theyre-just-not-open-sourced) have access to signals this project cannot reproduce. The contribution here is **architecture viability and integration discipline on public data**, not state-of-the-art accuracy on proprietary signals.

### 5. NeMo / Triton / NIM stack assumes accessible compute

Production-target serving (Mistral-7B-v0.3 + Triton + NIM) requires a CUDA-capable environment. The current pipeline (see README "Initial Results") falls back to a CPU-friendly DistilBERT smoke-test on the laptop where this is being developed. Bringing the production target online is the next milestone, not a current claim.

---

## Open Questions

*Coming in v0.2. Current open questions:*

- Should the judge see the FT classifier's score before reasoning, or operate independently and combine downstream?
- What's the right balance between Devign (clean labels, security focus) and self-labeled GitHub data (noisy, broader)?
- How to simulate DORA dashboard data realistically when this isn't deployed in any real org?

---

## References

*Full bibliography coming in v0.2.* Key sources to date:

- DORA — *Accelerate State of DevOps Report*
- DeepJIT (Hoang et al., 2019) — just-in-time defect prediction
- CodeBERT (Feng et al., 2020) — pre-trained model for programming languages
- Microsoft CodeXGLUE (Lu et al., 2021) — benchmark for code intelligence
- NVIDIA Garak documentation — red-team probe taxonomy
- NVIDIA NeMo Guardrails documentation — Colang language reference

---

*— Mingdong (Eric) Tan, 2026-05-09*
