# commit-risk-scorer — Design Document

> Status: **Draft v0.3** — last updated 2026-05-12.
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

### 1. Adjacent tools exist in big-tech — but not this specific combination

The engineering-productivity AI space at scale contains several production tools, each occupying a different quadrant of the design space. They are *not* equivalents of this project — they solve adjacent problems:

| Tool | Actual category | Public status |
| --- | --- | --- |
| Google Tricorder | Static analysis platform (rule-based) | Paper-public, code closed |
| Meta Sapienz | Search-based test generation | Paper-public, code closed |
| Meta Getafix | Automated bug repair | Paper-public, code closed |
| Microsoft TestImpact | Test selection / impact analysis | Internal |
| Microsoft CloudBuild fail-pred | Traditional-ML build-failure prediction | Internal / paper |
| Amazon CodeGuru Reviewer | ML-based PR review (SaaS) | Closed product |
| GitHub Copilot Code Review | Generative LLM review | Closed product |

None of these has the same shape as this project:

- **Static analysis tools** (Tricorder) are rule-based, not predictive.
- **Test generation / repair** (Sapienz, Getafix) tackle different parts of the lifecycle.
- **Test selection** (TestImpact) is commit-aware but answers *"which tests to run"*, not *"how risky is this change"*.
- **Build-failure prediction** (MS CloudBuild) is the closest functional analog but uses traditional ML, not LLM agents or commit-history RAG.
- **Generative review** (CodeGuru, Copilot Code Review, PR-Agent) is reactive prose, not a calibrated predictive score.

The specific combination this project ships — **predictive scoring + LLM agent orchestration + commit-history RAG + NVIDIA-native serving** — sits in an unoccupied quadrant. Academically the predictive side is well established (DeepJIT, CC2Vec, JITLine, PROMISE benchmark), but no open-source artifact integrates it with the rest. See the README §*Why this exists* for the quadrant diagram.

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

The system is a **tiered router** that escalates expensive computation only when cheap signals warrant it. The same architecture maps 1:1 to the JD's five "What you'll be doing" bullets and the three "Ways to stand out" differentiators.

### Tier breakdown

| Tier | Component | Triggered for | Output | Target latency | Target cost/call |
|---|---|---|---|---|---|
| **T1** | Classical-ML gate — **cuML GBDT** (sklearn fallback) on 30–50 engineered commit features | 100% of PRs | `risk_score_1` + SHAP top-3 | <10 ms | ~$0 |
| **T2** | Fine-tuned LLM classifier — **Mistral-7B-v0.3 + LoRA via NVIDIA NeMo**, served on Triton | `risk_score_1 ≥ 0.6` (~20% of PRs) | `risk_score_2` (calibrated) + structured risk tags | 1–3 s | ~$0.001 (local inference) |
| **T3** | Agentic RAG — **Claude Sonnet 4.6** with multi-source enterprise RAG gateway | `risk_score_2 ≥ 0.8` (~5% of PRs) | `risk_score_3` + NL report + cited evidence | 10–30 s | ~$0.05–0.20 |
| **T4** | Feedback loop — nightly job mining `(prediction, merge outcome, revert/incident link)` | Continuous, asynchronous | Retrain artifact for T1; calibration update for T2 | N/A | N/A |

**Routing invariant:** every PR stops at the lowest tier that produces a confident answer. The 95% of low-risk PRs never invoke an LLM. This is the system's operating-cost floor and the reason a hosted deployment is economically viable at large-org PR volume.

### Threshold calibration

Thresholds (0.6 / 0.8 in the table) are **operationally tunable** and derived from cost-balance economics:

- **T1→T2 threshold (0.6):** chosen so the expected cost of a false-negative at T1 (subsequent incident cost × P[bug | s1<θ]) equals the cost of invoking T2. Re-calibrated weekly from `FeedbackLog`.
- **T2→T3 threshold (0.8):** same logic against T3's higher cost. Both thresholds are exposed as deployment config, not hardcoded.

Documented in [`runbook.md`](runbook.md) under "Tuning operating thresholds".

### Hybrid scoring — how scores compose across tiers

**(Closes Open Question §1: "should the judge see the FT classifier's score before reasoning, or operate independently?")**

The composition is **sequential, not parallel**: T2 sees T1's score and SHAP factors as conditioning context; T3 sees T2's score and risk tags. This is preferred over independent-then-combine because:

1. **Calibration coherence** — each stage refines the prior rather than re-derives it from scratch, avoiding double-counting structural signals.
2. **Cost discipline** — independent invocation would force every tier to look at every PR; sequential conditioning makes early-exit possible.
3. **Explainability chain** — the final reasoning trace is a stack of refinements (`GBDT said X → FT said Y because Z → Agent verified via tools A, B`), which reviewers can audit.

The trade-off: T2/T3 are slightly biased by T1's view of the world. This is mitigated by (a) T2/T3 receiving the **full diff** independently, not just T1's score, and (b) T4's feedback loop catching systematic miscalibration over time.

### Multi-source enterprise RAG taxonomy (consumed by T3)

The JD's stand-out criterion of *"RAG and fine-tuning LLMs on enterprise data"* requires more than a similar-PR search. Real enterprise knowledge bases are heterogeneous; T3's RAG gateway dispatches across three retrieval layers, each with distinct chunking, embedding, and ranking strategies.

| Layer | Frequency | Source type | Examples | Retrieval strategy |
|---|---|---|---|---|
| **A. Code-adjacent operational data** | Always-on (every T3 call) | High-S/N structured signals | Past PRs + review comments, incident/postmortem store, CODEOWNERS + on-call, ADRs, build-failure RCA, dependency security advisories, coding standards | Voyage-Code-3 or CodeBERT-class embeddings + BM25 hybrid; commit-level chunking |
| **B. Operational context** | Metadata-triggered (e.g., diff touches oncall-critical service) | Time-series / tabular | Release calendar / freeze windows, on-call rotation + leave, feature-flag state, SLA & cost data, re-org / ownership transitions | Structured queries (SQL / KV lookup) augmented by light vector search; entity-level chunking |
| **C. Cross-functional enterprise knowledge** | LLM-judged (T3 agent decides based on diff semantics) | Heterogeneous prose / policy / domain | PRDs / roadmaps, compliance docs (GDPR / SOC2 / export control), customer-support themes / VoC, vendor contracts, strategic & OKR docs, internal Slack/Teams decision archives, **NVIDIA-specific domain KB** (hardware spec, ASIL automotive safety, GPU driver compat) | Long-context text embedding (general-purpose, not code-specialized) + cross-encoder re-ranker + strict provenance tracking |

**Design principle:** each layer is a pluggable retriever interface. OSS-mode implementations use public-data proxies (GitHub issues for "incidents", `/docs/adr/` for "ADRs", OSV for "advisories"); enterprise deployments swap the backend without changing the agent-facing tool signature. See [`../src/rag/`](../src/rag/) for the interface skeleton.

**Why three layers, not one flat tool list:** heterogeneous sources require heterogeneous retrieval pipelines. A single embedding + vector store cannot serve code, time-series, and long-form prose well. This layering is the architectural reason the system can claim "enterprise-data RAG", not just "similar-PR search".

### Cross-cutting components

- **Tiered router** — cascading cost layer ([`../src/models/tiered_router.py`](../src/models/tiered_router.py)); see §Tier breakdown above.
- **Feedback loop (T4)** — append-only prediction/outcome log ([`../src/storage/feedback_log.py`](../src/storage/feedback_log.py)) joined by `pr_id`, plus a nightly relabeler ([`../src/storage/outcome_labeler.py`](../src/storage/outcome_labeler.py)) that walks pending predictions past their observation window and writes labels via an adopter-supplied `SignalFetcher` (GitHub API / internal CI / ICM / Jira). Computes change failure rate and median MTTR directly — the DORA dashboard's source of truth and the next training run's labeled set.
- **Multi-vendor model gateway** — uniform `predict()` API across Claude, NVIDIA NIM, Triton-served NeMo fine-tune, and Azure OpenAI. Each Tier above selects its backend through the gateway; A/B comparison and graceful degradation are first-class.
- **Pluggable training-data sources for T2 fine-tuning** — `TrainingDataSource` abstract over OSS data (CodeXGLUE Devign + GitHub PR/CI scrapes) and enterprise data (internal commit history + internal CI + VoC + compliance labels). LoRA adapter design enables single-base / multi-tenant adapters.
- **Predictive classifier (T2)** — Mistral-7B-v0.3 (Apache 2.0 base) fine-tuned via NeMo + LoRA; compiled to a **TensorRT-LLM engine** via `trtllm-build` for low-latency Triton serving.
- **Classical-ML baseline (T1)** — NVIDIA RAPIDS **cuML GBDT** on engineered commit features; sklearn-backed implementation lives at [`../src/models/baselines/gbdt.py`](../src/models/baselines/gbdt.py) with cuML-swappable interface.
- **Audit / persistence** — multi-backend audit log (MongoDB / MySQL / Elasticsearch); required by [`enterprise-safety.md`](enterprise-safety.md) Control 6.
- **Safety layer** — NVIDIA NeMo Guardrails for output constraints; NVIDIA Garak probes integrated into eval CI.
- **DORA impact dashboard** (`streamlit`) — cycle time, change failure rate, MTTR, adoption rate, computed from T4 `FeedbackLog`.

### Data flow (single PR)

```
PR/Commit ──► T1 GBDT gate (always)
                │
                ├─ score < 0.6 ──► fast_track (95% of PRs end here)
                │
                └─ score ≥ 0.6 ──► T2 Mistral-LoRA (NeMo)
                                     │
                                     ├─ score < 0.8 ──► strong_review (~15% end here)
                                     │
                                     └─ score ≥ 0.8 ──► T3 Claude Agent + RAG Gateway
                                                          │
                                                          ├─ Layer A always queried
                                                          ├─ Layer B if metadata triggers
                                                          └─ Layer C if diff semantics trigger
                                                          │
                                                          └─► block_until_sme + report (~5%)

(asynchronous, every PR) ──► T4 FeedbackLog ──► nightly retrain T1, recalibrate T2 thresholds
```

### Mapping to JD requirements

| JD bullet / stand-out | Architectural component |
|---|---|
| Bullet 1: accelerate feedback loops, boost release reliability | T1 sub-10ms gate + T4 feedback loop |
| Bullet 2: design / build / deploy AI agents | T3 Claude Agent + RAG gateway with layered tool dispatch |
| Bullet 3: measure cycle time / CFR / MTTR | T4 + DORA dashboard |
| **Bullet 4: predictive models for high-risk commits / build-failure forecasting** ⭐ | T1 (classical ML) + T2 (fine-tuned LLM classifier) — direct 1:1 |
| Bullet 5: research emerging AI | Multi-vendor gateway + Garak / Guardrails integration |
| **Stand-out: RAG on enterprise data** | T3 three-layer enterprise RAG taxonomy (Layer A/B/C) |
| **Stand-out: Fine-tuning on enterprise data** | T2 Mistral + LoRA via NeMo, with pluggable `TrainingDataSource` |
| Stand-out: real-time + large-scale services | T1 <10ms gate + tier-based early exit |
| Stand-out: agentic AI for complex workflows | T3 agent's multi-step tool orchestration |

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

- ~~Should the judge see the FT classifier's score before reasoning, or operate independently and combine downstream?~~ **Resolved (v0.3):** *Sequential cascade with conditioning, not parallel ensemble.* T2 sees T1's score and SHAP factors as conditioning context (not replacement signal — T2 receives the full diff independently); T3 sees T2's score and risk tags the same way. Full rationale and trade-off analysis in [§Architecture → Hybrid scoring](#hybrid-scoring--how-scores-compose-across-tiers). The cascade is also what makes per-tier early-exit economically viable: at realistic PR volume an always-on judge is prohibitive, and 95% of commits are routine enough that T1's verdict is correct. Parallel ensemble remains a future option for the ~20% T2 band where both signals exist; not in v0.3 scope.
- What's the right balance between Devign (clean labels, security focus) and self-labeled GitHub data (noisy, broader)?
- How to simulate DORA dashboard data realistically when this isn't deployed in any real org?

---

## Inspired by Enterprise Engineering-Productivity Patterns

This project is not invented from scratch — it integrates patterns the industry
has been converging on for years. The combination is the contribution; each
component reflects an established enterprise practice:

- **PR-lifecycle automation** — automated reviewer routing, work-item linking,
  PR-summary generation, and PR-velocity dashboards have become standard inside
  large engineering orgs as PR throughput has grown.
- **AI build-failure diagnosis** — pre-merge prediction of CI failures (and
  post-merge root-cause assistance from logs, golden traces, and historical
  incident matches) is an active investment area across hyperscalers.
- **Security shift-left gates** — embedding a risk / security signal into the
  PR check surface, rather than only at deployment time, has emerged as the
  consensus design for high-velocity orgs that still want strong production
  safety properties.
- **Agentic-coding workflow telemetry** — as AI-assisted and agent-authored
  PRs become common, distinguishing the author class and adjusting risk
  scoring accordingly is a logical next step (see *AI-Generated PR Risk*
  below).
- **Reviewer / test recommendation systems** — using ownership signals, code
  embedding similarity, and historical failure mining to *recommend* the right
  human + the right tests, rather than running everything for every change.
- **Engineering-reliability loops** — closing the loop from PR signal →
  reviewer action → CI outcome → telemetry → model feedback, so the system
  improves with use rather than ossifying around its training distribution.

This project's contribution is a **single runnable artifact** that demonstrates
these patterns composing — not novel patterns, but coherent integration.

---

## AI-Generated PR Risk (Roadmap)

As AI-assisted and fully agent-authored PRs become a significant fraction of
total PR volume, the *author class* itself becomes a useful risk signal. The
v0.2 roadmap extends the scoring pipeline to capture this.

### PR-author classification

| Class | Definition |
| --- | --- |
| **human-only** | No AI assistance detectable in the PR's commit history. |
| **AI-assisted** | Author used Copilot-style autocomplete; commits show human-paced editing. |
| **agent-generated** | A full PR was authored by an agent (Copilot Agent, Devin-class systems, Claude Code, etc.) with the human acting as approver, not author. |

Detection signals (combined):

- Author identity (`actor.type == "Bot"` or known agent service accounts)
- Commit-message style (formulaic, machine-typical)
- Commit timing (humans pause to think; agents commit in bursts)
- PR description / scope markers added by agent runners

### Additional risk factors for agent-generated PRs

A diff produced by an agent has a different risk profile from one produced by a
human, even when both *look* similar. The agent-PR-specific factors:

| Risk factor | Why it matters |
| --- | --- |
| **Files modified outside requested scope** | Agents over-edit when the prompt is imprecise; broad blast radius without intent. |
| **Tests added but not behavior-covering** | Agents generate tests that exercise the new code path without asserting the intended behavior — passes CI, misses bugs. |
| **Large mechanical refactor** | A 1000-line "format / rename" PR may be a real cleanup, or may be an agent thrashing on an unclear goal. |
| **Missing human rationale** | No comment in commit message explaining *why* the change is correct — humans almost always include this; agents often omit. |
| **Prompt-to-diff drift** | The PR description (often regenerated from the prompt) doesn't match the changes shipped. |

### Why this matters now

In 2026 the question *"who will review the AI's code?"* is no longer
hypothetical — many engineering orgs already see double-digit percentages of
PRs originating from agents. The agent producing the PR and the agent scoring
it are two different agents with different incentives; that asymmetry is the
opening for a tool like this.

### Status

A v0.1 implementation of `AgentPRAuditor` ships in
[`../src/agent/sub_agents.py`](../src/agent/sub_agents.py). It supports:

- Author-class **inference** from a bot-shaped author login (`*[bot]`,
  `*-bot`, `copilot-*`, `devin-*`, `github-actions`) or from a commit-timing
  burst (≥ 5 commits within 5 minutes).
- Explicit `metadata["author_class"]` override (always wins).
- Three agent-PR-specific risk factors:
  - **Mechanical refactor** — 5+ files × ≥ 80 lines/file average diff size.
  - **Missing human rationale** — all commit messages are one-liners (no body).
  - **Scope drift** — `metadata["pr_description_paths"]` names paths absent
    from the diff (interpreted as prompt-vs-shipped-diff drift).

v0.2 upgrades this to:

- A trained classifier on PR-style features (commit-message embedding,
  inter-commit-time distribution, file-type entropy).
- LLM-judge-driven scope-drift detection that compares PR description text to
  diff content semantically (not just by path string matching).

---

## NVIDIA-Native Agent Stack Alternative

The v0.1 reference implementation uses the **Claude Agent SDK** for orchestration
because it's the most mature general-purpose agent SDK at time of writing and it
keeps the project independent of any single inference vendor. For an adopter
that wants the agent layer itself running on NVIDIA tooling (e.g., an internal
deploy that must keep code inside the org boundary), each layer has an NVIDIA-native
counterpart:

| Layer | v0.1 default | NVIDIA-native alternative |
| --- | --- | --- |
| Agent orchestration | Claude Agent SDK | **NVIDIA AIQ Toolkit** + NVIDIA AI Blueprints (Agent RAG / Multi-agent reference architectures) |
| Tool calling | Claude / Azure OpenAI function-calling | **NIM-hosted function-calling LLMs** (Llama-3.x Nemotron, Mistral-Nemo NIMs) |
| Memory / retrieval | Custom RAG over Elasticsearch | **NVIDIA NeMo Retriever** (multi-modal retriever models + reranker) |
| Output safety | NeMo Guardrails (already used) | NeMo Guardrails (no change — already NVIDIA-native) |
| Inference backend | Anthropic / Azure OpenAI | **NIM** (managed API) or **Triton + TensorRT-LLM** (self-hosted) |
| Dev environment | Standard Python / VS Code | **NVIDIA AI Workbench** (containerized AI dev environment) |

The model gateway (`src/models/gateway.py`) is already designed for this
substitution — every backend implements the same `predict()` interface, so an
adopter can swap the Claude judge for a NIM-served Nemotron with no caller-side
changes. The architectural cost of going fully NVIDIA-native is essentially
zero; the rollout cost is the engineering effort to provision the equivalents
(AIQ Toolkit setup, NIM endpoint provisioning, NeMo Retriever index
construction).

This alternative is documented (rather than implemented as v0.1's default)
because most adopters will want to start with the more familiar frontier-LLM
path and migrate inward as their compliance / cost profile dictates.

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

*— Mingdong (Eric) Tan, 2026-05-10*
