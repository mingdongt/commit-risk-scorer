# commit-risk-scorer — Documentation Index

> Entry point. Use this page to decide what to read.
>
> The project README (one level up) is the front door; this index is for going
> deeper.

---

## If you have X minutes

| Time | Read |
| --- | --- |
| **5 min** | Project [README](../README.md) §*What it does* + §*Demo* |
| **15 min** | + [`agentic-sdlc-architecture.md`](agentic-sdlc-architecture.md) — system framing + this repo as Node #1 of 5 |
| **30 min** | + [`design-doc.md`](design-doc.md) §*Architecture* + [`limitations.md`](limitations.md) |
| **60 min** | Full [`design-doc.md`](design-doc.md) + [`evaluation.md`](evaluation.md) + [`enterprise-safety.md`](enterprise-safety.md) |
| **90+ min** | Everything; start from "By audience" below |

---

## By audience — what to read in what order

### I'm a hiring manager / recruiter (~15 min)

Goal: understand what this is, how it fits the JD, whether to ask engineers to look.

1. [README](../README.md) — front door, demo, where this fits in the SDLC system
2. [`agentic-sdlc-architecture.md`](agentic-sdlc-architecture.md) — 5-agent vision, roadmap, JD mapping
3. [`notes/why-this-project.md`](notes/why-this-project.md) — author's framing (motivation + scope honesty)

### I'm an engineer evaluating the architecture (~45 min)

Goal: understand how it works, what's principled, what doesn't work.

1. [`design-doc.md`](design-doc.md) — full architecture (Tiered Router, Heterogeneous RAG, Hybrid Scoring)
2. [`evaluation.md`](evaluation.md) — three-layer eval methodology (model / agent / business)
3. [`limitations.md`](limitations.md) — what doesn't work or works less well (read before forming judgment)
4. [`comparison-pr-agent.md`](comparison-pr-agent.md) — how this differs from PR-Agent

### I'm a platform-team lead considering adoption (~30 min)

Goal: understand how to integrate it, what safety controls are baked in, how to operate it.

1. [`onboarding.md`](onboarding.md) — integration steps + 4-phase rollout
2. [`enterprise-safety.md`](enterprise-safety.md) — 6 production-safety controls (source-code boundary, PII redaction, audit log, etc.)
3. [`runbook.md`](runbook.md) — what to do when the agent misfires (severity levels + failure modes)
4. [`metrics.md`](metrics.md) — DORA metric definitions + estimation methodology honesty
5. [`postmortem-template.md`](postmortem-template.md) — incident template, kept blameless

### I'm comparing this to other tools (~10 min)

1. [`comparison-pr-agent.md`](comparison-pr-agent.md) — head-to-head with PR-Agent
2. [README §*Why this project exists*](../README.md#why-this-project-exists) — quadrant positioning vs Tricorder / Sapienz / TestImpact / CodeGuru / Copilot Code Review

---

## By topic

### Architecture & vision

- [`design-doc.md`](design-doc.md) — Node #1 deep architecture: Tiered Router (T1 GBDT → T2 NeMo+LoRA → T3 Claude+RAG → T4 feedback), Hybrid Scoring composition, Heterogeneous RAG Layer A/B/C taxonomy, NVIDIA-Native Agent Stack Alternative, AI-Generated PR Risk roadmap
- [`agentic-sdlc-architecture.md`](agentic-sdlc-architecture.md) — The 5-agent SDLC System this repo is Node #1 of: per-node interfaces, NVIDIA-IPP value mapping, shared infrastructure, "if I had one more week" prioritization

### Evaluation & honesty

- [`evaluation.md`](evaluation.md) — Three eval layers: model (F1 / PR-AUC / ECE), agent (groundedness / Garak red-team), business (DORA + adoption)
- [`metrics.md`](metrics.md) — DORA definitions + §*Estimation honesty* (how the "~5–8 hours/week saved" number was derived; what's deliberately not measured)
- [`limitations.md`](limitations.md) — 13 documented limitations across data / model / evaluation / deployment

### Adoption & operations

- [`onboarding.md`](onboarding.md) — Integration steps; 4-phase rollout (shadow → 10% canary → 50% → GA)
- [`enterprise-safety.md`](enterprise-safety.md) — 6 controls: source-code boundary, PII redaction, offline fallback, advisory-by-default, human approval, audit log
- [`runbook.md`](runbook.md) — Severity matrix + 4 failure modes with response actions
- [`postmortem-template.md`](postmortem-template.md) — Standard SRE postmortem template

### Comparisons & context

- [`comparison-pr-agent.md`](comparison-pr-agent.md) — Mission, architecture, feature-by-feature vs PR-Agent (with integration sketch)
- [`notes/why-this-project.md`](notes/why-this-project.md) — Public-facing blog draft on the unoccupied quadrant + integrator-not-inventor framing

---

## Document status

All docs current as of 2026-05-13.

| Doc | Version |
| --- | --- |
| `design-doc.md` | v0.3 |
| `agentic-sdlc-architecture.md` | v0.1 |
| `evaluation.md` / `metrics.md` / `limitations.md` / `enterprise-safety.md` / `onboarding.md` / `runbook.md` / `postmortem-template.md` / `comparison-pr-agent.md` | Last updated 2026-05-10 |
| `notes/why-this-project.md` | Blog draft, 2026-05-10 |

For anything dated earlier than the "last updated" footer of an individual file, treat that file as the source of truth.
