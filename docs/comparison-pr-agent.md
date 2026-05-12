# How commit-risk-scorer differs from PR-Agent

[PR-Agent](https://github.com/Codium-ai/pr-agent) (by Codium AI) is the most popular open-source AI code-review tool — ~11k stars, multi-LLM support, GitHub Action drop-in. A reasonable question from anyone evaluating this project:

> **"Why not just use PR-Agent?"**

This document answers that, by category.

---

## Mission

| | PR-Agent | commit-risk-scorer |
| --- | --- | --- |
| Primary goal | Generate human-quality review comments | Predict CI-failure / regression risk |
| Output style | Free-form prose | Calibrated numeric score + policy action |
| Decision support | Suggestions for the developer | Routing decisions for the CI |
| Optimized for | First-pass review (human-in-the-loop) | Pre-merge gating (agent-in-the-loop) |
| Cost model | One LLM call per `/review` | Hybrid: cheap classifier always + LLM judge selectively |

PR-Agent is the **review writer**. commit-risk-scorer is the **gatekeeper**. They cover different parts of the PR lifecycle — they are complements, not competitors.

---

## Architecture

| | PR-Agent | commit-risk-scorer |
| --- | --- | --- |
| Core pattern | Command handlers wrapping a single LLM call | Multi-agent harness with 5 sub-agents |
| Prediction model | None — purely generative | Hybrid: fine-tuned classifier + LLM judge |
| RAG / commit history | Light — usually just the diff in context | Yes — historical similar-PR retrieval over Elasticsearch |
| Fine-tuning | Not supported (config-based prompt customization only) | Yes — NeMo + LoRA on Mistral-7B-v0.3 |
| Output structure | Free-form markdown | Calibrated JSON + structured policy decision |
| Eval-gated CI | Not built-in | Yes — 86 regression tests in pytest |
| Audit / observability | Standard logging | Multi-backend audit store (Mongo / MySQL / ES) |
| Safety / red-team | Not built-in | NVIDIA Garak probes + NeMo Guardrails |

---

## Feature-by-feature

| Feature | PR-Agent | commit-risk-scorer |
| --- | --- | --- |
| Auto-summarize PR | ✅ | ⚠️ explanation generated only when risk > threshold |
| Suggest code improvements | ✅ | ❌ (out of scope) |
| Interactive `/ask` Q&A | ✅ | ❌ (out of scope) |
| Generate changelog | ✅ | ❌ |
| **Calibrated PR risk score** | ❌ | ✅ |
| **Forecast CI failure pre-merge** | ❌ | ✅ |
| **Recommend reviewer assignment** | ❌ | ✅ (`ownership-mapper` sub-agent) |
| **Detect AI-authored PRs** | ❌ | ✅ (`agent-pr-auditor` sub-agent) |
| **Detect prompt-vs-diff drift** | ❌ | ✅ |
| **Test impact analysis** | ❌ | ✅ (`test-impact-scout`) |
| **Historical-pattern retrieval (RAG)** | ❌ | ✅ (`historical-context`) |
| **DORA impact dashboard** | ❌ | ✅ |
| **NVIDIA-stack inference** | ❌ | ✅ (NIM + Triton + NeMo) |
| **Calibrated probability output** | ❌ | ✅ |
| **Policy-driven merge gating** | ❌ | ✅ (4-band gate keeper) |
| **Eval-gated regression suite** | ❌ | ✅ |
| **Multi-backend audit log** | ❌ | ✅ (Mongo / MySQL / ES) |

---

## License

| | PR-Agent | commit-risk-scorer |
| --- | --- | --- |
| License | AGPL-3.0 | Apache 2.0 |
| Implication for adopters | Viral for hosted services; legal review often needed | Permissive — drop-in for enterprises |

The Apache 2.0 choice is deliberate: it matches the licensing of every NVIDIA OSS dependency (NeMo, Triton, Garak, NeMo Guardrails) and makes enterprise adoption frictionless.

---

## When to use which

- **Use PR-Agent if** you want LLM-written review comments, automatic summaries, and interactive `/ask` Q&A. PR-Agent is the best in class here.
- **Use commit-risk-scorer if** you want pre-merge risk gating, predictive scoring, reviewer routing, test-impact analysis, and DORA-style impact measurement.
- **Use both** — they cover different parts of the PR lifecycle.

---

## Why not just fork PR-Agent and add prediction?

This was considered. Three reasons against:

### 1. License compatibility

AGPL-3.0 (PR-Agent) and Apache 2.0 (NeMo / Triton / Garak / Guardrails — the stack we depend on) don't compose cleanly. Apache 2.0 alignment with NVIDIA's open stack is **intentional** for ecosystem coherence.

### 2. Different architectural primitives

PR-Agent is built around single-LLM-call command handlers. commit-risk-scorer is built around multi-agent orchestration with five sub-agents, each retrieving evidence and emitting structured findings. Refactoring PR-Agent to the latter pattern is more work than greenfield construction with the right primitives from the start.

### 3. Different production discipline

commit-risk-scorer ships with:

- 86 eval-gated tests
- Regression-gated CI workflow
- Runbook (`docs/runbook.md`)
- Postmortem template (`docs/postmortem-template.md`)
- Onboarding guide for adopting teams (`docs/onboarding.md`)
- Enterprise-safety controls (`docs/enterprise-safety.md`)
- Multi-backend audit-log storage
- DORA impact dashboard

Adding that scaffolding to a fork is comparable work to building from scratch — without the architectural-primitive mismatch.

---

## Could the two be combined in production?

Yes. The natural integration:

```
                       ┌────────────────────┐
PR opened ────────────►│ commit-risk-scorer │
                       └────────┬───────────┘
                                │ risk score + policy
                                ▼
                       ┌────────────────────┐
                       │ Policy Gatekeeper  │
                       └────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
           Low risk        Medium / High      Critical
              │                 │                 │
              ▼                 ▼                 ▼
        fast-track        PR-Agent writes      Block merge
        (no review)       a review →           Manual gate
                          human reviewer
                          looks at PR-Agent's
                          notes + diff
```

This treats PR-Agent as **one of several escalation paths**, invoked when policy decides human review adds value. commit-risk-scorer handles the *routing*; PR-Agent handles the *reviewing*.

---

## Acknowledgements

PR-Agent set the bar for what an open-source AI code-review tool should look like. commit-risk-scorer is the predictive complement, not a replacement.

---

*Last updated: 2026-05-10*
