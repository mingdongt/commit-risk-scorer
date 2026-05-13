# commit-risk-scorer

[![eval](https://github.com/mingdongt/commit-risk-scorer/actions/workflows/eval.yml/badge.svg)](https://github.com/mingdongt/commit-risk-scorer/actions/workflows/eval.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> A **shift-left engineering intelligence agent** that predicts PR risk, recommends **reviewer / test / gate actions**, and **closes the loop** through CI, telemetry, and DORA-style engineering metrics. Built on NVIDIA's open AI stack with a hybrid predictive pipeline (FT classifier + LLM judge).

## Why I built this

Two reasons, both honest:

1. **There's a real gap in open-source tooling.** Generative LLM PR reviewers (PR-Agent) and trained predictive defect models (CodeBERT family) sit in different corners; no current OSS project integrates them with platform-team operating discipline. This project ships that integration.

2. **The artifact is the proof; the learning is part of the value.** Building this requires hands-on familiarity with NVIDIA's open AI stack (NeMo, Triton, NIM, Garak, NeMo Guardrails) and the rhythm of internal platform teams (eval-gated CI, runbooks, postmortems, partner-team onboarding). The motivation is candid — this is both a working artifact for adopting teams and a public showcase of how I approach enterprise AI tooling.

## What it does

**Input**: PR diff + metadata (author, files, target branch) + build/test history + ownership signals.

**Output**:

1. **Risk score** (0–100) and **risk level** (Low / Medium / High / Critical)
2. **Top risk factors** — evidence-backed (file-ownership gaps, weak test coverage, historically failing areas, deployment blast radius)
3. **Recommended actions** — reviewer assignment, test suite to run, gate decision (not just a numeric signal)
4. **DORA-style impact telemetry** — cycle time, change failure rate, MTTR, adoption, FP/FN feedback

The risk score is **not the product** — the *action* is. Score feeds into a policy decision surface:

### Risk → Action mapping

| Score | Level | Action |
| --- | --- | --- |
| 0–20 | Low | Fast-track / normal review |
| 21–50 | Medium | Add code-owner reviewer + targeted tests |
| 51–80 | High | Require SME review + extended CI |
| 81–100 | Critical | Block merge / manual gate |

### Example output

```json
{
  "riskScore": 72,
  "riskLevel": "High",
  "topRiskFactors": [
    "Touches auth middleware (high-incident area)",
    "No test coverage for the modified branch",
    "Similar historical PRs caused CI failures"
  ],
  "recommendedActions": [
    "Add security / code-owner reviewer",
    "Run extended integration test suite",
    "Block auto-merge until reviewer approval"
  ],
  "confidence": 0.81,
  "evidence": [
    "Changed file: src/auth/token_validator.py — owned by @security-team",
    "Historical match: PR #1842 failed `test_auth_session_refresh`",
    "Test impact: 0 of 12 covering tests modified"
  ]
}
```

Designed to run as a CI check on every PR — providing **pre-merge predictive signal that drives policy decisions**, not just numeric scores. See [`docs/evaluation.md`](docs/evaluation.md) for how each layer is measured and [`docs/enterprise-safety.md`](docs/enterprise-safety.md) for the production-safety controls.

## Demo — three scenarios, end-to-end

The repo ships a runnable demo that walks three real-shaped PRs through the
full pipeline (`5 sub-agents → policy gatekeeper → markdown PR comment`). The
captured output lives in [`demo/output.md`](demo/output.md); regenerate it
with `python -m demo.run_demo > demo/output.md`.

| Scenario | What it is | Risk score | Risk level | Action |
| --- | --- | --- | --- | --- |
| **A** | README typo fix by a regular contributor | 0.00 | 🟢 Low | `fast_track` |
| **B** | 4-file refactor inside `src/auth/` (sensitive path), CODEOWNERS provided | 0.46 | 🟡 Medium | `owner_review` |
| **C** | Bot-authored 8-file mechanical refactor; PR description claims paths absent from the diff (prompt-vs-diff drift) | 1.00 | 🔴 Critical | `block_merge` |

The output that's posted on the PR is real markdown — see
[`demo/output.md`](demo/output.md) for the verbatim agent comments for each
scenario, including evidence and sub-agent reports.

## Why this exists

Existing solutions occupy one corner of the design space:

| Tool | Approach | Limitation |
| --- | --- | --- |
| [PR-Agent](https://github.com/Codium-ai/pr-agent) (Codium) | Generative LLM review | No predictive scoring or calibration |
| CodeBERT / [Devign](https://github.com/microsoft/CodeXGLUE) | Trained classifier | No reasoning or agent integration |
| [NVIDIA Garak](https://github.com/NVIDIA/garak) | LLM red-teaming | Not specialized for code-review agents |

This project is the integration that unifies all three. Adjacent big-tech tools cover individual quadrants — Google Tricorder for static analysis, Meta Sapienz / Getafix for test generation and automated repair, Microsoft TestImpact for test selection, Microsoft CloudBuild for traditional-ML build-failure prediction, Amazon CodeGuru and GitHub Copilot Code Review for generative review — but **none combines predictive scoring + LLM agent orchestration + commit-history RAG in a single open-source artifact**. Academically the predictive side is well established (DeepJIT, CC2Vec, JITLine, PROMISE benchmark), but no open-source project integrates it with the rest.

The landscape mapped to four quadrants:

```
                          Predictive
                              ▲
                              │
           ┌──────────────────┼──────────────────┐
           │  MS CloudBuild   │  THIS PROJECT    │
           │  failure-pred    │  commit-risk-    │
           │  MS TestImpact   │  scorer          │
           │  (classical ML)  │  (LLM + RAG +    │
           │                  │   agent)         │
           ├──────────────────┼──────────────────┤
           │  Google          │  PR-Agent        │
           │  Tricorder       │  GH Copilot      │
           │  (rule-based     │  Code Review     │
           │   static)        │  AWS CodeGuru    │
           │                  │  (generative)    │
           └──────────────────┼──────────────────┘
                              ▼
                   Reactive / Generative

          ◄─── Rules / Static          LLM / RAG ───►
```

The **top-right quadrant** — *predictive + LLM/RAG/agent-driven* — is the unoccupied space this project fills. See [`docs/design-doc.md`](docs/design-doc.md) §*Why This Gap Exists* for the full prior-art breakdown and honest caveats.

## Where this fits — Agentic SDLC System

This repo is **Node #1** of a 5-agent **Agentic SDLC System** — a multi-agent
platform that uses LLMs + agentic AI to automate end-to-end software-engineering
workflows and measure their impact in DORA terms. Node #1 (Pre-Merge Risk
Workflow) is shipped here to production-shape; Nodes #2–#5 are scoped as
roadmap with concrete interfaces, deliberately not implemented yet so this
single node can remain deep rather than the system as a whole remaining shallow.

```
         ┌─ ★ Node 1: Pre-Merge Risk Workflow  ← THIS REPO (shipped)
         │
SDLC ────┼─   Node 2: Build Failure Triage     ← roadmap (NVIDIA MTTR lever)
Workflow │    Node 3: Smart Test Selection     ← roadmap
Agent    │    Node 4: Release Readiness        ← roadmap
         │    Node 5: Cross-Team Dependency    ← roadmap (HW-SW codesign)
         │
         └─ shared: Orchestrator · Heterogeneous RAG (A/B/C) · Model Gateway
                    FeedbackLog · Audit Store · Action Surface · DORA loop
```

Full vision, per-node interfaces, NVIDIA-IPP mapping, and prioritization
in [`docs/agentic-sdlc-architecture.md`](docs/agentic-sdlc-architecture.md).

## Architecture (in brief)

```
                git diff + PR metadata (+ codeowners, history)
                          |
                          v
              +----------------------------+
              | Multi-Agent Harness        |
              |   (Claude Agent SDK)       |
              |     - diff-analyzer        |  <- structural shape of the diff
              |     - ownership-mapper     |  <- reviewers + bus-factor risk
              |     - agent-pr-auditor     |  <- detects AI-authored PRs + agent-specific risk
              |     - test-impact-scout    |  <- which tests cover the change
              |     - historical-context   |  <- RAG over similar past PRs
              +-------------+--------------+
                            |
                            v
              +----------------------------+
              | Multi-Vendor Model Gateway |
              |     - Claude (judge)       |
              |     - NVIDIA NIM           |
              |     - Triton-served NeMo   |
              |     - Azure OpenAI         |
              +-------------+--------------+
                            |
                            v
              +----------------------------+
              | Policy Gatekeeper          |  <- score -> action (4 bands)
              |   + Explanation Writer     |  <- markdown PR comment
              +-------------+--------------+
                            |
                            v
                Risk Score + Reasoning + Action
                + Audit Log (Mongo/MySQL/ES)
                + DORA Impact Dashboard
```

## Tech stack

- **Agent harness**: Claude Agent SDK, MCP tool federation (NVIDIA-native alternative: AIQ Toolkit + NeMo Retriever — see [`docs/design-doc.md`](docs/design-doc.md))
- **Fine-tuning**: NVIDIA NeMo + LoRA (Mistral-7B-v0.3 base)
- **Classical-ML baseline**: NVIDIA RAPIDS cuML (GBDT on engineered features; sklearn CPU fallback for dev)
- **Inference optimization**: NVIDIA TensorRT-LLM (engine compilation for the Mistral adapter)
- **Serving**: NVIDIA Triton Inference Server, NVIDIA NIM
- **Evaluation**: pytest, NVIDIA Garak (red-teaming)
- **Safety**: NVIDIA NeMo Guardrails
- **Backend**: Python, FastAPI
- **Dashboard**: Streamlit
- **Storage**: MongoDB / MySQL / Elasticsearch (multi-backend audit log + RAG index — see [`src/storage/audit_store.py`](src/storage/audit_store.py))
- **CI**: GitHub Actions (eval-gated deploys)

## Try it locally

```bash
git clone https://github.com/mingdongt/commit-risk-scorer
cd commit-risk-scorer
pip install -e .                  # installs deps declared in pyproject.toml

# Library demo — full sub-agents -> policy -> markdown PR comment pipeline
python -m src.agent.harness

# HTTP service — same pipeline behind a FastAPI endpoint
uvicorn src.serving.api:app --port 8000
#   GET  /health
#   POST /score  { "diff": "...", "metadata": { "codeowners": {...} } }

# DORA impact dashboard — Streamlit (v0.1 simulated data; v0.2 reads audit-store)
streamlit run src/metrics/dora_dashboard.py

# Build the GitHub PR / CI training dataset (requires $GITHUB_TOKEN for useful volume)
python -m src.data.scrape_github_prs --repos kubernetes/kubernetes django/django \
    --max-prs-per-repo 100 --output data/raw/github_prs.jsonl

# Tests — 86 across agent / policy / explainer / API / red-team / audit-store /
#         scraper / dashboard / fine-tune
pytest tests/
```

## Repository structure

```
.
├── README.md                       <- you are here
├── LICENSE                          <- Apache 2.0
├── docs/
│   ├── design-doc.md                <- motivation, prior art, architecture
│   ├── onboarding.md                <- adoption guide for teams using this
│   ├── runbook.md                   <- what to do when the agent misfires
│   ├── postmortem-template.md
│   └── metrics.md                   <- DORA metric definitions
├── src/
│   ├── agent/                       <- Claude Agent SDK harness
│   ├── models/                      <- model gateway, NeMo fine-tune
│   ├── eval/                        <- pytest eval suite, Garak probes
│   ├── serving/                     <- FastAPI + Triton client
│   └── metrics/                     <- DORA dashboard
├── tests/                           <- pytest suite (regression-gated CI)
├── data/                            <- labeled commits (gitignored)
├── notebooks/                       <- exploration, baselines, fine-tune logs
└── .github/workflows/               <- GitHub Actions: eval.yml runs pytest on push + PR
```

## Initial Results — pipeline validation across baselines

Two pipelines have been validated end-to-end on subsamples of CodeXGLUE Devign. The point of this section is *pipeline validation* and *trade-off surfacing*, not capability claims — see *Production target* below for the meaningful comparison.

| Metric | DistilBERT + LoRA (HF PEFT smoke) | cuML GBDT baseline (sklearn-fallback) |
| --- | --- | --- |
| F1 | 0.383 | **0.436** |
| Precision | 0.368 | **0.494** |
| Recall | **0.398** | 0.390 |
| Accuracy | 0.473 | **0.570** |
| AUC-ROC | 0.466 | **0.545** |

- DistilBERT LoRA: 300 examples/split, 1 epoch, ~740 K trainable params (rank-8). CPU only.
- GBDT baseline: 500 examples/split, 10 engineered features (LOC, alloc/free, pointer arithmetic, branch/loop counts, etc.). CPU sklearn fallback (the script auto-detects RAPIDS cuML on a CUDA box).

**Findings**:

- On this subsample size, **the engineered-feature GBDT beats the LoRA-tuned DistilBERT** on F1, precision, accuracy, and AUC-ROC. The baseline existing isn't a defect — it's the point: simple features go a long way on small data, and DistilBERT is the wrong base (not pre-trained on code; ~300 samples insufficient).
- DistilBERT AUC-ROC at 0.466 is below random; GBDT at 0.545 is the first sign of real discriminative signal.
- **Implication for production**: don't use DistilBERT. The Mistral-7B-v0.3 path via NeMo (full dataset + code-pretrained base) is the right next step. The GBDT remains useful as a fast classifier ensemble component.

Raw metrics: [`data/models/smoke/smoke_metrics.json`](data/models/smoke/smoke_metrics.json) · [`data/models/baselines/cuml_gbdt_metrics.json`](data/models/baselines/cuml_gbdt_metrics.json).

**Production target**: NVIDIA NeMo + LoRA + Mistral-7B-v0.3 on full Devign + ~1 k self-labeled GitHub PR/CI scrapes, followed by **TensorRT-LLM engine compilation** for Triton serving. Code in [`src/models/finetune/train_nemo.py`](src/models/finetune/train_nemo.py); pending CUDA environment + base-model conversion.

## Status

**Active development.** Public technical artifact. See [`docs/design-doc.md`](docs/design-doc.md) for current scope and open questions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

License intentionally aligned with NVIDIA's open-source AI ecosystem (NeMo, Triton, Garak, NeMo Guardrails) for ecosystem coherence and contributor friendliness.

## Author

Built by **Mingdong (Eric) Tan**.
[github.com/mingdongt](https://github.com/mingdongt) · [linkedin.com/in/mingdongt](https://linkedin.com/in/mingdongt) · mingdongtan6@gmail.com
