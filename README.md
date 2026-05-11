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

## Why this exists

Existing solutions occupy one corner of the design space:

| Tool | Approach | Limitation |
| --- | --- | --- |
| [PR-Agent](https://github.com/Codium-ai/pr-agent) (Codium) | Generative LLM review | No predictive scoring or calibration |
| CodeBERT / [Devign](https://github.com/microsoft/CodeXGLUE) | Trained classifier | No reasoning or agent integration |
| [NVIDIA Garak](https://github.com/NVIDIA/garak) | LLM red-teaming | Not specialized for code-review agents |

This project is the integration that unifies all three. Internal big-tech equivalents (Google Tricorder, Meta Sapienz, Microsoft TestImpact, Amazon CodeGuru) exist at scale but are closed-source — this project bridges that open-source gap. See [`docs/design-doc.md`](docs/design-doc.md) for full motivation, prior art comparison, architectural trade-offs, and honest caveats.

## Architecture (in brief)

```
                git diff + PR metadata
                          |
                          v
              +----------------------------+
              | Multi-Agent Harness        |
              |   (Claude Agent SDK)       |
              |     - diff-analyzer        |
              |     - test-impact-scout    |
              |     - historical-context   |
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
                Risk Score + Explanation
                + DORA Impact Dashboard
```

## Tech stack

- **Agent harness**: Claude Agent SDK, MCP tool federation
- **Fine-tuning**: NVIDIA NeMo + LoRA (Mistral-7B-v0.3 base)
- **Serving**: NVIDIA Triton Inference Server, NVIDIA NIM
- **Evaluation**: pytest, NVIDIA Garak (red-teaming)
- **Safety**: NVIDIA NeMo Guardrails
- **Backend**: Python, FastAPI
- **Dashboard**: Streamlit
- **Index**: Elasticsearch
- **CI**: GitHub Actions (eval-gated deploys)

## Try it locally

```bash
git clone https://github.com/mingdongt/commit-risk-scorer
cd commit-risk-scorer
pip install -e .                  # installs deps declared in pyproject.toml
python -m src.agent.harness       # runs the harness demo on a hand-crafted diff
pytest tests/                     # 9 unit tests, ~70 ms
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

## Initial Results — smoke-test pipeline validation

The first end-to-end run validates the **data load → tokenize → LoRA → train → eval** pipeline on a CPU laptop. This is **pipeline validation, not a capability benchmark** — see *Production target* below for the meaningful comparison.

**Setup**: HuggingFace PEFT + LoRA on DistilBERT-base (~67 M params), 300-example subsample per split of CodeXGLUE Devign, 1 epoch, batch size 8, CPU only.
**Trainable parameters**: 739,586 (1.1 % of base model thanks to LoRA rank-8 adapter).

| Metric | Test split |
| --- | --- |
| F1 | 0.3828 |
| Precision | 0.3684 |
| Recall | 0.3984 |
| Accuracy | 0.4733 |
| AUC-ROC | 0.466 |

**Interpretation**: F1 below 0.5 is expected for this minimal smoke run — 300 training samples, single epoch, and a base model not pre-trained on code is not enough signal to learn the task. The point of this run is to confirm every step of the pipeline executes correctly and emits valid metrics, not to claim capability. Raw metrics: [`data/models/smoke/smoke_metrics.json`](data/models/smoke/smoke_metrics.json).

**Production target**: NVIDIA NeMo + LoRA + Mistral-7B-v0.3 on full Devign + ~1 k self-labeled GitHub PR/CI scrapes. Code in [`src/models/finetune/train_nemo.py`](src/models/finetune/train_nemo.py); pending CUDA environment + base-model conversion.

## Status

**Active development.** Public technical artifact. See [`docs/design-doc.md`](docs/design-doc.md) for current scope and open questions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

License intentionally aligned with NVIDIA's open-source AI ecosystem (NeMo, Triton, Garak, NeMo Guardrails) for ecosystem coherence and contributor friendliness.

## Author

Built by **Mingdong (Eric) Tan**.
[github.com/mingdongt](https://github.com/mingdongt) · [linkedin.com/in/mingdongt](https://linkedin.com/in/mingdongt) · mingdongtan6@gmail.com
