# commit-risk-scorer

> **Open-source predictive code-review agent** — bridges generative LLM review (PR-Agent style) and trained predictive defect models (CodeBERT family) into a hybrid pipeline served on NVIDIA's open-source AI stack.

## What it does

Given a git commit or pull request, produces:

1. **Risk score** (0–100%) — probability the change will cause CI failure or need revert
2. **Reasoning** — natural-language explanation grounded in similar historical PRs
3. **Recommended action** — reviewer assignment, extended test suite, manual review gate

Designed to run as a CI check on every PR — providing **pre-merge predictive signal** that complements (not replaces) traditional CI.

## Why this exists

Existing solutions occupy one corner of the design space:

| Tool | Approach | Limitation |
| --- | --- | --- |
| [PR-Agent](https://github.com/Codium-ai/pr-agent) (Codium) | Generative LLM review | No predictive scoring or calibration |
| CodeBERT / [Devign](https://github.com/microsoft/CodeXGLUE) | Trained classifier | No reasoning or agent integration |
| [NVIDIA Garak](https://github.com/NVIDIA/garak) | LLM red-teaming | Not specialized for code-review agents |

This project is the integration that unifies all three. See [`docs/design-doc.md`](docs/design-doc.md) for full motivation, prior art comparison, and architectural trade-offs.

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
├── data/                            <- labeled commits (gitignored)
├── notebooks/                       <- exploration, baselines, fine-tune logs
└── ci/                              <- GitHub Actions workflows
```

## Status

**Active development.** Public technical artifact. See [`docs/design-doc.md`](docs/design-doc.md) for current scope and open questions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

License intentionally aligned with NVIDIA's open-source AI ecosystem (NeMo, Triton, Garak, NeMo Guardrails) for ecosystem coherence and contributor friendliness.

## Author

Built by **Mingdong (Eric) Tan**.
[github.com/mingdongt](https://github.com/mingdongt) · [linkedin.com/in/mingdongt](https://linkedin.com/in/mingdongt) · mingdongtan6@gmail.com
