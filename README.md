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
├── data/                            <- labeled commits (gitignored)
├── notebooks/                       <- exploration, baselines, fine-tune logs
└── ci/                              <- GitHub Actions workflows
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
