# Honest Limitations

This document collects everything that *won't* work, *might not* work, or *works less well than the marketing makes it sound*. Read it before adopting, investing in, or judging this project.

> The single best signal an engineering project is honest is that it has a public, prominent limitations document. This is that document.

---

## Data limitations

### 1. Public CI outcome is noisy ground truth

CI failures conflate real signal (bad commits) with noise:

- **Flaky tests** — intermittent failures unrelated to the commit.
- **Infrastructure failures** — network, vendor outages, runner issues.
- **Upstream dependency churn** — third-party packages breaking unrelated to the change.
- **Test bugs** — the test was wrong, not the code.

Internal big-tech tools have access to richer ground truth (revert history, incident attribution, author-team telemetry, fix-commit linkage). This project's accuracy is upper-bounded by the noise in *public* CI signal.

**Mitigation**: pre-train on CodeXGLUE Devign (cleaner labels — manually curated security defects); fine-tune target is public CI outcome with explicit confidence intervals.

### 2. Class imbalance

Most commits pass CI. Naive accuracy is misleading on a 7:3 or 8:2 split. The training pipeline must use class weighting or focal loss; the eval suite reports F1 / precision / recall and per-class confusion, not accuracy.

### 3. Sampling bias

The scraped PR dataset overweights high-activity repos (Kubernetes, Django, PyTorch). Risk patterns from low-traffic codebases (legacy enterprise, embedded systems, internal tools) may be under-represented. Per-org calibration on the adopter's own historical PRs is recommended — see [`onboarding.md`](onboarding.md) §*Integration steps*.

### 4. Temporal drift

A model trained on 2024 PRs may underperform on 2026 patterns (new languages, new frameworks, new failure modes). Periodic retraining is required — cadence and trigger conditions documented in [`runbook.md`](runbook.md).

---

## Model limitations

### 5. The classifier doesn't reason

The fine-tuned Mistral classifier outputs a single number. When wrong, it can't explain why. The LLM judge layer compensates but at 5–10× inference cost and ~1–2 s additional latency per PR.

### 6. The judge can be wrong with high confidence

Frontier LLMs are well-calibrated *on average* but produce confidently-wrong reasoning in adversarial cases. NeMo Guardrails + Garak red-team catch *some* failure modes — prompt injection, output-format violations, fabricated file paths — but not all. Confidently-wrong-but-fluent output remains the hardest LLM failure to detect automatically.

### 7. RAG retrieval is similarity-based, not causal

"Similar past PR was reverted" is correlation, not causation. The judge prompt explicitly instructs it to avoid concluding *"this PR will be reverted"* from *"PR-#1842 was reverted"* alone, but the model can still anchor on the retrieved evidence. Watch for over-confident reasoning when the retrieved context is dominated by negative outcomes.

---

## Evaluation limitations

### 8. F1 on a held-out subset ≠ production accuracy

Held-out PRs come from the same distribution as training. Production PRs from a new team may be out-of-distribution. The reported F1 is a useful *upper bound on what you should expect* in your own environment, not a guarantee.

### 9. The DORA dashboard uses simulated data

The DORA impact dashboard in this repo currently ingests **simulated** cycle-time / MTTR / change-failure-rate / adoption data. Real impact numbers require actual production deployment in a real organization — out of scope for this open-source artifact. The dashboard demonstrates the *measurement shape*, not validated impact.

### 10. No causal inference

The dashboard shows correlation between "agent flagged" and "PR outcome", not causation. Real ROI claims require an A/B experiment in a real org — agent-flagged PRs vs control-group PRs, controlled for confounders.

---

## Deployment limitations

### 11. CUDA dependency for production fine-tune

NeMo + LoRA on Mistral-7B-v0.3 requires CUDA. Production training is not feasible on CPU. The smoke-test path (HuggingFace PEFT on DistilBERT) is provided for CPU-only development environments — see [`src/models/finetune/train_smoke.py`](../src/models/finetune/train_smoke.py).

### 12. NIM / Triton are NVIDIA-stack

Adopters on AWS-only or GCP-only infrastructure may need to substitute TGI / vLLM / SageMaker endpoints. The multi-vendor gateway design ([`src/models/gateway.py`](../src/models/gateway.py)) intentionally abstracts this — add a new backend by implementing the `ModelBackend` protocol.

### 13. Real-time inference cost

A hybrid pipeline (classifier + judge + RAG retrieval) on every PR has non-trivial per-PR cost. Budget estimate at the production target:

| Component | Cost per PR |
| --- | --- |
| Classifier (Triton + Mistral) | ~$0.001 (amortized) |
| LLM judge (Claude or NIM) | ~$0.01–0.04 |
| RAG retrieval (Elasticsearch) | ~$0.0001 |
| **Total** | **~$0.01–0.05** |

At 1k PRs/day, that's ~$300–1500/month. Significant but defensible if it catches a single production incident.

---

## Comparison to internal big-tech tools

| Internal tool (e.g. MS TestImpact, Google Tricorder) | This project |
| --- | --- |
| Trained on years of internal CI + revert + incident data | Trained on public PR data + Devign |
| Tuned per-repo by SREs | Generic; per-org calibration recommended |
| Operates inside corporate CI infrastructure | Operates as a CI plugin |
| Closed source, paid talent maintains | Apache 2.0, individual maintainer |

This project demonstrates **architectural viability** with public data and the NVIDIA open-source stack — not state-of-the-art accuracy.

---

## What's *not* a limitation

A few things this project has been asked about that *are* solid:

- **The architecture pattern** (hybrid classifier + LLM judge + RAG) is well-supported by literature (DeepJIT, CC2Vec, JITLine, JIT-QA benchmarks).
- **The 4-quadrant differentiation analysis** ([README](../README.md) §*Why this exists*) is accurate — adjacent tools verified by paper / public docs.
- **The 86-test green build, 3-scenario demo, and platform-team documentation suite** reflect real engineering discipline, not marketing surface area.
- **The Apache 2.0 license alignment** with the NVIDIA open-source AI stack is deliberate and well-formed.

If you can verify the above and still find them lacking, please [file an issue](https://github.com/mingdongt/commit-risk-scorer/issues).

---

*Last updated: 2026-05-10*
