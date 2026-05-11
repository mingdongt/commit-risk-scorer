# Evaluation: Three Layers, Not One

A common failure mode of LLM-agent projects is to report a single offline metric
(F1 on a holdout) and call the work done. That tells you the model has learned
*something* — it doesn't tell you whether the agent is grounded, whether the
action is useful, or whether engineers' lives got better.

This document defines the **three eval layers** the project commits to, and how
each one feeds the regression-gated CI described in
[`design-doc.md`](design-doc.md) §Eval Methodology.

---

## Layer 1 — Model evaluation (offline, automated)

Measures **whether the classifier learned signal from data.** Runs on every push.

### Dataset

- **Training**: CodeXGLUE Devign + self-labeled GitHub PR/CI outcomes (see
  [`onboarding.md`](onboarding.md) for how teams produce their own slice).
- **Holdout**: 20 % of labeled commits, stratified by repo and outcome.

### Metrics

| Metric | Why it matters |
| --- | --- |
| **F1 (binary)** | Headline number — balances precision and recall. |
| **PR-AUC** | More informative than ROC-AUC under class imbalance (defective PRs are rare). |
| **Precision @ top-10 %** | How clean is the "must review" bucket the policy gate exposes? |
| **Calibration error (ECE)** | A risk score of 0.8 should mean ~80 % failure rate, not just "high". Without calibration, the policy bands (Low / Medium / High / Critical) are meaningless. |
| **AUC-ROC** | Standard reference; reported for completeness. |

### Gates

- F1 drop > 2 pp from previous main → CI red, merge blocked.
- ECE > 0.10 → CI red.

Implementation: [`src/eval/metrics.py`](../src/eval/metrics.py); HF Trainer
`compute_metrics` callback.

---

## Layer 2 — Agent evaluation (semi-automated, sampled)

Measures **whether the LLM-judge layer is producing trustworthy reasoning.** A
high-F1 classifier with a hallucinating judge is worse than no judge at all.

### What we check

| Dimension | Definition | Measurement |
| --- | --- | --- |
| **Evidence groundedness** | Every claim in `evidence[]` cites a real file / PR / test. | Programmatic: each citation is resolvable in the repo / PR index. |
| **Reasoning faithfulness** | The `topRiskFactors[]` are derivable from the `evidence[]`. | Manual sampling: 50 PRs/week, 2 reviewers, Cohen's κ ≥ 0.6 inter-rater. |
| **Action usefulness** | Engineers acted on the recommendation. | Click-through / accept rate on the recommended action. |
| **Refusal correctness** | When the diff is hostile (prompt injection), the judge declines. | Garak red-team probe suite (see [`../src/eval/red_team.py`](../src/eval/red_team.py)). |

### Gates

- Garak probe-failure rate > 10 % overall, or > the per-probe `max_failure_rate`,
  → CI red.
- Manual-sample faithfulness < 80 % → escalate to design review; no merge gate
  (faithfulness is too slow to gate per-commit).

---

## Layer 3 — Business evaluation (online, slow loop)

Measures **whether the agent moves the DORA-style needles the platform team is
on the hook for.** This is the only eval that ultimately matters; the first two
are necessary but not sufficient.

### Metrics (defined in [`metrics.md`](metrics.md))

| Metric | Direction | Source |
| --- | --- | --- |
| **Cycle time** | ↓ (agent shouldn't add net delay) | merged_at − first_commit_at |
| **Change failure rate** | ↓ | merges followed by red CI / revert / incident within 7 d |
| **MTTR** | ↓ or flat | green_at − failed_at, median |
| **Adoption rate** | ↑ | fraction of PRs whose authors engaged with the agent output |
| **False-positive feedback** | ↓ | author-submitted "this flag was wrong" signal |

### Cadence

- Weekly snapshot per repo, rendered on the DORA dashboard
  ([`../src/metrics/`](../src/metrics/) — Streamlit, v0.3).
- Quarterly review with the adopting team; threshold re-tune if FP feedback rate
  climbs above 15 %.

### Honesty note

Some impact estimates referenced in this repo (e.g., *"~5–8 hours/week of manual
review-and-triage toil saved"*) are derived from **action-count × per-action
time savings**, not from a randomized A/B trial. The methodology is documented
in [`metrics.md`](metrics.md) — published numbers are the lower bound of the
estimation range, not the midpoint.

---

## Why three layers, not one

The three layers gate different failure modes:

| Failure mode | Caught by |
| --- | --- |
| Classifier is undertrained | Layer 1 — F1 / PR-AUC drop |
| Classifier overconfident | Layer 1 — ECE blows up |
| Judge hallucinates evidence | Layer 2 — groundedness check |
| Judge jailbroken by malicious diff | Layer 2 — Garak probe |
| Score is fine but engineers ignore it | Layer 3 — adoption / FP feedback |
| Engineers act on it but DORA stays flat | Layer 3 — cycle time / CFR delta |

A regression in any of the three is a regression. The CI gates the fast layers
(1, 2); the slow layer (3) is monitored, not blocked-on.

---

*Last updated: 2026-05-10*
