# Onboarding Guide: Adopting commit-risk-scorer in Your Team's CI

## Who this is for

Engineering teams (any org running GitHub + a CI system) that want to:

- Reduce CI failure rate by **flagging high-risk commits pre-merge**
- Cut down reviewer-routing toil with **predictive risk scores + reasoning**
- Get a calibrated, trained signal — not just generative LLM critiques

## Prerequisites

- GitHub repo with PR-based workflow
- GitHub Actions, GitLab CI, or Jenkins (any CI supporting check-runs)
- ~500 historical PRs with CI outcomes (used to calibrate to your codebase)
- (Optional) Anthropic API key — for the LLM judge layer
- (Optional) NVIDIA NIM API key — for hosted fine-tune inference

## Integration steps

```bash
# [1] Install
pip install commit-risk-scorer

# [2] Calibrate on your repo
commit-risk-scorer ingest --repo your-org/your-repo --max-prs 500
commit-risk-scorer calibrate --output ./calibration.json

# [3] Add to CI
# See ci/.github/workflows/example.yml for a drop-in template.

# [4] Tune the risk threshold
# Edit calibration.json — set the risk threshold above which the agent
# escalates a PR for extended review.
```

## Rollout phases (recommended)

A four-phase rollout limits blast radius. Move forward only when each phase's exit
criterion is met.

### Phase 1 — Shadow mode (Week 1–2)

- Agent predicts but does **not** block merges.
- Comments are posted but marked `[shadow — not blocking]`.
- **Exit**: ~50 predictions logged; manual validation confirms signal direction.
- **Rollback**: disable the workflow file.

### Phase 2 — 10% canary (Week 3–4)

- Agent flags PRs from a 10% bucket of contributors (deterministic hash bucketing).
- Flagged PRs require an additional reviewer ack.
- **Exit**: < 20% false-positive rate observed; threshold tuned.
- **Rollback**: reduce bucket back to 0%.

### Phase 3 — 50% rollout (Week 5)

- Expand the bucket to 50%.
- Watch for adoption complaints, latency regressions.

### Phase 4 — GA (Week 6+)

- 100% of PRs scored.
- Owner monitors the weekly DORA dashboard (see [`metrics.md`](metrics.md)).

## Expected impact

Estimates based on the Devign baseline plus preliminary self-labeled GitHub PR
data. Your mileage will vary by codebase shape and CI flakiness.

| Metric | Expected delta |
| --- | --- |
| CI-failed-PR rate | -25% to -40% |
| Reviewer-time / PR | -3 to -5 hours/week per 10 engineers |
| MTTR for CI failures | unchanged to -10% |
| Adoption | track in [`metrics.md`](metrics.md) |

## Rollback plan

If the agent makes too many wrong calls or causes friction:

1. Set `commit-risk-scorer.threshold = 1.0` in your CI config (effective off-switch).
2. The agent continues running silently (still collecting data) but does not block.
3. Re-tune from the calibration step.

## Getting help

File an issue: <https://github.com/mingdongt/commit-risk-scorer/issues>

---

*Status: this guide describes the target adopter experience. CLI commands shown above are illustrative; the actual CLI ships in v0.2.*
