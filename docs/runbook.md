# Runbook: When commit-risk-scorer Misfires

This runbook covers operational responses to the most likely failure modes. Triage in
order of user impact — false negatives that ship broken code outrank false positives
that annoy authors.

---

## Severity levels

| Sev | Definition | Response time |
| --- | --- | --- |
| **SEV-1** | Agent silently passes a commit that breaks production | Page oncall; full block until root-caused |
| **SEV-2** | Agent blocks > 5 PRs/day that should not have been blocked | Disable for the affected repo within 24 h |
| **SEV-3** | Single-PR mistakes (false positives or negatives) | Log via the failure-feedback button; review weekly |

---

## Failure mode 1: Risk score consistently too low (false negatives)

**Symptom**: PRs that broke CI are being scored below 0.3.

**Common causes**:

1. Calibration drift — your codebase has shifted; thresholds need re-tuning.
2. Underrepresented diff patterns (new framework adoption, new file types).
3. Stale historical RAG index.

**Action**:

1. Pull the last 2 weeks of incident PRs; compute their predicted scores.
2. If median predicted score for incident PRs is below 0.5 → re-run `calibrate`.
3. Refresh the RAG index: `commit-risk-scorer reindex --repo your-repo`.

---

## Failure mode 2: Risk score consistently too high (false positives)

**Symptom**: PRs that pass CI are being scored above 0.7. Authors are annoyed.

**Common causes**:

1. Threshold set too aggressively low.
2. Class imbalance during calibration (defective examples over-represented).
3. Author-pattern bias (new-contributor PRs scored higher even when fine).

**Action**:

1. Inspect calibration set composition: `commit-risk-scorer inspect-calibration`.
2. Re-balance to match your repo's defect base rate.
3. Raise the threshold; re-deploy.

---

## Failure mode 3: Agent crashes / latency spikes

**Symptom**: PR check times out, or agent returns 5xx.

**Common causes**:

1. LLM judge backend (Claude / NIM) rate-limited.
2. Triton inference server out of VRAM.
3. RAG index unavailable.

**Action**:

1. Check backend health: `commit-risk-scorer health`.
2. The gateway should auto-fall-back; verify in logs.
3. If sustained, switch to FT-classifier-only mode (lighter, no judge).

---

## Failure mode 4: Garak red-team probe regression

**Symptom**: CI eval gate fails with `Garak failure rate > 10%`.

**Common causes**:

1. Recent prompt change introduced a jailbreak / injection vector.
2. Updated Garak probe suite caught a previously-missed weakness.

**Action**:

1. Inspect failed probes in the CI artifact.
2. If a new probe → triage as a known finding; document.
3. If a prompt change → revert and re-test.

---

## Postmortem trigger

Any SEV-1 or repeated SEV-2 → write a postmortem using
[`postmortem-template.md`](postmortem-template.md). Link the postmortem back into
this runbook's relevant failure-mode entry.

---

*Status: skeleton — the failure modes above are derived from analogous internal tools at Microsoft. Real entries land as the agent is exercised against live data in v0.2+.*
