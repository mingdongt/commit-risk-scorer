# Metrics: DORA-Style Definitions and Measurement Methodology

The commit-risk-scorer's success is measured in *engineering productivity outcomes*,
not just classifier F1. This document defines each metric, how we measure it, and
the methodology behind the impact estimates referenced in the README and
[`onboarding.md`](onboarding.md).

---

## Cycle Time

**Definition**: Wall-clock time from a PR's first commit to its merge.

**Measurement**: For each merged PR, record `merged_at - first_commit_at` in hours.

**Aggregation**: Median per week, segmented by repository.

**Why it matters**: A useful agent should *reduce* cycle time, not increase it. If
flagging risky PRs adds 6 hours of median delay but reduces incidents by less than
10 %, the trade is negative.

---

## Change Failure Rate (CFR)

**Definition**: Fraction of merges that result in any of:

- A failed CI run on `main` within 24 hours of merge
- A revert PR within 7 days
- An incident report linked back to that PR within 7 days

**Measurement**:

- Numerator: count of merges meeting any of the above criteria.
- Denominator: total merges in the time window.

**Aggregation**: Weekly rate per repo.

**Why it matters**: This is the metric the agent most directly targets. Pre-merge
risk scoring should reduce CFR by intercepting risky changes before they merge.

---

## Mean Time To Recovery (MTTR)

**Definition**: For a CI failure on `main`, time elapsed until a green build returns.

**Measurement**: `green_at - failed_at` in hours, taken as the median over the window.

**Aggregation**: Weekly median per repo.

**Why it matters**: The agent reduces MTTR indirectly — flagging high-risk changes
earlier triggers more careful review, which produces faster root-cause identification
when something does fail.

---

## Adoption Rate

**Definition**: Fraction of PRs whose authors interacted with the agent's output
(read the comment, requested re-evaluation, clicked the "feedback" link, etc.).

**Measurement**: Per-PR interaction events emitted by the GitHub Action and
aggregated weekly per team.

**Why it matters**: A tool nobody uses is a tool that doesn't matter. Adoption is
the leading indicator of whether the agent's signal is credible to engineers.

---

## Methodology notes

### Estimation honesty

Some impact estimates referenced in the README and Eric's resume — e.g.,
*"~5–8 hours/week of manual review-and-triage toil saved"* — are derived from
**action-count × per-action time savings**, not from a randomized A/B trial. The
methodology:

- **Per-action savings** are estimated conservatively (e.g., 1.5 minutes per
  auto-assigned PR — typical Slack-thread length to find a human reviewer).
- **Volume** is observed directly from the agent's logs.
- **Total estimate** = volume × per-action savings, with the published number taken
  from the lower bound of the resulting range to favor defensibility under
  questioning.

This is the standard methodology used in internal platform-team ROI reports. We
document it transparently here rather than presenting numbers as if they came from
controlled trials.

### What we explicitly do *not* measure (yet)

- **Counterfactual cycle time** — what the cycle time would have been without the agent.
- **Reviewer cognitive load** — proxy metrics here are easy to mis-aggregate.
- **Author satisfaction** — survey data only justified at meaningful adoption scale.

These appear in the v1 dashboard if and when adoption data justifies the
instrumentation investment.

---

*Last updated: 2026-05-10*
