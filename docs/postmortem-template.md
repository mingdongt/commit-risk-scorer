# Postmortem: [Incident title]

> **Status**: draft | review | resolved
> **Severity**: SEV-1 | SEV-2 | SEV-3
> **Date of incident**: YYYY-MM-DD
> **Date of postmortem**: YYYY-MM-DD
> **Authors**: [names]
> **Reviewers**: [names]

---

## TL;DR

What happened, what was the user impact, and what's the one-line lesson — three sentences max.

---

## Timeline (UTC)

| Time | Event |
| --- | --- |
| HH:MM | Incident begins (silent failure / first user complaint / first alert) |
| HH:MM | Detection — what surfaced it (alert / report / monitoring)? |
| HH:MM | First action — who took it, what did they try? |
| HH:MM | Mitigation — what stopped the bleeding? |
| HH:MM | Resolution — what fully fixed it? |

---

## What went well

- [Things to keep doing — alerting that fired, instincts that paid off]

## What went wrong

- [Root causes — be specific, e.g., *"Silent data loss because the Cosmos vector-batch throttling cliff at ~55 docs was never characterized."*]
- [Contributing factors — alerts that should have fired but didn't, missing runbook entries, etc.]

## Where we got lucky

- [Things that could have been worse but weren't — useful for understanding latent risks]

---

## Root cause analysis

Trace the failure to its root. Use the "5 whys" or a similar technique. The surface
symptom is rarely the root.

---

## Action items

| Action | Owner | Due | Tracking |
| --- | --- | --- | --- |
| Fix the root cause | @who | YYYY-MM-DD | issue link |
| Add an alert for the leading indicator | @who | YYYY-MM-DD | issue link |
| Update [`runbook.md`](runbook.md) failure-mode N | @who | YYYY-MM-DD | PR link |

---

## Lessons (for future incidents and design reviews)

- [What did we learn that should change how we design or operate?]

---

*Adapted from the standard SRE postmortem format. Keep it blameless: focus on systems and processes, not individuals.*
