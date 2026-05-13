# Agentic SDLC System — Architecture Vision

> Status: **Draft v0.1** — last updated 2026-05-13.
>
> This document positions `commit-risk-scorer` inside the broader **Agentic
> SDLC System** it is the first node of. It is a vision document, not a
> shipped-feature document: only Node #1 (Pre-Merge Risk Workflow) is
> implemented in this repo at v0.1. The remaining four nodes are scoped here
> as **roadmap with concrete interfaces**, deliberately not implemented yet
> so the shipped node can remain over-built rather than the system as a whole
> remaining under-built.

---

## Table of Contents

1. [TL;DR — Where this fits](#tldr--where-this-fits)
2. [The 5-Agent SDLC System](#the-5-agent-sdlc-system)
3. [Node 1 — Pre-Merge Risk Workflow (Shipped)](#node-1--pre-merge-risk-workflow-shipped)
4. [Node 2 — Build Failure Triage Workflow (Roadmap)](#node-2--build-failure-triage-workflow-roadmap)
5. [Node 3 — Smart Test Selection Workflow (Roadmap)](#node-3--smart-test-selection-workflow-roadmap)
6. [Node 4 — Release Readiness Workflow (Roadmap)](#node-4--release-readiness-workflow-roadmap)
7. [Node 5 — Cross-Team Dependency Tracker (Roadmap)](#node-5--cross-team-dependency-tracker-roadmap)
8. [Shared Infrastructure](#shared-infrastructure)
9. [Why Not Build All Five At Once](#why-not-build-all-five-at-once)
10. [Mapping to NVIDIA IPP — JD Bullet-by-Bullet](#mapping-to-nvidia-ipp--jd-bullet-by-bullet)
11. [Prioritization — If I Had One More Week](#prioritization--if-i-had-one-more-week)

---

## TL;DR — Where this fits

`commit-risk-scorer` (the rest of this repo) is **Node #1** of an Agentic SDLC
System — a multi-agent platform that uses LLMs + agentic AI to automate
end-to-end software-engineering workflows and measure their impact in DORA terms.

- **Node #1 (this repo)**: Pre-Merge Risk Workflow — deep, production-shaped,
  over-built. Five sub-agents, three-tier cascading router (GBDT → fine-tuned
  Mistral → Claude-judge + heterogeneous RAG), policy gatekeeper, multi-vendor
  model gateway, eval-gated CI, NVIDIA Garak red-team probes, NeMo Guardrails
  for output safety, DORA-aligned eval harness, 86 tests.
- **Nodes #2–#5**: scoped here as roadmap with interfaces. Each addresses a
  distinct part of the SDLC where LLM/agent intervention produces measurable
  DORA-metric improvement — and each maps to a different NVIDIA IPP pain point.

The architectural claim is **depth-first**: ship one node to production-shaped
maturity before scaffolding the next. The five-node framing is the system
target; the shipped node is the proof-of-execution.

---

## The 5-Agent SDLC System

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EVENT SURFACE                                                          │
│  PR opened │ Push │ Build done │ Nightly cron │ Release branch cut      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR — SDLCWorkflowAgent                                       │
│  Routes events to specialist workflows. Each workflow encapsulates one  │
│  shaped problem; the orchestrator does not embed business logic.        │
└──┬─────────┬─────────┬─────────┬─────────┬──────────────────────────────┘
   │         │         │         │         │
   ▼         ▼         ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│ ★    │ │      │ │      │ │      │ │          │
│Node 1│ │Node 2│ │Node 3│ │Node 4│ │  Node 5  │
│ Pre- │ │Build │ │Smart │ │Rele- │ │  Cross-  │
│Merge │ │ Fail │ │ Test │ │ ase  │ │   Team   │
│ Risk │ │Triage│ │Select│ │Ready-│ │   Dep    │
│      │ │      │ │      │ │ness  │ │  Tracker │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───────┘
   │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ACTION SURFACE                                                         │
│  PR comment │ Reviewer assign │ Block merge │ Skip CI config            │
│  Trigger downstream regression │ Slack/Teams notify │ Page on-call      │
│  Open JIRA │ Update release dashboard                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ retrieve / ground
                              │
┌─────────────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE LAYER — Heterogeneous RAG Gateway                            │
│  Layer A (code-adjacent, always-on)                                     │
│    past PRs · incidents · ADRs · CODEOWNERS · build-failure RCA         │
│  Layer B (operational metadata, triggered)                              │
│    release calendar · on-call · feature flags · SLA · re-org map        │
│  Layer C (high-stakes enterprise, LLM-judged)                           │
│    HW spec · ASIL safety case · customer RMA · errata · PRDs            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  METRICS LOOP — DORA-aligned                                            │
│  Node 1 → Change Failure Rate ↓                                         │
│  Node 2 → MTTR ↓                                                        │
│  Node 3 → Lead Time ↓ (CI cost ↓)                                       │
│  Node 4 → Deployment Frequency ↑                                        │
│  Node 5 → Cross-org incident ↓                                          │
│                                                                         │
│  All gated by: offline eval + shadow mode + A/B + counterfactual        │
└─────────────────────────────────────────────────────────────────────────┘
```

★ = shipped in this repo (Node #1). All other nodes are scoped below as roadmap.

---

## Node 1 — Pre-Merge Risk Workflow (Shipped)

**Status:** v0.1 production-shaped. Full design in
[`design-doc.md`](design-doc.md). Code at [`../src/agent/`](../src/agent/),
[`../src/models/`](../src/models/), [`../src/rag/`](../src/rag/),
[`../src/metrics/`](../src/metrics/).

**Trigger:** PR opened / push to PR branch.

**Output contract:**

```python
@dataclass
class PreMergeRiskResult:
    risk_score: float                 # 0.0–1.0
    risk_level: str                   # Low | Medium | High | Critical
    recommended_action: str           # fast_track | owner_review |
                                      # sme_review | block_merge
    top_risk_factors: list[str]       # evidence-backed
    pr_comment_markdown: str          # what gets posted on the PR
    evidence: list[Evidence]          # cited retrieved context
    confidence: float                 # 0.0–1.0
```

**DORA target:** Change Failure Rate ↓.

**Why it goes first:** The pre-merge gate is the highest-leverage SDLC point
for predictive intervention — risk caught here costs orders of magnitude less
than risk caught post-merge. It is also the most data-tractable on public
inputs (GitHub PR + CI outcomes), making the shipped node fully reproducible.

See [`design-doc.md`](design-doc.md) for full architecture; this section only
positions it within the broader SDLC system.

---

## Node 2 — Build Failure Triage Workflow (Roadmap)

**Status:** roadmap (highest priority among unshipped nodes — see
[Prioritization](#prioritization--if-i-had-one-more-week)).

**Trigger:** CI build fails on `main` / a release branch / a PR branch.

**What it does:** Classifies the failure cause and dispatches the right
follow-up. Within minutes of a red build (not days), the failure is labeled
as one of:

- **Flaky test** — retry, log to flake-tracker, no human paged.
- **Real bug** — bisect to introducing commit, page commit author + CODEOWNERS.
- **Environment failure** — runner / vendor / network — escalate to infra
  on-call, retry on a different runner.
- **Known issue** — match against open incidents in
  [Knowledge Layer A](#shared-infrastructure); link to existing tracking issue.
- **Upstream dependency churn** — match against recent dependency updates;
  page the dependency-management owner.

**Output contract:**

```python
@dataclass
class TriageResult:
    failure_class: str                # flaky | real_bug | env | known | upstream
    confidence: float
    suspected_commits: list[CommitRef]
    suggested_assignee: str           # CODEOWNERS / on-call / individual
    related_incidents: list[IncidentRef]
    rationale: str
    auto_actions_taken: list[str]     # e.g., retry-on-different-runner
```

**Why NVIDIA cares about this most:** NVIDIA IPP serves teams whose builds run
for hours, not minutes (CUDA compiles, multi-arch link, NeMo/TensorRT/Driver
matrix). Every red build that takes a human 30+ minutes to triage represents
direct engineer-hour cost; at the volumes IPP supports, this is the largest
single MTTR lever in the SDLC. **Triage is more valuable than prediction at
NVIDIA scale**, because the cost of triage delay scales with build duration —
which scales with NVIDIA's domain reality (HW-SW codesign, multi-arch).

**Shared infrastructure it consumes:**

- [Knowledge Layer A](#shared-infrastructure) — past incidents, build-failure
  RCA, dependency advisories (already implemented in
  [`../src/rag/layer_a.py`](../src/rag/layer_a.py)).
- [Multi-vendor model gateway](#shared-infrastructure) — for classifier and
  reasoning.
- [FeedbackLog](#shared-infrastructure) — `(triage_label, actual_root_cause)`
  is the supervision signal for the next training run; closes the same loop
  shape as Node 1's `(prediction, merge_outcome)`.

**Interface signature** (matches the `SDLCWorkflowAgent` abstraction):

```python
class BuildFailureTriageWorkflow(SDLCWorkflowAgent):
    """Triggered by: CI build fail event.

    Input: BuildFailureEvent (build_id, repo, branch, failed_step, logs_uri).
    Output: TriageResult.
    """

    def handle(self, event: BuildFailureEvent) -> TriageResult: ...
```

**Why not in v0.1:** Requires either real CI failure logs (private to most
orgs) or a credible synthetic-failure corpus to train the classifier. A 5–7
day stub on public OSS CI logs (GitHub Actions failure logs from a small set
of large repos) is feasible and is the recommended next-week build (see
[Prioritization](#prioritization--if-i-had-one-more-week)).

---

## Node 3 — Smart Test Selection Workflow (Roadmap)

**Status:** roadmap.

**Trigger:** PR opened / push to PR branch (parallel to Node 1).

**What it does:** Given a diff, decides which subset of the test matrix to
run — instead of running the full matrix on every change. For organizations
with a meaningful test matrix (multi-arch builds, GPU × OS × driver matrices,
DL framework × backend combinations), full-matrix execution per PR is
prohibitively slow; intelligent subset selection is direct cost recovery.

**Output contract:**

```python
@dataclass
class TestPlan:
    selected_configs: list[BuildConfig]  # subset of full matrix
    skipped_configs: list[BuildConfig]   # explicitly excluded, with reason
    estimated_ci_minutes: float
    baseline_ci_minutes: float            # what full-matrix would cost
    rationale: str
    confidence: float
```

**Why NVIDIA cares about this:** GPU × OS × driver × CUDA-version ×
DL-framework is a combinatorial space. Full-matrix CI per PR is not the
status quo at scale — selective execution is. An LLM-grounded selector
(diff content + dependency graph + historical test-fail correlation) is the
shape NVIDIA's IPP team is most likely already prototyping internally.

**Shared infrastructure it consumes:**

- Build-artifact dependency graph (would need to be added as a new shared
  index — not currently in the repo).
- [Knowledge Layer A](#shared-infrastructure) — historical test-fail patterns
  per subsystem.
- Same [Multi-vendor model gateway](#shared-infrastructure) for the selector
  reasoning.

**Interface signature:**

```python
class SmartTestSelectionWorkflow(SDLCWorkflowAgent):
    """Triggered by: PR open / push.

    Input: Diff + BuildConfigMatrix (the full matrix the org could run).
    Output: TestPlan.
    """

    def handle(self, event: PrEvent, matrix: BuildConfigMatrix) -> TestPlan: ...
```

**Why not in v0.1:** Realistic dependency-graph construction requires
project-specific tooling (Bazel / Buck2 / equivalent). The OSS prototype
would need a representative target (e.g., a project with a Bazel BUILD
graph) and would not generalize to the NVIDIA-stack-specific reality without
adapter work. Lower portability than Node #2; recommended after Node #2.

---

## Node 4 — Release Readiness Workflow (Roadmap)

**Status:** roadmap.

**Trigger:** Release branch cut / scheduled nightly / on-demand.

**What it does:** Aggregates risk across **N commits on a release branch**,
not just one. Combines:

- Per-commit risk scores (output of Node #1)
- Test coverage delta vs the previous release
- Open customer hotfix backlog targeting this release
- Compliance gate status (if Node #5 has flagged any cross-org dependency
  changes)
- Recent CI failure rate trend on this branch

…into a single **release-readiness signal**: `ready` / `blocked` /
`conditional-ready`, with cited reasons.

**Output contract:**

```python
@dataclass
class ReadinessReport:
    branch: str
    decision: str                    # ready | conditional | blocked
    aggregate_risk: float
    contributing_signals: list[Signal]
    open_blockers: list[Blocker]
    customer_hotfix_backlog: list[HotfixRef]
    compliance_status: ComplianceStatus
    confidence: float
```

**Why NVIDIA cares about this:** NVIDIA ships against multi-SKU release
windows (driver releases, CUDA toolkit releases, NeMo releases, automotive
firmware drops). The "is the branch ready to cut?" decision is currently
human-judgment-heavy and information-distributed across dashboards,
spreadsheets, and Slack threads. A grounded aggregator that cites its sources
is direct relief.

**Shared infrastructure it consumes:**

- Per-commit risk scores from Node #1's [FeedbackLog](#shared-infrastructure).
- [Knowledge Layer B](#shared-infrastructure) — release calendar, freeze
  windows.
- [Knowledge Layer C](#shared-infrastructure) — customer hotfix backlog,
  compliance gate status (NVIDIA-specific: ASIL automotive sign-off).

**Interface signature:**

```python
class ReleaseReadinessWorkflow(SDLCWorkflowAgent):
    """Triggered by: release branch cut / nightly / on-demand.

    Input: ReleaseBranchEvent (branch, target_release, cutoff_commit).
    Output: ReadinessReport.
    """

    def handle(self, event: ReleaseBranchEvent) -> ReadinessReport: ...
```

**Why not in v0.1:** Requires Node #1 to have produced enough labeled
historical predictions for the aggregator to have signal to aggregate. Also
requires the Layer C retrievers (compliance, customer hotfix) to be hooked
up to real sources. Highest-value node once Node #1 has been running for a
quarter; lower value before that.

---

## Node 5 — Cross-Team Dependency Tracker (Roadmap)

**Status:** roadmap.

**Trigger:** Push that modifies an API boundary, ABI surface, register
layout, public header, or feature flag (detected via path-pattern + AST
diff).

**What it does:** Traces downstream consumers of the changed surface across
the codebase / org. Notifies downstream owners; optionally auto-triggers
downstream regression suites.

For NVIDIA-shape orgs, "downstream consumers" includes:

- Driver code consuming a firmware register layout
- CUDA libraries consuming a driver ABI
- DL frameworks (PyTorch / TensorFlow / NeMo / TensorRT) consuming CUDA APIs
- Customer SDKs consuming framework APIs
- Automotive customers consuming driver behavior contracts (with ASIL
  constraints)

**Output contract:**

```python
@dataclass
class DependencyImpactReport:
    changed_surface: list[SurfaceRef]   # API / ABI / register / flag
    downstream_consumers: list[ConsumerRef]
    suggested_notifications: list[Notification]
    auto_triggered_regressions: list[RegressionRun]
    cross_org_owners_to_loop_in: list[Owner]
```

**Why NVIDIA cares about this:** This is the **HW-SW codesign reality**
explicitly. A driver change that breaks a customer-SDK ABI is the kind of
incident NVIDIA cannot afford. The current state of practice at most orgs
(including NVIDIA, by the JD's framing of *"cross-functionally with
Graphics, Mobile, DL, AI, Driverless Cars"*) is tribal-knowledge + Slack +
manual CODEOWNERS lookup. Automating this with grounded retrieval over the
dependency graph + change-history pattern mining is direct cross-org
incident reduction.

**Shared infrastructure it consumes:**

- A cross-repo dependency graph (would need to be added — not in v0.1).
- [Knowledge Layer A](#shared-infrastructure) — historical co-change
  patterns.
- [Knowledge Layer C](#shared-infrastructure) — domain-specific contracts
  (NVIDIA: HW errata, driver compat matrices).
- Action-surface notification primitives.

**Interface signature:**

```python
class CrossTeamDependencyWorkflow(SDLCWorkflowAgent):
    """Triggered by: push that modifies an API/ABI/register/flag surface.

    Input: SurfaceChangeEvent (commit, changed_surface_kind, scope).
    Output: DependencyImpactReport.
    """

    def handle(self, event: SurfaceChangeEvent) -> DependencyImpactReport: ...
```

**Why not in v0.1:** Requires cross-repo dependency-graph construction —
the most expensive piece of shared infrastructure in the system. A v0.2
single-repo version (using import-graph + AST diff inside one repo) is
tractable but lacks the cross-org payoff that makes it valuable at NVIDIA
scale.

---

## Shared Infrastructure

Five nodes, but they share the same plumbing. Building Node #1 to
production-shape forced the right abstractions; the remaining nodes plug in.

### Orchestrator — `SDLCWorkflowAgent`

A top-level workflow agent that routes events to specialist workflows.
Conceptually:

```python
class SDLCWorkflowAgent:
    """Top-level orchestrator. Routes SDLC events to specialist workflows."""

    workflows: dict[EventKind, SDLCWorkflow]

    def dispatch(self, event: SDLCEvent) -> WorkflowResult: ...


class SDLCWorkflow(ABC):
    """Base class for a single specialist workflow (one of the 5 nodes)."""

    @abstractmethod
    def handle(self, event: SDLCEvent) -> WorkflowResult: ...
```

In v0.1 this abstraction is implicit — the
[`AgentHarness`](../src/agent/harness.py) is in practice the
`PreMergeRiskWorkflow`. Elevating the harness to be one specialization of
`SDLCWorkflow` is a small refactor that buys the full architectural framing;
it is the recommended v0.2 first step.

### Knowledge Layer — Heterogeneous RAG Gateway

Already implemented in [`../src/rag/`](../src/rag/). The three-layer
taxonomy (A: code-adjacent, B: operational metadata, C: high-stakes
enterprise) is exactly the shape every workflow needs — Node #2 queries
Layer A heavily, Node #4 queries Layers B and C, Node #5 queries Layer C for
domain-specific contracts.

The design principle is that **each layer is a pluggable retriever**: OSS
deployments use public-data proxies; enterprise deployments swap the backend
without changing the workflow-facing tool signature. This is the same
principle that makes the multi-vendor model gateway substitutable.

### Multi-vendor Model Gateway

Already implemented in [`../src/models/gateway.py`](../src/models/gateway.py).
Uniform `predict()` interface across Claude, NVIDIA NIM, Triton-served NeMo
fine-tune, and Azure OpenAI. Every node selects its backend through the
gateway. A/B comparison and graceful degradation are first-class.

### FeedbackLog + OutcomeLabeler

Already implemented in
[`../src/storage/feedback_log.py`](../src/storage/feedback_log.py) and
[`../src/storage/outcome_labeler.py`](../src/storage/outcome_labeler.py).
The append-only `(prediction, outcome)` log is the same shape for every
workflow:

- Node #1: `(risk_score, merge_outcome | revert | incident_link)`
- Node #2: `(triage_label, actual_root_cause)`
- Node #3: `(skipped_configs, actual_failed_configs)`
- Node #4: `(readiness_decision, actual_release_outcome)`
- Node #5: `(impact_report, actual_downstream_incidents)`

Every workflow's supervision signal is the prediction-outcome pair recorded
through the same primitive. This is what makes "DORA dashboard" honest: the
metrics are computed off the actual outcome record, not synthesized.

### Audit Store

Multi-backend audit log in
[`../src/storage/audit_store.py`](../src/storage/audit_store.py) — MongoDB /
MySQL / Elasticsearch. Every workflow's input / output / action is recorded
for traceability. Required by
[`enterprise-safety.md`](enterprise-safety.md) Control 6.

### Action Surface

Currently implemented in [`../src/agent/policy.py`](../src/agent/policy.py)
and [`../src/agent/explainer.py`](../src/agent/explainer.py) — the Pre-Merge
node has PR-comment, reviewer-assign, and merge-gate actions. v0.2 extends
to: CI skip-config (consumed by Node #3), on-call page (consumed by Nodes
#2 and #4), and notification fan-out (consumed by Node #5).

---

## Why Not Build All Five At Once

A reasonable question. Three reasons, in order of importance:

### 1. Depth signals more than breadth in a portfolio artifact

A reviewer evaluating an open-source artifact for a senior platform-team role
is looking for *"does this person know how to build one thing to production
shape?"* — not *"does this person know how to scaffold five things at 20%
depth each?"* The Pre-Merge node at v0.1 is deliberately over-built: tiered
router, heterogeneous RAG, eval-gated CI, NVIDIA Garak red-team, NeMo
Guardrails, multi-backend audit, 86 tests. This is the depth signal; five
half-finished nodes would be the anti-signal.

### 2. The shared infrastructure carries the architectural claim

Once Node #1 is built to production-shape, the abstractions it forced
(`SubAgent` base class, `ModelGateway`, three-layer RAG, `FeedbackLog`,
`AuditStore`, policy gatekeeper) are exactly the abstractions the other four
nodes need. The architectural integration claim — *"these are not five
disconnected agents, they share infrastructure"* — is **provable from the
existing code**, not aspirational.

### 3. Honest scoping beats optimistic scoping

A v0.1 that ships one node end-to-end with explicit roadmap for the
remaining four is more credible than a v0.1 that ships five stubs and
hopes the reviewer fills in the rest. The
[`limitations.md`](limitations.md) document is the same intellectual move at
the project level; this document is its system-level analog.

---

## Mapping to NVIDIA IPP — JD Bullet-by-Bullet

The NVIDIA IPP JR2017785 JD has five "What you'll be doing" bullets and
three "Ways to stand out" differentiators. The 5-node system maps as
follows:

| JD bullet / stand-out | System component |
|---|---|
| **Bullet 1**: improve developer efficiency, accelerate feedback loops, boost release reliability | All 5 nodes — Node #1 (CFR), Node #2 (MTTR), Node #3 (cycle time), Node #4 (deployment frequency), Node #5 (cross-org incident) |
| **Bullet 2**: design, develop, deploy AI agents to automate dev workflows | Orchestrator + 5 specialist workflows; v0.1 ships Node #1 end-to-end |
| **Bullet 3**: measure / report cycle time, CFR, MTTR | DORA-aligned metrics loop; each node ties to one DORA metric explicitly |
| **Bullet 4** ⭐: predictive models for high-risk commits / build-failure forecasting | Node #1 (high-risk commits — 1:1 match) + Node #2 (build-failure forecasting — natural extension) |
| **Bullet 5**: research emerging AI | Multi-vendor gateway + Garak / NeMo Guardrails integration |
| **Stand-out**: RAG on enterprise data | Three-layer enterprise RAG taxonomy (Layer A/B/C) — already implemented |
| **Stand-out**: Fine-tuning on enterprise data | Node #1 T2 tier — Mistral-7B-v0.3 + LoRA via NeMo, with pluggable `TrainingDataSource` |
| **Stand-out**: large-scale, real-time + agentic AI for complex workflows | Tiered router with sub-10ms T1 gate + multi-step agent orchestration in T3 |

The "shipped node + roadmap" framing is what allows this project to claim
coverage of all 5 JD bullets honestly — Node #1 carries bullets 3 and 4
already; Nodes #2–#5 carry the rest as roadmap with concrete interfaces.

---

## Prioritization — If I Had One More Week

For an adopting team (or a hiring panel asking *"what would you build
next?"*), the priority order is:

1. **Node #2 — Build Failure Triage Workflow.** Highest NVIDIA-specific
   value (multi-hour CI economics + MTTR is the most expensive DORA metric
   at NVIDIA scale). Most reusable shared infrastructure (consumes Layer A
   and the same gateway as Node #1). 5–7 days for a credible stub on
   public OSS CI failure logs.
2. **Node #5 — Cross-Team Dependency Tracker (single-repo version first).**
   Strong HW-SW codesign signal. v0.2 could ship a single-repo version
   (import-graph + AST diff) in ~1 week; cross-repo version is ~3 weeks.
3. **Node #4 — Release Readiness Workflow.** Most valuable *once Node #1
   has been running long enough to have labeled history*. Build after
   Node #1 has been deployed for at least a quarter.
4. **Node #3 — Smart Test Selection.** Lowest portability (build-system
   specific). Build last, against a specific adopter's BUILD graph.

This priority order is **honest at interview**: the reasoning is that the
highest NVIDIA-pain-point match (multi-hour build triage) wins over the
generically-valuable but adopter-specific node (test selection).

---

## See Also

- [`design-doc.md`](design-doc.md) — Node #1 full architecture (Tiered Router,
  Hybrid Scoring, Heterogeneous RAG taxonomy)
- [`limitations.md`](limitations.md) — what doesn't work / works less well
- [`metrics.md`](metrics.md) — DORA metric definitions
- [`enterprise-safety.md`](enterprise-safety.md) — production-safety controls
- [`onboarding.md`](onboarding.md) — adopter integration guide
- [`runbook.md`](runbook.md) — what to do when the agent misfires

---

*— Mingdong (Eric) Tan, 2026-05-13*
