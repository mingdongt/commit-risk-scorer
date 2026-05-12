"""Runs the three demo scenarios end-to-end and prints the agent's output.

Usage:
    python -m demo.run_demo               # to stdout
    python -m demo.run_demo > demo/output.md   # capture to a file the repo can show

What it exercises (full v0.1 pipeline):

    Scenario.diff + metadata
        -> AgentHarness.score (5 sub-agents in parallel)
            -> PolicyGatekeeper.decide (4-band action mapping)
                -> ExplanationWriter.render (markdown PR comment)
"""
from __future__ import annotations

import sys

from demo.scenarios import ALL_SCENARIOS, Scenario
from src.agent.explainer import ExplanationWriter
from src.agent.harness import AgentHarness, HarnessResult
from src.agent.policy import PolicyDecision, PolicyGatekeeper


def _render_scenario(scenario: Scenario, harness: AgentHarness, policy: PolicyGatekeeper,
                     explainer: ExplanationWriter) -> None:
    result: HarnessResult = harness.score(scenario.diff, metadata=scenario.metadata)
    decision: PolicyDecision = policy.decide(result)

    print(f"## Scenario {scenario.name}")
    print()
    print(f"_{scenario.narrative}_")
    print()
    print("### Sub-agent reports")
    print()
    print(f"Aggregated risk score: **{result.risk_score:.2f}**")
    print()
    for report in result.sub_agent_reports:
        if report.confidence == 0.0 and not report.risk_factors:
            # Pure stub — keep the demo compact.
            continue
        print(f"- **`{report.sub_agent_name}`** (confidence `{report.confidence:.2f}`)")
        for k, v in report.observations.items():
            if isinstance(v, list) and not v:
                continue
            print(f"  - `{k}`: `{v}`")
        for rf in report.risk_factors:
            print(f"  - ⚠️ {rf}")
    print()
    print(f"### Policy decision")
    print()
    print(f"- **Action:** `{decision.action}`")
    print(f"- **Risk level:** `{decision.risk_level}`")
    print(f"- **Rationale:** {decision.rationale}")
    print()
    print(f"### Rendered PR comment (what the agent posts on the PR)")
    print()
    print("```markdown")
    print(explainer.render(result, decision))
    print("```")
    print()
    print("---")
    print()


def main() -> None:
    harness = AgentHarness()
    policy = PolicyGatekeeper()
    explainer = ExplanationWriter()

    print("# commit-risk-scorer — Demo Walkthrough")
    print()
    print("Three end-to-end runs across the v0.1 pipeline:")
    print("**5 sub-agents → policy gatekeeper → markdown PR comment**.")
    print()
    print("Each scenario uses a different PR shape; the agent's output should")
    print("differ in ways a human reviewer would agree with.")
    print()
    print("---")
    print()

    for scenario in ALL_SCENARIOS:
        _render_scenario(scenario, harness, policy, explainer)

    print("_Regenerate this file with `python -m demo.run_demo > demo/output.md`._")


if __name__ == "__main__":
    # Force UTF-8 stdout on Windows GBK terminals; harmless elsewhere.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
