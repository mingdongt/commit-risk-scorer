"""Tests for the red-team probe scaffolding.

These tests validate the *registry*, not Garak execution itself — that wiring
ships in v0.2. They guard against accidental probe-set regressions (severity
typos, threshold drift, missing rationale).
"""
from __future__ import annotations

import pytest

from src.eval.red_team import PROBES, VALID_SEVERITIES, ProbeSpec, run_probes


def test_probe_registry_nonempty():
    """At least 3 probes shipped — anything less is suspicious."""
    assert len(PROBES) >= 3


def test_every_probe_has_rationale():
    """No probe ships without a documented rationale."""
    for probe in PROBES:
        assert probe.rationale.strip(), f"probe {probe.name} has empty rationale"
        assert len(probe.rationale) > 30, f"probe {probe.name} rationale too terse"


def test_every_probe_has_valid_severity():
    for probe in PROBES:
        assert probe.severity in VALID_SEVERITIES, (
            f"probe {probe.name} has invalid severity {probe.severity!r}"
        )


def test_every_probe_has_garak_module_path():
    """Garak module path follows garak.probes.<name>."""
    for probe in PROBES:
        assert probe.garak_module.startswith("garak.probes."), (
            f"probe {probe.name} has non-garak module path: {probe.garak_module}"
        )


def test_failure_rate_thresholds_reasonable():
    """Thresholds must be within (0, 0.5] — anything above 50% is meaningless."""
    for probe in PROBES:
        assert 0.0 < probe.max_failure_rate <= 0.5, (
            f"probe {probe.name} has unreasonable threshold {probe.max_failure_rate}"
        )


def test_critical_probes_have_tight_thresholds():
    """Critical-severity probes should never tolerate > 5% failure."""
    for probe in PROBES:
        if probe.severity == "critical":
            assert probe.max_failure_rate <= 0.05, (
                f"critical probe {probe.name} has lax threshold {probe.max_failure_rate}"
            )


def test_probe_names_unique():
    names = [p.name for p in PROBES]
    assert len(names) == len(set(names)), f"duplicate probe names: {names}"


def test_run_probes_is_stub_until_v0_2():
    """Until the Claude judge backend lands, run_probes must raise — protects against
    accidental "succeeded" claims when no Garak run actually happened."""
    with pytest.raises(NotImplementedError):
        run_probes(judge_backend=None)
