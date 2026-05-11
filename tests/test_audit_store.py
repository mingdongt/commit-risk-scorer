"""Tests for the audit-store interfaces.

Validates the AuditEntry contract and the stub-raising behavior of each backend.
Driver-level integration tests land in v0.2 with the actual backend wiring.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.storage.audit_store import (
    AuditEntry,
    ElasticsearchAuditStore,
    MongoAuditStore,
    MySQLAuditStore,
    TeeAuditStore,
)


def _sample_entry() -> AuditEntry:
    return AuditEntry(
        pr_id="repo/owner#1842",
        commit_sha="deadbeef",
        risk_score=0.72,
        risk_level="High",
        top_risk_factors=["touches auth", "no test coverage"],
        recommended_actions=["add SME reviewer", "run extended tests"],
        confidence=0.81,
        evidence=["src/auth/token_validator.py", "PR #1842 prior failure"],
        model_version="mistral-7b-v0.3-lora-v2",
        adapter_version="commit-risk-v0.1.3",
        prompt_version="judge-v7",
        backend_used="triton_nemo",
        latency_ms=43.0,
    )


def test_audit_entry_round_trip_serialization():
    """AuditEntry.to_dict produces JSON-friendly output (timestamp is ISO string)."""
    import json

    entry = _sample_entry()
    payload = entry.to_dict()
    assert payload["pr_id"] == "repo/owner#1842"
    assert payload["risk_level"] == "High"
    assert isinstance(payload["timestamp"], str)
    # Timestamp must parse back as ISO format
    datetime.fromisoformat(payload["timestamp"])
    # Must serialize to JSON without custom encoder
    json.dumps(payload)


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda: MongoAuditStore(),
        lambda: MySQLAuditStore(),
        lambda: ElasticsearchAuditStore(),
    ],
)
def test_stub_backends_raise_until_v0_2(store_factory):
    """Stubs raise NotImplementedError until backend wiring lands.

    Protects against accidental 'succeeded' claims when no real backend is
    plugged in — the agent must not silently drop audit records.
    """
    store = store_factory()
    with pytest.raises(NotImplementedError):
        store.write(_sample_entry())
    with pytest.raises(NotImplementedError):
        store.query_by_pr("anything")


def test_backend_names_unique():
    """Each backend reports a distinct name (used in audit-log metadata)."""
    names = {
        MongoAuditStore().name(),
        MySQLAuditStore().name(),
        ElasticsearchAuditStore().name(),
    }
    assert names == {"mongodb", "mysql", "elasticsearch"}


def test_tee_store_composes_name():
    """TeeAuditStore reports the composition of its underlying backends."""
    tee = TeeAuditStore(
        primary=MySQLAuditStore(),
        mirrors=[ElasticsearchAuditStore()],
    )
    assert tee.name() == "tee[mysql+elasticsearch]"


def test_tee_primary_failure_bubbles():
    """A primary-backend failure must propagate to the caller (no silent drop)."""
    tee = TeeAuditStore(primary=MySQLAuditStore())
    with pytest.raises(NotImplementedError):
        tee.write(_sample_entry())
