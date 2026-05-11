"""Audit-log storage for the agent's score / evidence / model-version trail.

Required by `docs/enterprise-safety.md` Control 6 (Audit log). Three concrete
backends are supported so that adopting orgs can plug into the storage stack
they already operate:

    - MongoAuditStore           — document-oriented; good for the flexible-schema
                                  evidence-trail payload (variable-length lists,
                                  nested objects).
    - MySQLAuditStore           — relational + transactional; good for the
                                  structured rollout columns (model_version,
                                  prompt_version, backend_used) and tight
                                  per-PR / per-time indexes.
    - ElasticsearchAuditStore   — search-first; good when the org already
                                  indexes ICM / build telemetry in ES and wants
                                  the agent's audit trail in the same surface.

Large orgs typically pick one as the source of truth; smaller orgs often
co-locate the audit trail with the historical-PR RAG index (Elasticsearch).

Status (v0.1): interfaces + 3 stub backends. Driver wiring lands in v0.2 once
the first adopter team chooses its primary backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_utc() -> datetime:
    """Timezone-aware UTC now — avoids the Python 3.12+ utcnow() deprecation."""
    return datetime.now(UTC)


@dataclass
class AuditEntry:
    """One agent-output record. The fields here are the *minimum* required by
    `enterprise-safety.md` Control 6; backends may add their own metadata.
    """

    pr_id: str
    commit_sha: str
    risk_score: float
    risk_level: str  # "Low" | "Medium" | "High" | "Critical"
    top_risk_factors: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    # Reproducibility fields — required for regression triage.
    model_version: str = ""
    adapter_version: str = ""
    prompt_version: str = ""
    backend_used: str = ""

    # Operational fields.
    latency_ms: float = 0.0
    human_override: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class AuditStore(ABC):
    """Interface every backend implements."""

    @abstractmethod
    def write(self, entry: AuditEntry) -> None: ...

    @abstractmethod
    def query_by_pr(self, pr_id: str) -> list[AuditEntry]: ...

    @abstractmethod
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------


class MongoAuditStore(AuditStore):
    """MongoDB-backed audit store.

    Schema: a single collection `agent_audit`. Each document = one AuditEntry.
    Indexes recommended: `pr_id` (ascending) + `timestamp` (descending).

    Status: STUB. v0.2 wires `pymongo` and adds bulk-write batching.
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        database: str = "commit_risk_scorer",
        collection: str = "agent_audit",
    ):
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection = collection

    def write(self, entry: AuditEntry) -> None:
        raise NotImplementedError(
            "MongoAuditStore.write stub — v0.2 wires pymongo.MongoClient.insert_one()."
        )

    def query_by_pr(self, pr_id: str) -> list[AuditEntry]:
        raise NotImplementedError(
            "MongoAuditStore.query_by_pr stub — v0.2 wires pymongo find({'pr_id': pr_id})."
        )

    def name(self) -> str:
        return "mongodb"


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------


class MySQLAuditStore(AuditStore):
    """MySQL-backed audit store.

    Schema: a single table `agent_audit` with columns matching AuditEntry plus
    JSON columns for the variable-length fields (evidence, top_risk_factors,
    recommended_actions). Primary key (pr_id, timestamp); index on commit_sha.

    Status: STUB. v0.2 wires SQLAlchemy + Alembic for schema migrations.
    """

    def __init__(
        self,
        connection_string: str = "mysql+pymysql://localhost:3306/commit_risk_scorer",
    ):
        self.connection_string = connection_string

    def write(self, entry: AuditEntry) -> None:
        raise NotImplementedError(
            "MySQLAuditStore.write stub — v0.2 wires SQLAlchemy session.add() + commit()."
        )

    def query_by_pr(self, pr_id: str) -> list[AuditEntry]:
        raise NotImplementedError(
            "MySQLAuditStore.query_by_pr stub — v0.2 wires SQLAlchemy filter_by(pr_id=...)."
        )

    def name(self) -> str:
        return "mysql"


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------


class ElasticsearchAuditStore(AuditStore):
    """Elasticsearch-backed audit store.

    Co-locates the audit trail with the historical-PR RAG index used by the
    `historical-context` sub-agent — single ES cluster, two index patterns
    (`agent-audit-YYYY.MM` and `pr-history-YYYY.MM`).

    Status: STUB. v0.2 wires `elasticsearch-py` with daily-rolling indices and
    ILM (Index Lifecycle Management) policies for retention.
    """

    def __init__(
        self,
        hosts: list[str] | None = None,
        index_pattern: str = "agent-audit-{date}",
    ):
        self.hosts = hosts or ["http://localhost:9200"]
        self.index_pattern = index_pattern

    def write(self, entry: AuditEntry) -> None:
        raise NotImplementedError(
            "ElasticsearchAuditStore.write stub — v0.2 wires elasticsearch.Elasticsearch.index()."
        )

    def query_by_pr(self, pr_id: str) -> list[AuditEntry]:
        raise NotImplementedError(
            "ElasticsearchAuditStore.query_by_pr stub — v0.2 wires elasticsearch search by term."
        )

    def name(self) -> str:
        return "elasticsearch"


# ---------------------------------------------------------------------------
# Multiplexing (write-to-many)
# ---------------------------------------------------------------------------


class TeeAuditStore(AuditStore):
    """Writes to multiple backends in order — useful during migration or for
    keeping a search-friendly mirror (ES) alongside the system of record (MySQL).

    Failures on a non-primary backend are recorded but do not raise; the primary
    backend's success is what gates the agent's return.
    """

    def __init__(self, primary: AuditStore, mirrors: list[AuditStore] | None = None):
        self.primary = primary
        self.mirrors = mirrors or []

    def write(self, entry: AuditEntry) -> None:
        self.primary.write(entry)  # raises on failure → bubbles to caller
        for m in self.mirrors:
            try:
                m.write(entry)
            except Exception:  # noqa: BLE001 — best-effort mirroring
                # In v0.2 this is logged via OpenTelemetry; the agent return is
                # not blocked on mirror failures.
                pass

    def query_by_pr(self, pr_id: str) -> list[AuditEntry]:
        return self.primary.query_by_pr(pr_id)

    def name(self) -> str:
        return f"tee[{self.primary.name()}+{','.join(m.name() for m in self.mirrors)}]"
