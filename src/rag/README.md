# `src/rag/` — Multi-source enterprise RAG

This package implements the Tier-3 retrieval layer described in
[design-doc.md §Architecture → "Multi-source enterprise RAG taxonomy"](../../docs/design-doc.md).
It is the architectural answer to the JD stand-out criterion of
*"Applied AI using RAG and fine-tuning LLMs on enterprise data"*.

## Why three layers, not one flat vector store

Real enterprise knowledge bases are heterogeneous: code, time-series,
long-form prose, structured KV, domain spec. A single embedding + vector
store cannot serve all of them well. The cost of pretending otherwise is
uniformly mediocre retrieval and high token bills.

The three layers partition by **frequency × structure × signal-shape**:

| Layer | Frequency | Source shape | Activation | Retrieval |
|---|---|---|---|---|
| **A — Code-adjacent operational** | Every Tier-3 call | Short, structured, code-related | Always-on | Code-aware embeddings + BM25 hybrid |
| **B — Operational context** | When metadata triggers | Time-series / tabular | Trigger flags from PR metadata | Structured queries + light vector |
| **C — Enterprise knowledge** | Rare, high-stakes | Long prose, heterogeneous | Agent decides per diff (agentic RAG) | Long-context embeddings + cross-encoder |

## OSS-mode vs enterprise-mode

Every retriever defines an interface and provides an OSS-data proxy
implementation (currently as a stub that documents the proxy). Enterprise
adopters swap the backend without changing the agent-facing tool signature:

```python
# OSS mode — uses public-data proxies (default)
gateway = RagGateway()

# Enterprise mode — swap specific retrievers
gateway = RagGateway(retrievers=[
    InternalIncidentICMRetriever(icm_client=icm),
    InternalComplianceConfluenceRetriever(confluence=confluence),
    NvidiaDriverCompatRetriever(driver_db=driver_db),
    # ...plus all OSS retrievers for the rest
])
```

This is the selling point for an enterprise sales conversation: *"the
architecture supports your data sources, not just ours"*.

## Status

| File | Status |
|---|---|
| `types.py`, `base.py`, `gateway.py` | Concrete, tested |
| `layer_a.py` | Interfaces concrete, implementations are stubs |
| `layer_b.py` | Interfaces concrete, implementations are stubs |
| `layer_c.py` | Interfaces concrete, implementations are stubs |

Stubs raise `NotImplementedError` with a message naming the OSS proxy and
the enterprise backend. The gateway catches these and degrades gracefully
to "empty result" so the system runs end-to-end while implementations
land incrementally.

## Agentic RAG (Layer C)

Layer C is **not** auto-invoked. Each Layer-C retriever exposes a
`when_to_invoke` description; the Tier-3 agent receives these as tool
descriptions and decides which to call based on the diff. This is the
difference between:

- *Flat RAG*: system embeds everything, queries on every PR, hopes the
  ranker picks the right thing.
- *Agentic RAG*: agent reasons about the diff first, then picks the
  retriever whose description matches.

Layer C carries the rarest, highest-stakes signals (compliance, vendor
contracts, domain spec). Calling all of them on every PR would be wasteful;
calling none would miss the signals when they fire. Letting the agent
choose threads the needle.

## Where this maps in the data flow

```
T3 Claude Agent (Sonnet 4.6)
        │
        ├─► RagGateway.dispatch(query)         ◄── Layers A (always) + B (if triggered)
        │        returns evidence bundle
        │
        └─► RagGateway.dispatch_layer_c(name)  ◄── Layer C, called by the agent
                 per-call, when its reasoning requires it
```

The combined evidence — Layer-A always + Layer-B triggered + Layer-C agent-chosen
— is what the agent grounds its final risk verdict on. Provenance (via the
`Citation` type) is preserved end-to-end so reviewers can audit any high-risk
conclusion.
