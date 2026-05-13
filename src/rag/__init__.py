"""Multi-source enterprise RAG layer consumed by the Tier-3 agent.

See `src/rag/README.md` for the architecture rationale (why three layers
rather than one flat vector store) and the OSS-proxy / enterprise-backend
swap pattern. See also `docs/design-doc.md` §Architecture →
"Multi-source enterprise RAG taxonomy".
"""
from src.rag.base import Retriever, RetrieverLayer
from src.rag.gateway import RagGateway
from src.rag.types import Citation, Document, RetrievalQuery, RetrievalResult

__all__ = [
    "Citation",
    "Document",
    "RagGateway",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "RetrieverLayer",
]
