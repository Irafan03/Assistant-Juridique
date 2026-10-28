"""Assistant juridique — ingestion, indexation et retrieval RAG."""

# --- Schéma commun (Prompt 1 & 2) ---
from .models import Candidate, Chunk, SearchResponse

# --- Retrieval (Prompt 2) ---
from .retrieval import LegalRetriever

# --- Ingestion & indexation (Prompt 1) ---
from .extraction import RawDocument, RawPage, extract_document
from .chunking import chunk_document
from .indexing import EmbeddingModel, LegalIndex

__all__ = [
    # Partagé
    "Candidate",
    "Chunk",
    "SearchResponse",
    # Retrieval
    "LegalRetriever",
    # Ingestion
    "RawDocument",
    "RawPage",
    "extract_document",
    "chunk_document",
    "EmbeddingModel",
    "LegalIndex",
]
