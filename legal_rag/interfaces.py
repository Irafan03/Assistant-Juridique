from __future__ import annotations

from typing import Protocol, Sequence

from .models import Candidate, Chunk


STRATEGIES = ("structural", "hierarchical", "parent_child")


class IndexAdapter(Protocol):
    """Adapter expected from Prompt 1's indexing module.

    `search` must search the named, independently built index. `iter_chunks` feeds
    BM25/reference lookup. Parent lookup returns the requested ancestor when known.
    """

    def search(self, query: str, *, strategy: str, limit: int) -> Sequence[Chunk]: ...

    def iter_chunks(self, *, strategy: str) -> Sequence[Chunk]: ...

    def get_chunk(self, chunk_id: str, *, strategy: str) -> Chunk | None: ...

    def get_parent(self, chunk: Chunk, *, strategy: str) -> Chunk | None: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Chunk]) -> Sequence[Chunk]: ...


class AnswerGenerator(Protocol):
    def generate(self, query: str, *, evidence: Sequence[Candidate]) -> str: ...
