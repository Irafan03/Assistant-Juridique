from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Common index contract; optional fields may be absent in source documents."""

    document_id: str
    chunk_id: str
    text: str
    document_type: str | None = None
    title: str | None = None
    parent_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    article_number: str | None = None
    references: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    verified_metadata: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Chunk":
        allowed = cls.__dataclass_fields__
        data = {key: item for key, item in value.items() if key in allowed}
        data["references"] = tuple(data.get("references") or ())
        data["verified_metadata"] = tuple(data.get("verified_metadata") or ())
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["references"] = list(self.references)
        result["verified_metadata"] = list(self.verified_metadata)
        return result


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    exact_reference: bool = False
    relation: str = "semantic_similarity"
    citation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "exact_reference": self.exact_reference,
            "relation": self.relation,
            "citation_id": self.citation_id,
        }


@dataclass
class SearchResponse:
    query: str
    strategy: str
    retrieval_mode: str = "hybrid"
    circulares: list[Candidate] = field(default_factory=list)
    associated_texts: list[Candidate] = field(default_factory=list)
    articles: list[Candidate] = field(default_factory=list)
    passages: list[Candidate] = field(default_factory=list)
    answer: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "retrieval_mode": self.retrieval_mode,
            "circulaires": [item.to_dict() for item in self.circulares],
            "textes_associes": [item.to_dict() for item in self.associated_texts],
            "articles": [item.to_dict() for item in self.articles],
            "passages": [item.to_dict() for item in self.passages],
            "answer": self.answer,
            "warnings": self.warnings,
        }
