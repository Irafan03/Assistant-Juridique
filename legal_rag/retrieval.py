from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from .interfaces import STRATEGIES, AnswerGenerator, IndexAdapter, Reranker
from .models import Candidate, Chunk, SearchResponse
from .references import extract_references, normalize_reference


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06ff]+", text.casefold(), flags=re.UNICODE)


def _is_type(chunk: Chunk, *types: str) -> bool:
    value = (chunk.document_type or "").casefold()
    return any(label in value for label in types)


class LegalRetriever:
    def __init__(
        self,
        index: IndexAdapter,
        *,
        reranker: Reranker | None = None,
        answer_generator: AnswerGenerator | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.index = index
        self.reranker = reranker
        self.answer_generator = answer_generator
        self.rrf_k = rrf_k

    def _lexical_search(self, query: str, chunks: Sequence[Chunk], limit: int) -> list[tuple[Chunk, float]]:
        # Small dependency-free BM25 implementation, with document frequency over
        # the currently selected strategy's chunks.
        query_terms = set(_tokens(query))
        docs = [_tokens(f"{c.title or ''} {c.article_number or ''} {c.text} {' '.join(c.references)}") for c in chunks]
        if not query_terms or not docs:
            return []
        avg_len = sum(map(len, docs)) / len(docs) or 1
        df = {term: sum(term in set(doc) for doc in docs) for term in query_terms}
        scored: list[tuple[Chunk, float]] = []
        k1, b = 1.5, 0.75
        for chunk, doc in zip(chunks, docs):
            if not doc:
                continue
            frequencies = {term: doc.count(term) for term in query_terms}
            score = 0.0
            for term, tf in frequencies.items():
                if not tf:
                    continue
                idf = __import__("math").log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
            if score:
                scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def search(self, query: str, *, strategy: str = "structural", retrieval_mode: str = "hybrid", limit: int = 10, generate_answer: bool = False) -> SearchResponse:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
        if retrieval_mode not in {"hybrid", "semantic"}:
            raise ValueError("retrieval_mode must be 'hybrid' or 'semantic'")
        if not query.strip():
            raise ValueError("query must not be empty")
        pool = list(self.index.iter_chunks(strategy=strategy))
        semantic = list(self.index.search(query, strategy=strategy, limit=max(limit * 4, 20)))
        lexical = self._lexical_search(query, pool, max(limit * 4, 20)) if retrieval_mode == "hybrid" else []

        ranks: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        semantic_ids = {c.chunk_id for c in semantic}
        lexical_ids = {c.chunk_id for c, _ in lexical}
        for results in (semantic, [c for c, _ in lexical]):
            for rank, chunk in enumerate(results, start=1):
                by_id[chunk.chunk_id] = chunk
                ranks[chunk.chunk_id] += 1 / (self.rrf_k + rank)

        query_refs = {normalize_reference(ref) for ref in extract_references(query)}
        exact_ids: set[str] = set()
        if retrieval_mode == "hybrid":
            for chunk in pool:
                refs = [*chunk.references, *extract_references(chunk.text)]
                if query_refs.intersection(normalize_reference(ref) for ref in refs):
                    exact_ids.add(chunk.chunk_id)
                    by_id[chunk.chunk_id] = chunk
                    ranks[chunk.chunk_id] += 1.0

        # Explicit reference matches are included even when they were not returned
        # by either ranker. Similarity alone is never labeled as a legal relation.
        ranked = sorted(ranks, key=ranks.get, reverse=True)
        semantic_rank = {chunk.chunk_id: rank for rank, chunk in enumerate(semantic, 1)}
        lexical_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(lexical, 1)}
        candidates = [Candidate(
            chunk=by_id[cid], score=ranks[cid],
            semantic_score=(1 / (self.rrf_k + semantic_rank[cid]) if cid in semantic_ids else None),
            lexical_score=(1 / (self.rrf_k + lexical_rank[cid]) if cid in lexical_ids else None),
            exact_reference=cid in exact_ids,
            relation="explicit_reference" if cid in exact_ids else "semantic_similarity",
            citation_id=f"[{ranked.index(cid) + 1}]",
        ) for cid in ranked[:max(limit * 3, 20)]]

        if self.reranker and candidates:
            reranked = list(self.reranker.rerank(query, [c.chunk for c in candidates]))
            rank_map = {chunk.chunk_id: rank for rank, chunk in enumerate(reranked, 1)}
            candidates.sort(key=lambda c: rank_map.get(c.chunk.chunk_id, len(rank_map) + 1))

        # Add parent context while retaining article-level children as evidence.
        contextual: list[Candidate] = []
        seen = set()
        for item in candidates:
            chunk = item.chunk
            if chunk.chunk_id not in seen:
                contextual.append(item)
                seen.add(chunk.chunk_id)
            if chunk.parent_id:
                parent = self.index.get_chunk(chunk.parent_id, strategy=strategy) or self.index.get_parent(chunk, strategy=strategy)
                if parent and parent.chunk_id not in seen:
                    contextual.append(Candidate(parent, item.score * 0.95, relation="parent_context"))
                    seen.add(parent.chunk_id)

        passages = [replace(item, citation_id=f"[{rank}]") for rank, item in enumerate(contextual[:limit], 1)]
        circulares = []
        circular_documents: set[str] = set()
        for item in contextual:
            if _is_type(item.chunk, "circulaire", "circular", "منشور") and item.chunk.document_id not in circular_documents:
                circulares.append(item)
                circular_documents.add(item.chunk.document_id)
            if len(circulares) >= limit:
                break
        # Resolve relations cited by relevant circulars against corpus metadata and
        # text. These are explicit links; semantic neighbors stay separate.
        associated_ids: set[str] = set()
        for item in contextual:
            if not _is_type(item.chunk, "circulaire", "circular", "منشور"):
                continue
            cited = {normalize_reference(ref) for ref in (*item.chunk.references, *extract_references(item.chunk.text))}
            if not cited:
                continue
            for target in pool:
                if target.document_id == item.chunk.document_id or not _is_type(target, "loi", "décret", "decret", "arrêté", "arrete", "texte", "قانون", "مرسوم", "قرار"):
                    continue
                target_refs = {normalize_reference(ref) for ref in (*target.references, *extract_references(target.text))}
                target_refs.update(normalize_reference(ref) for ref in extract_references(target.title or ""))
                if cited.intersection(target_refs):
                    associated_ids.add(target.chunk_id)
                    if target.chunk_id not in seen:
                        contextual.append(Candidate(target, 0.0, exact_reference=True, relation="explicit_reference"))
                        seen.add(target.chunk_id)
        associated = []
        associated_documents: set[str] = set()
        for item in contextual:
            if item.chunk.chunk_id in associated_ids and item.chunk.document_id not in associated_documents:
                associated.append(item)
                associated_documents.add(item.chunk.document_id)
            if len(associated) >= limit:
                break
        articles = [c for c in contextual if c.chunk.article_number or _is_type(c.chunk, "article", "مادة")][:limit]
        citation_by_chunk = {item.chunk.chunk_id: item.citation_id for item in passages}
        circulares = [replace(item, citation_id=citation_by_chunk.get(item.chunk.chunk_id)) for item in circulares]
        associated = [replace(item, citation_id=citation_by_chunk.get(item.chunk.chunk_id)) for item in associated]
        articles = [replace(item, citation_id=citation_by_chunk.get(item.chunk.chunk_id)) for item in articles]
        warnings: list[str] = []
        if not contextual:
            warnings.append("Aucun passage candidat trouvé dans l’index sélectionné.")
        if not associated and query_refs:
            warnings.append("Une référence explicite apparaît dans la requête, mais aucun texte correspondant n’a été résolu dans les métadonnées du corpus.")
        if not self.answer_generator and generate_answer:
            warnings.append("Aucun générateur de réponse n’est configuré; aucune réponse juridique n’a été produite.")
        answer = None
        if generate_answer and self.answer_generator and passages:
            proposed = self.answer_generator.generate(query, evidence=passages)
            cited = {int(number) for number in re.findall(r"\[(\d+)\]", proposed)}
            allowed = set(range(1, len(passages) + 1))
            if cited and cited.issubset(allowed):
                answer = proposed
            else:
                warnings.append("La réponse générée ne contient pas de citations vérifiables vers les passages fournis; elle a été masquée.")
        return SearchResponse(query, strategy, retrieval_mode, circulares, associated, articles, passages, answer, warnings)
