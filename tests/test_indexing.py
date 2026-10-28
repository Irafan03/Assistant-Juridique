"""Tests du LegalIndex avec un EmbeddingModel bouchonné (sans GPU/réseau)."""

from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path

from legal_rag.models import Chunk
from legal_rag.indexing import LegalIndex, StrategyIndex, EmbeddingModel


# ---------------------------------------------------------------------------
# Stub d'embedding — vecteurs aléatoires reproductibles, sans SentenceTransformer
# ---------------------------------------------------------------------------

class StubEmbeddingModel(EmbeddingModel):
    """Retourne des vecteurs L2-normalisés aléatoires sans charger de modèle."""

    DIM = 32

    def __init__(self) -> None:
        super().__init__(model_name="stub")
        self._rng = np.random.default_rng(42)

    def encode(self, texts, *, batch_size=64, show_progress=False) -> np.ndarray:
        vecs = self._rng.random((len(texts), self.DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)

    @property
    def dimension(self) -> int:
        return self.DIM

    def _load(self):
        return self  # contourne le chargement réel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_chunks(n: int = 5, doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(
            document_id=doc_id,
            chunk_id=f"{doc_id}::chunk::{i}",
            text=f"Article {i} — Disposition juridique numéro {i}.",
            document_type="circulaire" if i % 2 == 0 else "loi",
            article_number=str(i),
            source_file=f"/corpus/{doc_id}.pdf",
            page_number=i,
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests StrategyIndex
# ---------------------------------------------------------------------------

class TestStrategyIndex:
    def _make_index(self, tmp_path: Path) -> StrategyIndex:
        return StrategyIndex(tmp_path, "structural", StubEmbeddingModel())

    def test_add_and_iter(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunks = _sample_chunks(4)
        idx.add_chunks(chunks)
        stored = idx.iter_chunks()
        assert len(stored) == 4

    def test_get_chunk(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunks = _sample_chunks(3)
        idx.add_chunks(chunks)
        retrieved = idx.get_chunk(chunks[1].chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunks[1].chunk_id
        assert retrieved.article_number == chunks[1].article_number

    def test_get_chunk_missing_returns_none(self, tmp_path):
        idx = self._make_index(tmp_path)
        assert idx.get_chunk("inexistant") is None

    def test_search_returns_chunks(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add_chunks(_sample_chunks(5))
        results = idx.search("disposition juridique", limit=3)
        assert len(results) <= 3
        assert all(isinstance(c, Chunk) for c in results)

    def test_no_duplicate_on_reindex(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunks = _sample_chunks(3)
        idx.add_chunks(chunks)
        idx.add_chunks(chunks)  # même chunks une 2e fois
        stored = idx.iter_chunks()
        ids = [c.chunk_id for c in stored]
        assert len(ids) == len(set(ids)), "Des doublons ont été créés"
        assert len(stored) == 3

    def test_parent_child_relation(self, tmp_path):
        idx = self._make_index(tmp_path)
        parent = Chunk("d", "parent-1", "Texte parent.", parent_id=None)
        child = Chunk("d", "child-1", "Texte enfant.", parent_id="parent-1")
        idx.add_chunks([parent, child])
        retrieved_parent = idx.get_parent(child)
        assert retrieved_parent is not None
        assert retrieved_parent.chunk_id == "parent-1"

    def test_get_parent_without_parent_id(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunk = Chunk("d", "orphan", "Sans parent.")
        idx.add_chunks([chunk])
        assert idx.get_parent(chunk) is None

    def test_metadata_preserved(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunk = Chunk(
            "d", "c1", "Texte.",
            references=("Loi n° 12-03", "Article 5"),
            metadata={"level": "article", "source": "auto"},
        )
        idx.add_chunks([chunk])
        stored = idx.get_chunk("c1")
        assert stored is not None
        assert "Loi n° 12-03" in stored.references
        assert stored.metadata.get("level") == "article"

    def test_search_empty_index_returns_empty(self, tmp_path):
        idx = self._make_index(tmp_path)
        assert idx.search("question") == []

    def test_persistence_across_instances(self, tmp_path):
        """L'index doit survivre à la recréation de l'objet (lecture depuis disque)."""
        idx1 = self._make_index(tmp_path)
        idx1.add_chunks(_sample_chunks(3))
        idx1.close()

        idx2 = self._make_index(tmp_path)
        stored = idx2.iter_chunks()
        assert len(stored) == 3


# ---------------------------------------------------------------------------
# Tests LegalIndex (façade multi-stratégie)
# ---------------------------------------------------------------------------

class TestLegalIndex:
    def _make_legal_index(self, tmp_path: Path) -> LegalIndex:
        return LegalIndex(tmp_path, embedding_model=StubEmbeddingModel())

    def test_strategies_independent(self, tmp_path):
        li = self._make_legal_index(tmp_path)
        chunks_s = _sample_chunks(3, "docA")
        chunks_h = _sample_chunks(5, "docB")
        li._indices["structural"].add_chunks(chunks_s)
        li._indices["hierarchical"].add_chunks(chunks_h)

        assert len(li.iter_chunks(strategy="structural")) == 3
        assert len(li.iter_chunks(strategy="hierarchical")) == 5
        assert len(li.iter_chunks(strategy="parent_child")) == 0

    def test_index_adapter_protocol(self, tmp_path):
        """Vérifie que LegalIndex respecte le contrat IndexAdapter de Prompt 2."""
        li = self._make_legal_index(tmp_path)
        chunks = _sample_chunks(4)
        li._indices["structural"].add_chunks(chunks)

        # search
        results = li.search("article juridique", strategy="structural", limit=2)
        assert isinstance(results, list)
        assert len(results) <= 2

        # iter_chunks
        all_chunks = li.iter_chunks(strategy="structural")
        assert len(all_chunks) == 4

        # get_chunk
        c = li.get_chunk(chunks[0].chunk_id, strategy="structural")
        assert c is not None and c.chunk_id == chunks[0].chunk_id

        # get_parent
        parent_chunk = Chunk("d", "parent-x", "Parent.")
        child_chunk = Chunk("d", "child-x", "Enfant.", parent_id="parent-x")
        li._indices["structural"].add_chunks([parent_chunk, child_chunk])
        p = li.get_parent(child_chunk, strategy="structural")
        assert p is not None and p.chunk_id == "parent-x"

    def test_invalid_strategy_raises(self, tmp_path):
        li = self._make_legal_index(tmp_path)
        try:
            li.search("test", strategy="inconnu", limit=5)
            assert False, "ValueError attendu"
        except ValueError:
            pass

    def test_get_by_document(self, tmp_path):
        li = self._make_legal_index(tmp_path)
        chunks_a = _sample_chunks(3, "docA")
        chunks_b = _sample_chunks(2, "docB")
        li._indices["structural"].add_chunks(chunks_a + chunks_b)
        results = li.get_by_document("docA", strategy="structural")
        assert len(results) == 3
        assert all(c.document_id == "docA" for c in results)
