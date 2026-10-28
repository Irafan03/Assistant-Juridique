"""Tests des trois stratégies de chunking."""

from __future__ import annotations

from legal_rag.chunking import chunk_document, chunk_structural, chunk_hierarchical, chunk_parent_child
from legal_rag.extraction import RawDocument, RawPage
from legal_rag.models import Chunk


def _make_doc(text: str, doc_type: str | None = "circulaire") -> RawDocument:
    """Crée un RawDocument minimal pour les tests."""
    pages = [RawPage(page_number=1, text=text)]
    return RawDocument(
        source_file="/test/doc.pdf",
        document_id="test_doc",
        pages=pages,
        full_text=text,
        detected_type=doc_type,
        metadata={"title": "Document test", "date": "01/01/2021", "_auto_keys": ["title", "date"]},
    )


_LEGAL_TEXT = """\
Circulaire n° 12 du Gouverneur de Bank Al-Maghrib

Chapitre I — Dispositions générales

Article 1
Les établissements de crédit sont tenus de respecter les présentes dispositions.

Article 2
Toute dérogation doit être approuvée par le Gouverneur conformément à la loi n° 34-03.

Chapitre II — Conditions d'application

Article 3
Les modalités d'application sont définies par arrêté.

Article 4 bis
Des dispositions transitoires peuvent être accordées jusqu'au 31/12/2022.
"""


class TestStructural:
    def test_produces_chunks(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_structural(doc)
        assert len(chunks) > 0, "Doit produire au moins un chunk"

    def test_all_chunks_are_chunk_instances(self):
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_structural(doc):
            assert isinstance(c, Chunk)

    def test_article_numbers_extracted(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_structural(doc)
        art_nums = {c.article_number for c in chunks if c.article_number}
        # Articles 1, 2, 3, 4 bis attendus
        assert len(art_nums) >= 2

    def test_chunk_ids_are_unique(self):
        doc = _make_doc(_LEGAL_TEXT)
        ids = [c.chunk_id for c in chunk_structural(doc)]
        assert len(ids) == len(set(ids)), "Les chunk_id doivent être uniques"

    def test_document_id_propagated(self):
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_structural(doc):
            assert c.document_id == "test_doc"

    def test_references_extracted(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_structural(doc)
        all_refs = [ref for c in chunks for ref in c.references]
        # "loi n° 34-03" doit être capturée dans le chunk de l'Article 2
        assert any("34" in r for r in all_refs), f"Références trouvées : {all_refs}"

    def test_source_file_preserved(self):
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_structural(doc):
            assert c.source_file == "/test/doc.pdf"

    def test_max_chars_respected(self):
        """Aucun chunk ne doit dépasser 2× max_chars (la subdivision peut produire des restes)."""
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_structural(doc, max_chars=200):
            assert len(c.text) <= 400, f"Chunk trop long : {len(c.text)} chars"

    def test_no_verified_metadata_auto(self):
        """Les chunks auto-extraits ne doivent pas avoir verified_metadata."""
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_structural(doc):
            assert c.verified_metadata == ()


class TestHierarchical:
    def test_has_root_chunk(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_hierarchical(doc)
        roots = [c for c in chunks if c.metadata.get("level") == "doc"]
        assert len(roots) == 1

    def test_parent_child_relations(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_hierarchical(doc)
        ids = {c.chunk_id for c in chunks}
        for c in chunks:
            if c.parent_id:
                assert c.parent_id in ids, f"parent_id {c.parent_id!r} introuvable"

    def test_levels_present(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_hierarchical(doc)
        levels = {c.metadata.get("level") for c in chunks}
        assert "doc" in levels
        assert "section" in levels or "article" in levels

    def test_unique_ids(self):
        doc = _make_doc(_LEGAL_TEXT)
        ids = [c.chunk_id for c in chunk_hierarchical(doc)]
        assert len(ids) == len(set(ids))


class TestParentChild:
    def test_parents_and_children(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_parent_child(doc)
        parents = [c for c in chunks if c.metadata.get("level") == "parent"]
        children = [c for c in chunks if c.metadata.get("level") == "child"]
        assert len(parents) > 0
        assert len(children) > 0

    def test_children_reference_parents(self):
        doc = _make_doc(_LEGAL_TEXT)
        chunks = chunk_parent_child(doc)
        parent_ids = {c.chunk_id for c in chunks if c.metadata.get("level") == "parent"}
        for c in chunks:
            if c.metadata.get("level") == "child":
                assert c.parent_id in parent_ids

    def test_unique_ids(self):
        doc = _make_doc(_LEGAL_TEXT)
        ids = [c.chunk_id for c in chunk_parent_child(doc)]
        assert len(ids) == len(set(ids))

    def test_child_size_bounded(self):
        doc = _make_doc(_LEGAL_TEXT)
        for c in chunk_parent_child(doc, child_chars=300):
            if c.metadata.get("level") == "child":
                # Un enfant peut légèrement dépasser child_chars à cause des limites de mots
                assert len(c.text) <= 300 + 50


class TestDispatch:
    def test_unknown_strategy_raises(self):
        doc = _make_doc(_LEGAL_TEXT)
        try:
            chunk_document(doc, strategy="inconnue")
            assert False, "ValueError attendu"
        except ValueError as exc:
            assert "inconnue" in str(exc)

    def test_all_strategies_produce_chunks(self):
        doc = _make_doc(_LEGAL_TEXT)
        for strategy in ("structural", "hierarchical", "parent_child"):
            chunks = chunk_document(doc, strategy)
            assert len(chunks) > 0, f"Aucun chunk pour {strategy}"


class TestEdgeCases:
    def test_empty_text(self):
        doc = _make_doc("")
        for strategy in ("structural", "hierarchical", "parent_child"):
            chunks = chunk_document(doc, strategy)
            assert isinstance(chunks, list)

    def test_single_article(self):
        doc = _make_doc("Article 1\nUne seule disposition.")
        chunks = chunk_structural(doc)
        assert len(chunks) >= 1
