"""Tests de cohérence : index réel, article numbers, déduplication, concepts AML."""
from __future__ import annotations

import pathlib
import re
import sqlite3

import numpy as np
import pytest

from legal_rag.indexing import LegalIndex, EmbeddingModel, StrategyIndex
from legal_rag.models import Chunk
from legal_rag.chunking import _ARTICLE_RE
from legal_rag.postprocess import (
    clean_article_number, deduplicate_passages, rerank_passages, process_response,
)
from legal_rag.concepts import is_aml_query, expand_query_terms, detect_themes, _matches_any
from legal_rag.answer import _key_terms, _overlap, _fold
from legal_rag.models import Candidate, SearchResponse

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
INDEX_EXISTS = INDEX.exists() and (INDEX / "structural" / "metadata.db").exists()


# ── Stub embedding ───────────────────────────────────────────────────────────
class StubEmbed(EmbeddingModel):
    DIM = 384
    def __init__(self):
        super().__init__("stub")
        self._rng = np.random.default_rng(7)
    def encode(self, texts, *, batch_size=64, show_progress=False):
        v = self._rng.random((len(texts), self.DIM)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-9)
    @property
    def dimension(self): return self.DIM
    def _load(self): return self


# ── Helpers ──────────────────────────────────────────────────────────────────
def conn_for(strategy: str):
    c = sqlite3.connect(str(INDEX / strategy / "metadata.db"), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def load_vecs(strategy: str) -> np.ndarray | None:
    faiss_path = INDEX / strategy / "faiss.index"
    npy_path   = INDEX / strategy / "faiss.npy"
    if faiss_path.exists():
        try:
            import faiss
            idx = faiss.read_index(str(faiss_path))
            vecs = np.zeros((idx.ntotal, idx.d), dtype=np.float32)
            idx.reconstruct_n(0, idx.ntotal, vecs)
            return vecs
        except Exception:
            pass
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    return None

def _make_cand(chunk_id, text, doc_id="doc1", doc_type="loi",
               article_number="1", title="Titre", score=0.5, refs=()):
    chunk = Chunk(doc_id, chunk_id, text, document_type=doc_type,
                  title=title, article_number=article_number,
                  references=tuple(refs))
    return Candidate(chunk, score=score, relation="semantic_similarity",
                     citation_id=f"[1]")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Numéros d'article — pattern ARTICLE_RE
# ══════════════════════════════════════════════════════════════════════════════

class TestArticleNumberParsing:
    """Vérifie que _ARTICLE_RE ne génère plus de numéros d'article aberrants."""

    def test_concatenated_page_number_now_extracts_article(self):
        """'Article 35254 Les actifs...' = Article 35 + note 254 → extrait '35'."""
        m = _ARTICLE_RE.search("\nArticle 35254 Les actifs éligibles")
        assert m is not None
        from legal_rag.chunking import _article_number_from_match
        assert _article_number_from_match(m) == "35"

    def test_rejects_pure_four_digit_without_separator(self):
        """Un article > 999 sans séparateur ET sans note collée reste rejeté."""
        # 'Article 12345texte' : 5 chiffres suivis de lettres, pas de séparateur
        # Le cas 2 exige un séparateur après les 3 chiffres de note.
        # Ici '12345' + 'texte' = pas de séparateur entre 45 et texte → peut matcher 12+345
        # mais 345 n'est pas une note valide... on accepte quand même '12' (heuristique).
        # Ce test vérifie juste qu'on n'accepte pas un faux article > 999 comme seul numéro.
        m = _ARTICLE_RE.search("\nArticle 1000 texte")
        if m:
            from legal_rag.chunking import _article_number_from_match
            num = _article_number_from_match(m)
            assert num is None or int(num) <= 999

    def test_article_16237_extracts_16(self):
        """'Article 16237' = Article 16 + note 237 → extrait '16'."""
        m = _ARTICLE_RE.search("\nArticle 16237 texte")
        assert m is not None
        from legal_rag.chunking import _article_number_from_match
        assert _article_number_from_match(m) == "16"

    def test_accepts_three_digit_article(self):
        m = _ARTICLE_RE.search("\nArticle 999 texte")
        assert m is not None
        assert (m.group(1) or "").strip() == "999"

    def test_accepts_bis(self):
        m = _ARTICLE_RE.search("\narticle 4 bis texte")
        assert m is not None
        assert "4 bis" in (m.group(1) or "")

    def test_accepts_normal_articles(self):
        for n in ("1", "12", "76", "100"):
            m = _ARTICLE_RE.search(f"\nArticle {n} texte")
            assert m is not None, f"Article {n} devrait matcher"
            assert (m.group(1) or "").strip() == n


# ══════════════════════════════════════════════════════════════════════════════
# 2. clean_article_number (postprocess)
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanArticleNumber:
    def test_none_stays_none(self):
        assert clean_article_number(None) is None

    def test_empty_string(self):
        assert clean_article_number("") is None

    def test_valid_simple(self):
        assert clean_article_number("12") == "12"
        assert clean_article_number("76") == "76"
        assert clean_article_number("999") == "999"

    def test_valid_bis(self):
        assert clean_article_number("4 bis") == "4 bis"

    def test_valid_dash(self):
        assert clean_article_number("595-6") == "595-6"

    def test_rejects_over_999(self):
        assert clean_article_number("1000") is None
        assert clean_article_number("35254") is None
        assert clean_article_number("16237") is None

    def test_rejects_long_string(self):
        assert clean_article_number("Article 35 texte long") is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Déduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    def test_removes_identical_text_same_doc(self):
        text = "Article 35 Les actifs éligibles font l'objet de décotes."
        c1 = _make_cand("c1", text, score=0.8)
        c2 = _make_cand("c2", text, score=0.5)  # même texte, même doc, score inférieur
        result = deduplicate_passages([c1, c2])
        assert len(result) == 1
        assert result[0].chunk.chunk_id == "c1"  # garde le meilleur score

    def test_keeps_different_articles(self):
        c1 = _make_cand("c1", "Article 35 texte A")
        c2 = _make_cand("c2", "Article 36 texte B")
        result = deduplicate_passages([c1, c2])
        assert len(result) == 2

    def test_keeps_same_text_different_docs(self):
        # Avec la nouvelle dédup cross-document, même texte dans docs différents
        # → un seul passage conservé (comportement voulu).
        c1 = _make_cand("c1", "Texte identique", doc_id="docA")
        c2 = _make_cand("c2", "Texte identique", doc_id="docB")
        result = deduplicate_passages([c1, c2])
        # Cross-doc dédup : on garde le premier (meilleur score ex aequo)
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Reranking par pertinence
# ══════════════════════════════════════════════════════════════════════════════

class TestReranking:
    def test_aml_passage_ranks_above_irrelevant(self):
        """Passage sur le blanchiment doit dépasser un passage hors-sujet."""
        aml = _make_cand("aml", "Les établissements de crédit sont tenus de déclarer "
                         "les opérations suspectes liées au blanchiment de capitaux.",
                         score=0.02)
        irrelevant = _make_cand("irr", "Les actifs éligibles en tant que garantie "
                                "aux opérations de politique monétaire.",
                                score=0.03)  # score RRF supérieur, mais hors-sujet
        query = "obligations blanchiment capitaux établissements crédit"
        reranked = rerank_passages([irrelevant, aml], query)
        # Après re-classement, le passage AML doit passer devant
        assert reranked[0].chunk.chunk_id == "aml", (
            f"Le passage AML devrait être premier, mais {reranked[0].chunk.chunk_id!r} passe devant"
        )

    def test_stable_sort_preserves_rrf_for_equal_relevance(self):
        """Passages de même pertinence : ordre RRF initial préservé."""
        c1 = _make_cand("c1", "Texte juridique A", score=0.5)
        c2 = _make_cand("c2", "Texte juridique B", score=0.3)
        query = "texte juridique"  # les deux matchent également
        reranked = rerank_passages([c1, c2], query)
        # c1 avait un meilleur score RRF et overlap égal → reste premier
        assert reranked[0].chunk.chunk_id == "c1"

    def test_citation_ids_renumbered(self):
        """Après re-classement, les citations doivent être [1], [2], …"""
        candidates = [_make_cand(f"c{i}", f"Texte {i}") for i in range(3)]
        reranked = rerank_passages(candidates, "texte")
        for rank, c in enumerate(reranked, 1):
            assert c.citation_id == f"[{rank}]", f"citation_id devrait être [{rank}], got {c.citation_id}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Concepts AML
# ══════════════════════════════════════════════════════════════════════════════

class TestAMLConcepts:
    def test_detects_aml_query(self):
        assert is_aml_query("obligations lutte contre le blanchiment d'argent")
        assert is_aml_query("blanchiment de capitaux banques")
        assert not is_aml_query("conditions d'agrément des établissements de crédit")
        assert not is_aml_query("ratio de liquidité")

    def test_expands_query_for_aml(self):
        base = _key_terms("obligations blanchiment argent")
        expanded, is_domain, themes = expand_query_terms("blanchiment argent", base)
        assert is_domain is True
        assert len(expanded) > len(base)
        # Les termes AML clés doivent être inclus
        assert "vigilance" in expanded or "declaration" in expanded

    def test_no_expansion_for_non_aml(self):
        base = _key_terms("conditions agrément établissements")
        _, is_domain, _ = expand_query_terms("conditions agrément établissements crédit", base)
        assert is_domain is False

    def test_detect_themes_in_aml_text(self):
        text = _fold("Les établissements sont tenus de procéder à l'identification "
                     "et à la vérification de l'identité de leurs clients.")
        themes = detect_themes(text)
        assert "identification_client" in themes

    def test_detect_declaration_theme(self):
        text = _fold("Les établissements déclarent les opérations suspectes "
                     "à l'autorité compétente.")
        themes = detect_themes(text)
        assert "declaration" in themes


# ══════════════════════════════════════════════════════════════════════════════
# 6. process_response (pipeline complet)
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessResponse:
    def test_cleans_bad_article_numbers(self):
        text = "Article 35 Les établissements sont tenus de respecter les exigences prudentielles."
        cand = _make_cand("c1", text, article_number="35254")
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[], associated_texts=[], articles=[],
                              passages=[cand])
        cleaned = process_response(resp, "test")
        # Le passage n'est pas filtré (contenu réel) et l'article est nettoyé
        assert len(cleaned.passages) == 1
        assert cleaned.passages[0].chunk.article_number is None

    def test_deduplicates_passages(self):
        text = "Article 35 Les actifs éligibles sont soumis aux règles prudentielles applicables."
        c1 = _make_cand("c1", text, score=0.8)
        c2 = _make_cand("c2", text, score=0.5)  # doublon
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[], associated_texts=[], articles=[],
                              passages=[c1, c2])
        cleaned = process_response(resp, "actifs prudentiels")
        assert len(cleaned.passages) == 1

    def test_reranks_aml_passages(self):
        aml_text = ("Article 7 Déclaration des opérations suspectes liées au "
                    "blanchiment de capitaux auprès de l'autorité compétente.")
        irr_text = ("Article 30 Les actifs éligibles en tant que garantie "
                    "aux opérations de politique monétaire sont listés ici.")
        aml_cand = _make_cand("aml", aml_text, score=0.01)
        irr_cand = _make_cand("irr", irr_text, score=0.05)
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[], associated_texts=[], articles=[],
                              passages=[irr_cand, aml_cand])
        cleaned = process_response(resp, "obligations blanchiment capitaux")
        assert cleaned.passages[0].chunk.chunk_id == "aml"

    def test_citation_ids_updated_across_categories(self):
        text = ("Article 5 Les établissements sont tenus de procéder "
                "à la vigilance renforcée pour les clients à risque élevé.")
        cand = _make_cand("c1", text, doc_type="circulaire")
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[cand], associated_texts=[], articles=[cand],
                              passages=[cand])
        cleaned = process_response(resp, "blanchiment")
        assert len(cleaned.passages) >= 1
        expected_cid = cleaned.passages[0].citation_id
        if cleaned.circulares:
            assert cleaned.circulares[0].citation_id == expected_cid
        if cleaned.articles:
            assert cleaned.articles[0].citation_id == expected_cid

    def test_empty_passages_unchanged(self):
        resp = SearchResponse("q", "structural", "semantic",
                              circulares=[], associated_texts=[], articles=[],
                              passages=[])
        cleaned = process_response(resp, "test")
        assert cleaned.passages == []


# ══════════════════════════════════════════════════════════════════════════════
# 7. Tests sur l'index réel (skip si absent)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not INDEX_EXISTS, reason="Index réel absent")
class TestIndexCoherence:
    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_no_duplicate_chunk_ids(self, strategy):
        c = conn_for(strategy)
        dup = c.execute(
            "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*)>1)"
        ).fetchone()[0]
        c.close()
        assert dup == 0, f"{strategy}: {dup} doublons de chunk_id"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_no_duplicate_faiss_rows(self, strategy):
        c = conn_for(strategy)
        dup = c.execute(
            "SELECT COUNT(*) FROM (SELECT faiss_row FROM chunks GROUP BY faiss_row HAVING COUNT(*)>1)"
        ).fetchone()[0]
        c.close()
        assert dup == 0, f"{strategy}: {dup} faiss_rows dupliqués"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_faiss_rows_in_bounds(self, strategy):
        vecs = load_vecs(strategy)
        if vecs is None:
            pytest.skip("Vecteurs non lisibles")
        n_vecs = len(vecs)
        c = conn_for(strategy)
        out = c.execute(
            f"SELECT COUNT(*) FROM chunks WHERE faiss_row < 0 OR faiss_row >= {n_vecs}"
        ).fetchone()[0]
        c.close()
        assert out == 0, f"{strategy}: {out} faiss_rows hors borne"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_no_null_vectors(self, strategy):
        vecs = load_vecs(strategy)
        if vecs is None:
            pytest.skip("Vecteurs non lisibles")
        zero_vecs = int(np.sum(np.all(vecs == 0, axis=1)))
        assert zero_vecs == 0, f"{strategy}: {zero_vecs} vecteurs nuls"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_vectors_are_normalized(self, strategy):
        vecs = load_vecs(strategy)
        if vecs is None:
            pytest.skip("Vecteurs non lisibles")
        norms = np.linalg.norm(vecs, axis=1)
        low = int(np.sum(norms < 0.5))
        assert low == 0, f"{strategy}: {low} vecteurs avec norme < 0.5"

    def test_structural_orphan_vectors_are_inoffensive(self):
        c = conn_for("structural")
        vecs = load_vecs("structural")
        if vecs is None:
            pytest.skip()
        n_vecs = len(vecs)
        used_rows = set(r[0] for r in c.execute("SELECT faiss_row FROM chunks").fetchall())
        orphans = [i for i in range(n_vecs) if i not in used_rows]
        c.close()
        idx = LegalIndex(INDEX, embedding_model=StubEmbed())
        for orphan_row in orphans[:5]:
            c2 = sqlite3.connect(str(INDEX / "structural" / "metadata.db"), check_same_thread=False)
            c2.row_factory = sqlite3.Row
            row = c2.execute("SELECT * FROM chunks WHERE faiss_row = ?", (orphan_row,)).fetchone()
            c2.close()
            assert row is None, f"Row orphelin {orphan_row} pointe vers un chunk"
        idx.close()

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_no_inherited_recueil_title(self, strategy):
        c = conn_for(strategy)
        bad = c.execute(
            "SELECT COUNT(*) FROM chunks WHERE title LIKE '%RECUEIL DES TEXTES%'"
        ).fetchone()[0]
        c.close()
        assert bad == 0, f"{strategy}: {bad} chunks avec titre du recueil entier"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_all_chunks_have_document_type(self, strategy):
        c = conn_for(strategy)
        missing = c.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_type IS NULL"
        ).fetchone()[0]
        c.close()
        assert missing == 0, f"{strategy}: {missing} chunks sans document_type"

    @pytest.mark.parametrize("strategy", ["structural", "hierarchical", "parent_child"])
    def test_document_types_are_valid(self, strategy):
        VALID_TYPES = {"loi", "circulaire", "décret", "arrêté", "directive",
                       "recommandation", "texte"}
        c = conn_for(strategy)
        types = {r[0] for r in c.execute("SELECT DISTINCT document_type FROM chunks").fetchall()}
        c.close()
        unknown = types - VALID_TYPES
        assert len(unknown) == 0, f"{strategy}: types inconnus = {unknown}"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Cache iter_chunks
# ══════════════════════════════════════════════════════════════════════════════

class TestIterChunksCache:
    def _make_index(self, tmp_path):
        return LegalIndex(tmp_path, embedding_model=StubEmbed())

    def test_cache_returns_same_list(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunks = [Chunk(f"d{i}", f"c{i}", f"Texte {i}.") for i in range(5)]
        idx._indices["structural"].add_chunks(chunks)
        list1 = idx.iter_chunks(strategy="structural")
        list2 = idx.iter_chunks(strategy="structural")
        assert list1 is list2, "Le cache doit retourner le même objet liste"
        idx.close()

    def test_cache_invalidated_after_add(self, tmp_path):
        idx = self._make_index(tmp_path)
        c1 = Chunk("d1", "c1", "Texte 1.")
        idx._indices["structural"].add_chunks([c1])
        list1 = idx.iter_chunks(strategy="structural")
        assert len(list1) == 1
        c2 = Chunk("d2", "c2", "Texte 2.")
        idx._indices["structural"].add_chunks([c2])
        list2 = idx.iter_chunks(strategy="structural")
        assert list2 is not list1, "Le cache doit être invalidé après add_chunks"
        assert len(list2) == 2
        idx.close()

    def test_reindex_same_chunk_no_extra_vector(self, tmp_path):
        idx = self._make_index(tmp_path)
        chunk = Chunk("d1", "c1", "Texte stable.")
        idx._indices["structural"].add_chunks([chunk])
        si = idx._indices["structural"]
        n_vecs_before = si._load_faiss().ntotal
        n_chunks_before = len(idx.iter_chunks(strategy="structural"))
        idx._indices["structural"].add_chunks([chunk])
        n_vecs_after = si._load_faiss().ntotal
        n_chunks_after = len(idx.iter_chunks(strategy="structural"))
        assert n_vecs_after == n_vecs_before, "Réindexation ne doit pas ajouter de vecteur"
        assert n_chunks_after == n_chunks_before
        idx.close()


# ══════════════════════════════════════════════════════════════════════════════
# Tests priorité 1-5 : en-têtes, doublons, compteurs, génération AML
# ══════════════════════════════════════════════════════════════════════════════

from legal_rag.postprocess import (
    is_header_chunk, is_content_too_short, filter_passages,
    deduplicate_passages, _strip_header_lines,
)
from legal_rag.answer import FaithfulAnswerGenerator


HEADER_TEXT = (
    "RECUEIL DES TEXTES LEGISLATIFS ET REGLEMENTAIRES "
    "REGISSANT L'ACTIVITE DES ETABLISSEMENTS DE CREDIT "
    "ET ORGANISMES ASSIMILES\nMis à jour à fin 2023 639­ 3."
)


def _cand(chunk_id, text, doc_id="d1", doc_type="circulaire",
          article_number="1", score=0.5):
    c = Chunk(doc_id, chunk_id, text, document_type=doc_type,
              article_number=article_number)
    return Candidate(c, score=score, relation="semantic_similarity",
                     citation_id="[1]")


class TestHeaderFiltering:
    """Priorité 1 : filtrage des en-têtes et pieds de page."""

    def test_header_chunk_detected(self):
        c = Chunk("d1", "c1", HEADER_TEXT)
        assert is_header_chunk(c), "L'en-tête du recueil doit être détecté"

    def test_short_chunk_detected(self):
        c = Chunk("d1", "c2", "Article 3")
        assert is_content_too_short(c), "Chunk < 80 chars doit être détecté"

    def test_valid_obligation_not_filtered(self):
        text = ("Article 3 Les personnes assujetties sont tenues de mettre en place "
                "des politiques et procédures internes de lutte contre le blanchiment "
                "de capitaux et le financement du terrorisme.")
        c = Chunk("d1", "c3", text)
        assert not is_header_chunk(c)
        assert not is_content_too_short(c)

    def test_filter_passages_removes_headers(self):
        good = _cand("c1", "Article 5 Les établissements sont tenus d'identifier "
                     "leurs clients et de vérifier leur identité avant toute relation.")
        bad  = _cand("c2", HEADER_TEXT, article_number=None)
        kept, rejected = filter_passages([good, bad])
        assert len(kept) == 1
        assert kept[0].chunk.chunk_id == "c1"
        assert len(rejected) == 1

    def test_filter_passages_removes_short(self):
        short = _cand("c1", "Article 12")
        kept, rejected = filter_passages([short])
        assert len(kept) == 0
        assert len(rejected) == 1

    def test_strip_header_lines_preserves_content(self):
        text = ("RECUEIL DES TEXTES LEGISLATIFS ET REGLEMENTAIRES\n"
                "Article 5 Les établissements sont tenus d'identifier leurs clients.")
        cleaned = _strip_header_lines(text)
        assert "Article 5" in cleaned
        assert "RECUEIL" not in cleaned


class TestCrossDocumentDedup:
    """Priorité 2 : déduplication cross-document (loi 43-05 versions 2007/2021)."""

    def test_same_text_different_docs_deduped(self):
        """Article 31 copié dans loi 2007 et loi 2021 → ne compte qu'une fois."""
        text = ("Article 31 Afin de faciliter la coopération internationale en "
                "matière de blanchiment de capitaux, les dispositions des articles "
                "595-6 s'appliquent.")
        c1 = _cand("chunk_2007", text, doc_id="loi_2007_0001", score=0.016)
        c2 = _cand("chunk_2021", text, doc_id="loi_2021_0002", score=0.015)
        result = deduplicate_passages([c1, c2])
        assert len(result) == 1
        assert result[0].chunk.chunk_id == "chunk_2007"  # meilleur score conservé

    def test_different_articles_same_doc_kept(self):
        c1 = _cand("c1", "Article 5 Les établissements sont tenus d'identifier.")
        c2 = _cand("c2", "Article 6 Les mesures de vigilance sont renforcées.", article_number="6")
        result = deduplicate_passages([c1, c2])
        assert len(result) == 2

    def test_old_dedup_kept_same_doc_same_text(self):
        """Même doc, même texte (chevauchement) → fusion comme avant."""
        text = "Article 5 Les établissements sont tenus de procéder à l'identification."
        c1 = _cand("c1", text, score=0.8)
        c2 = _cand("c2", text, score=0.5)  # même doc implicite (d1)
        result = deduplicate_passages([c1, c2])
        assert len(result) == 1
        assert result[0].chunk.chunk_id == "c1"


class TestAMLGeneration:
    """Priorité 3 : génération sans gabarit, extraits réels, pas d'en-têtes."""

    gen = FaithfulAnswerGenerator()

    def _evidence_from_text(self, texts_and_arts):
        evidence = []
        for i, (text, art, doc_type) in enumerate(texts_and_arts, 1):
            c = Chunk(f"doc{i}", f"c{i}", text, document_type=doc_type,
                      article_number=art, title=f"Circulaire AML {i}")
            evidence.append(Candidate(c, score=0.5, relation="semantic_similarity",
                                      citation_id=f"[{i}]"))
        return evidence

    def test_no_header_in_generated_answer(self):
        """Un passage d'en-tête ne doit jamais apparaître dans la réponse."""
        evidence = [
            _cand("c_header", HEADER_TEXT, article_number=None),
            _cand("c_real",
                  "Article 5 Les établissements sont tenus d'identifier leurs clients "
                  "et de vérifier leur identité conformément aux dispositions de la loi.",
                  article_number="5"),
        ]
        query = "obligations blanchiment capitaux banques"
        answer = self.gen.generate(query, evidence=evidence)
        assert "RECUEIL DES TEXTES" not in (answer or ""), \
            "L'en-tête du recueil ne doit jamais apparaître dans la réponse"
        assert "Mis à jour à fin" not in (answer or "")

    def test_no_duplicate_citation_in_theme(self):
        """Le même extrait ne doit pas apparaître deux fois sous un thème."""
        text = ("Article 5 Les établissements sont tenus d'identifier leurs clients "
                "et de vérifier leur identité avant l'entrée en relation d'affaires.")
        # Deux passages quasi-identiques (doublon cross-doc)
        evidence = self._evidence_from_text([
            (text, "5", "circulaire"),
            (text, "5", "loi"),  # même contenu, autre doc
        ])
        query = "obligations blanchiment identification client"
        answer = self.gen.generate(query, evidence=evidence)
        # Ne doit pas apparaître en double sous le même thème
        if answer:
            # Compter les occurrences du titre de l'article 5 dans la réponse
            count = answer.count("Article 5") + answer.count("Art. 5")
            assert count <= 2, f"Doublon de citation : 'Article 5' apparaît {count} fois"

    def test_each_theme_backed_by_real_passage(self):
        """Chaque thème cité doit être justifié par un passage réel (non d'en-tête)."""
        evidence = self._evidence_from_text([
            ("Article 3 Les personnes assujetties sont tenues de mettre en place des "
             "politiques et procédures internes de lutte contre le blanchiment.",
             "3", "circulaire"),
            ("Article 7 Les établissements déclarent les opérations suspectes à "
             "l'Unité de traitement du renseignement financier.",
             "7", "loi"),
        ])
        query = "obligations blanchiment capitaux banques"
        answer = self.gen.generate(query, evidence=evidence)
        if answer and "ne permet pas" not in answer:
            # La réponse doit contenir au moins une citation [n]
            import re
            citations = re.findall(r"\[(\d+)\]", answer)
            assert citations, "La réponse thématique doit contenir des citations [n]"

    def test_obligation_sentence_extracted(self):
        """_obligation_sentence doit extraire la phrase avec un verbe d'obligation."""
        from legal_rag.answer import FaithfulAnswerGenerator
        gen = FaithfulAnswerGenerator()
        text = ("Introduction sans intérêt. "
                "Les établissements sont tenus de vérifier l'identité du client "
                "avant l'entrée en relation d'affaires.")
        result = gen._obligation_sentence(text)
        assert "sont tenus" in result.lower() or "vérifier" in result.lower(), \
            f"La phrase d'obligation n'a pas été extraite : {result!r}"

    def test_answer_empty_when_only_headers(self):
        """Si l'evidence ne contient que des en-têtes, réponse = insuffisante."""
        evidence = [_cand("c1", HEADER_TEXT, article_number=None)]
        query = "obligations blanchiment capitaux"
        answer = self.gen.generate(query, evidence=evidence)
        # Soit insuffisant, soit vide
        if answer:
            assert "ne permet pas" in answer or len(answer) < 50, \
                "Avec seulement des en-têtes, la réponse doit signaler l'insuffisance"


class TestCounterConsistency:
    """Priorité 4-5 : compteurs cohérents avec ce qui est affiché."""

    def test_counters_based_on_passages_only(self):
        """Les compteurs ne comptent pas les catégories dérivées (double-comptage)."""
        c1 = Chunk("d1", "chunk_A", "Art. 5 Identifier le client et vérifier son identité.",
                   document_type="circulaire")
        c2 = Chunk("d2", "chunk_B", "Art. 7 Déclarer les soupçons à l'autorité compétente.",
                   document_type="loi")
        cA = Candidate(c1, 0.5, relation="semantic_similarity", citation_id="[1]")
        cB = Candidate(c2, 0.3, relation="semantic_similarity", citation_id="[2]")
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[cA],         # cA dans circulaires
                              associated_texts=[cB],   # cB dans associés
                              articles=[cA, cB],       # les deux dans articles
                              passages=[cA, cB])       # 2 passages

        d = resp.to_dict()
        # Les compteurs doivent se baser sur passages uniquement
        passage_chunks = {p["chunk_id"] for p in d["passages"]}
        passage_docs   = {p["document_id"] for p in d["passages"]}
        assert len(passage_chunks) == 2, "2 passages uniques"
        assert len(passage_docs)   == 2, "2 documents distincts"

    def test_no_header_passage_in_response(self):
        """Après process_response, aucun passage d'en-tête ne subsiste."""
        good = _cand("c1", "Article 5 Les établissements sont tenus d'identifier "
                     "leurs clients et de vérifier leur identité complète.")
        bad  = _cand("c2", HEADER_TEXT, article_number=None)
        resp = SearchResponse("q", "structural", "hybrid",
                              circulares=[], associated_texts=[], articles=[],
                              passages=[good, bad])
        from legal_rag.postprocess import process_response
        cleaned = process_response(resp, "blanchiment")
        header_found = any(
            "RECUEIL DES TEXTES" in (p.chunk.text or "")
            for p in cleaned.passages
        )
        assert not header_found, "Aucun passage d'en-tête ne doit subsister après process_response"
        assert len(cleaned.passages) == 1
