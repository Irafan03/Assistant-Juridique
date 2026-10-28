"""Tests couvrant les 12 cas demandés dans la demande d'amélioration.

Ces tests couvrent :
1.  Recherche retournant plusieurs passages pertinents.
2.  Recherche ne retournant aucun passage.
3.  Réponse avec plusieurs catégories de sources.
4.  Catégories présentes mais certains tableaux vides.
5.  Résultat dupliqué dans plusieurs catégories.
6.  Question sur les fonds propres et la solvabilité.
7.  Question sur les comptes inactifs (pas de résultats prudentiels).
8.  Question hors périmètre corpus.
9.  Réponse générée avec citations.
10. Réponse insuffisante qui signale ses limites.
11. Cohérence compteurs interface / données API.
12. Comportement de chaque stratégie de chunking.
"""
from __future__ import annotations

import re
import pytest

from legal_rag.models import Candidate, Chunk, SearchResponse
from legal_rag.answer import (
    FaithfulAnswerGenerator, MetadataReranker,
    _key_terms, _overlap, _fold, _is_relevant,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures et helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(chunk_id, text, doc_type="loi", title="Titre", article_number="1",
                doc_id="doc1", refs=()):
    return Chunk(
        document_id=doc_id, chunk_id=chunk_id, text=text,
        document_type=doc_type, title=title, article_number=article_number,
        references=tuple(refs),
    )


def _make_candidate(chunk, score=0.5):
    return Candidate(
        chunk=chunk, score=score,
        semantic_score=score, lexical_score=None,
        exact_reference=False, relation="semantic_similarity",
        citation_id=f"[1]",
    )


gen = FaithfulAnswerGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Recherche retournant plusieurs passages pertinents
# ─────────────────────────────────────────────────────────────────────────────

def test_1_several_relevant_passages():
    """Vérifier que le générateur exploite plusieurs passages pertinents."""
    evidence = [
        _make_candidate(_make_chunk("c1", "Les établissements de crédit doivent disposer de fonds propres suffisants.", article_number="76")),
        _make_candidate(_make_chunk("c2", "Le ratio de solvabilité des établissements de crédit est fixé par Bank Al-Maghrib.", article_number="77")),
        _make_candidate(_make_chunk("c3", "Article 78 Les exigences en fonds propres s'appliquent à toutes les banques.", article_number="78")),
    ]
    query = "Quelles règles encadrent les fonds propres et la solvabilité ?"
    answer = gen.generate(query, evidence=evidence)
    assert answer is not None
    assert "ne permet pas d'établir une réponse fiable" not in answer
    # Doit contenir des citations [n]
    assert re.search(r"\[\d+\]", answer), "Réponse sans citation [n]"
    # Doit couvrir les passages pertinents
    assert "fonds propres" in answer.lower() or "solvabilité" in answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recherche ne retournant aucun passage
# ─────────────────────────────────────────────────────────────────────────────

def test_2_empty_passages():
    """Générateur avec une liste d'evidence vide → réponse insuffisante."""
    answer = gen.generate("Quelle est la recette de la tarte ?", evidence=[])
    assert "ne permet pas d'établir une réponse fiable" in answer


# ─────────────────────────────────────────────────────────────────────────────
# 3. Réponse avec plusieurs catégories de sources
# ─────────────────────────────────────────────────────────────────────────────

def test_3_multiple_source_categories():
    """SearchResponse avec passages, circulaires et textes_associes renseignés."""
    circ_chunk = _make_chunk("circ1", "Circulaire relative aux fonds propres.", doc_type="circulaire",
                              doc_id="doc-circ")
    law_chunk  = _make_chunk("law1",  "Loi sur la solvabilité des banques.",    doc_type="loi",
                              doc_id="doc-law")
    circ_cand = _make_candidate(circ_chunk)
    law_cand  = _make_candidate(law_chunk, score=0.3)
    resp = SearchResponse(
        query="solvabilité fonds propres",
        strategy="structural",
        retrieval_mode="hybrid",
        circulares=[circ_cand],
        associated_texts=[law_cand],
        articles=[],
        passages=[circ_cand, law_cand],
    )
    d = resp.to_dict()
    assert len(d["circulaires"]) == 1
    assert len(d["textes_associes"]) == 1
    assert len(d["passages"]) == 2
    # Chaque passage a les champs requis
    for p in d["passages"]:
        assert "document_id" in p
        assert "text" in p
        assert "score" in p


# ─────────────────────────────────────────────────────────────────────────────
# 4. Catégories présentes mais certains tableaux vides
# ─────────────────────────────────────────────────────────────────────────────

def test_4_some_categories_empty():
    """SearchResponse avec articles et textes_associes vides."""
    cand = _make_candidate(_make_chunk("c1", "Texte pertinent."))
    resp = SearchResponse("query", "structural", "semantic",
                          circulares=[], associated_texts=[], articles=[],
                          passages=[cand])
    d = resp.to_dict()
    assert d["circulaires"] == []
    assert d["textes_associes"] == []
    assert d["articles"] == []
    assert len(d["passages"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Résultat dupliqué dans plusieurs catégories
# ─────────────────────────────────────────────────────────────────────────────

def test_5_same_chunk_in_multiple_categories():
    """Un même chunk peut apparaître dans passages ET circulaires (comportement actuel du moteur).
    Le frontend doit déduplicer pour les compteurs (chunk_id) mais les deux tableaux
    peuvent légitimement contenir le même chunk."""
    circ_chunk = _make_chunk("c1", "Circulaire fonds propres.", doc_type="circulaire")
    cand = _make_candidate(circ_chunk)
    resp = SearchResponse("query", "structural", "hybrid",
                          circulares=[cand], associated_texts=[], articles=[],
                          passages=[cand])
    d = resp.to_dict()
    # Le même chunk_id apparaît dans passages et circulaires
    passage_ids  = {p["chunk_id"] for p in d["passages"]}
    circulaire_ids = {p["chunk_id"] for p in d["circulaires"]}
    assert passage_ids == circulaire_ids  # même chunk, c'est attendu
    # Compteur frontend (unique par chunk_id) = 1, pas 2
    all_ids = {p["chunk_id"] for cat in (d["passages"], d["circulaires"],
                                          d["textes_associes"], d["articles"])
               for p in cat}
    assert len(all_ids) == 1, "Compteur doit déduplicer par chunk_id"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Question sur les fonds propres et la solvabilité
# ─────────────────────────────────────────────────────────────────────────────

def test_6_fonds_propres_question():
    """Passages contenant 'fonds propres' et 'solvabilité' → pertinents."""
    chunk = _make_chunk("c1",
        "Article 76 Afin de préserver leur liquidité et leur solvabilité, "
        "les établissements de crédit sont tenus de respecter les exigences en fonds propres.",
        article_number="76", title="Loi n°51-20"
    )
    evidence = [_make_candidate(chunk)]
    query = "Quelles règles prudentielles encadrent les fonds propres et la solvabilité ?"
    answer = gen.generate(query, evidence=evidence)
    assert "ne permet pas d'établir une réponse fiable" not in answer
    assert "[1]" in answer  # cité
    assert "76" in answer   # article cité


# ─────────────────────────────────────────────────────────────────────────────
# 7. Question sur les comptes inactifs (pas de résultats prudentiels)
# ─────────────────────────────────────────────────────────────────────────────

def test_7_comptes_inactifs_no_prudential_contamination():
    """'Comptes inactifs' → pertinent, 'solvabilité' → non pertinent pour cette question."""
    comptes_chunk = _make_chunk("c1", "Article 12 Les comptes inactifs sont clôturés après 10 ans.",
                                article_number="12")
    fonds_chunk   = _make_chunk("c2", "Article 76 Afin de préserver la solvabilité les établissements "
                                      "sont tenus de respecter les ratios prudentiels.",
                                article_number="76")
    evidence = [_make_candidate(comptes_chunk), _make_candidate(fonds_chunk, score=0.3)]
    query = "Quelles sont les règles relatives aux comptes inactifs ?"
    key_terms = _key_terms(query)
    # Overlap pour comptes inactifs
    ov_comptes = _overlap(comptes_chunk, key_terms)
    ov_fonds   = _overlap(fonds_chunk,   key_terms)
    assert ov_comptes > ov_fonds, (
        f"Chunk 'comptes inactifs' devrait avoir plus d'overlap ({ov_comptes}) "
        f"que 'solvabilité' ({ov_fonds}) pour la question comptes inactifs"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Question hors périmètre corpus
# ─────────────────────────────────────────────────────────────────────────────

def test_8_out_of_scope_question():
    """Question hors périmètre → réponse insuffisante."""
    # Evidence sans aucun terme de la question
    chunk = _make_chunk("c1", "Article 1 Les établissements de crédit sont soumis à agrément.")
    evidence = [_make_candidate(chunk)]
    answer = gen.generate("Quelle est la recette de la tarte aux pommes ?", evidence=evidence)
    assert "ne permet pas d'établir une réponse fiable" in answer


# ─────────────────────────────────────────────────────────────────────────────
# 9. Réponse générée avec citations
# ─────────────────────────────────────────────────────────────────────────────

def test_9_answer_contains_valid_citations():
    """Toutes les citations [n] dans la réponse doivent pointer vers des passages existants."""
    evidence = [
        _make_candidate(_make_chunk("c1", "Les fonds propres des établissements de crédit.", article_number="10")),
        _make_candidate(_make_chunk("c2", "La solvabilité est vérifiée annuellement.", article_number="11")),
    ]
    for i, cand in enumerate(evidence, 1):
        from dataclasses import replace
        evidence[i-1] = replace(cand, citation_id=f"[{i}]")
    query = "Quelles règles encadrent les fonds propres et la solvabilité ?"
    answer = gen.generate(query, evidence=evidence)
    # Les citations doivent toutes être valides (n dans [1, len(evidence)])
    citations = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}
    allowed = set(range(1, len(evidence) + 1))
    invalid = citations - allowed
    assert not invalid, f"Citations invalides : {invalid}"
    assert citations, "La réponse doit contenir au moins une citation"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Réponse insuffisante qui signale ses limites
# ─────────────────────────────────────────────────────────────────────────────

def test_10_insufficient_answer_signals_limits():
    """Une réponse insuffisante doit explicitement signaler ses limites."""
    answer = gen.generate("Quelle est la température de fusion du diamant ?", evidence=[])
    assert "ne permet pas d'établir une réponse fiable" in answer
    # Doit contenir une explication, pas juste un code d'erreur
    assert len(answer) > 80


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cohérence compteurs interface / données API
# ─────────────────────────────────────────────────────────────────────────────

def test_11_counter_consistency():
    """Simuler la logique frontend : comptage unique par chunk_id."""
    c1 = _make_chunk("chunk-A", "Texte A.", doc_type="circulaire", doc_id="doc1")
    c2 = _make_chunk("chunk-B", "Texte B.", doc_type="loi",        doc_id="doc2")
    cA = _make_candidate(c1)
    cB = _make_candidate(c2, score=0.3)
    resp = SearchResponse("q", "structural", "hybrid",
                          circulares=[cA], associated_texts=[cB], articles=[cA],
                          passages=[cA, cB])
    d = resp.to_dict()
    # Logique frontend : union de tous les chunk_ids
    all_ids = {p["chunk_id"]
               for cat_key in ("passages", "circulaires", "textes_associes", "articles")
               for p in d.get(cat_key, [])}
    # 2 chunks distincts même si chunk-A apparaît dans passages, circulaires, et articles
    assert len(all_ids) == 2, f"Attendu 2 chunks distincts, trouvé {len(all_ids)}: {all_ids}"
    # Compteur documents distincts
    doc_ids = {p["document_id"]
               for cat_key in ("passages", "circulaires", "textes_associes", "articles")
               for p in d.get(cat_key, [])}
    assert len(doc_ids) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 12. Comportement de chaque stratégie de chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestAllStrategies:
    """Vérifie que les 3 stratégies retournent des SearchResponse valides."""

    STRATEGIES = ("structural", "hierarchical", "parent_child")

    def _fake_retriever(self, strategy):
        """Crée un retriever minimal pour la stratégie donnée."""
        from legal_rag.retrieval import LegalRetriever

        chunk = _make_chunk(f"{strategy}_chunk", "Article 76 Les fonds propres.",
                            article_number="76", title="Loi test")

        class FakeIndex:
            def search(self, q, *, strategy, limit):
                return [chunk]
            def iter_chunks(self, *, strategy):
                return [chunk]
            def get_chunk(self, cid, *, strategy):
                return chunk if cid == chunk.chunk_id else None
            def get_parent(self, c, *, strategy):
                return None

        return LegalRetriever(FakeIndex(), answer_generator=FaithfulAnswerGenerator())

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_strategy_returns_valid_response(self, strategy):
        retr = self._fake_retriever(strategy)
        resp = retr.search("fonds propres solvabilité", strategy=strategy,
                            retrieval_mode="semantic", limit=5, generate_answer=True)
        assert isinstance(resp.passages, list)
        assert isinstance(resp.circulares, list)
        d = resp.to_dict()
        assert "passages" in d
        assert "warnings" in d
        # Vérifier que les champs requis sont présents
        if d["passages"]:
            p = d["passages"][0]
            for field in ("document_id", "chunk_id", "text", "score", "relation"):
                assert field in p

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_strategy_answer_generated(self, strategy):
        retr = self._fake_retriever(strategy)
        resp = retr.search("fonds propres solvabilité établissements crédit",
                            strategy=strategy, retrieval_mode="semantic",
                            limit=5, generate_answer=True)
        # Avec l'evidence contenant "fonds propres" → réponse pertinente
        assert resp.answer is not None
        assert "ne permet pas d'établir une réponse fiable" not in resp.answer


# ─────────────────────────────────────────────────────────────────────────────
# Tests unitaires du générateur (overlap, fold, key_terms, is_relevant)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerInternals:
    def test_fold_removes_accents(self):
        assert _fold("Solvabilité") == "solvabilite"
        assert _fold("ÉTABLISSEMENTS") == "etablissements"
        assert _fold("Fonds") == "fonds"

    def test_key_terms_removes_stopwords(self):
        terms = _key_terms("Quelles sont les règles relatives aux établissements ?")
        assert "quelles" not in terms
        assert "sont" not in terms
        assert "regles" in terms or "règles" not in terms  # normalisé sans accents

    def test_overlap_word_boundary(self):
        """'recette' dans la question ne doit pas matcher 'recettes' dans le corpus."""
        chunk = _make_chunk("c1", "Les recettes fiscales sont versées au Trésor.")
        terms = _key_terms("Quelle est la recette de la tarte ?")
        assert _overlap(chunk, terms) == 0, (
            "Faux positif : 'recette' (question) ne doit pas matcher 'recettes' (corpus)"
        )

    def test_overlap_accent_insensitive(self):
        """'solvabilite' (sans accent) doit matcher 'solvabilité' (avec accent)."""
        chunk = _make_chunk("c1", "La solvabilité des banques est vérifiée annuellement.")
        terms = _key_terms("solvabilite fonds propres")  # sans accent
        ov = _overlap(chunk, terms)
        assert ov >= 1, f"'solvabilite' devrait matcher 'solvabilité', overlap={ov}"

    def test_is_relevant_requires_two_terms(self):
        """Un seul terme ne suffit pas (sauf question courte)."""
        assert not _is_relevant(1, 5), "1 terme sur 5 ne suffit pas"
        assert _is_relevant(2, 5),     "2 termes sur 5 suffisent"
        assert _is_relevant(1, 1),     "1 terme sur 1 (question courte) suffit"

    def test_no_citation_invention(self):
        """La réponse générée ne doit pas inventer de citation hors de l'evidence."""
        evidence = [_make_candidate(_make_chunk("c1", "Texte légal sur les fonds propres."))]
        answer = gen.generate("règles fonds propres établissements crédit", evidence=evidence)
        citations = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}
        # Les citations doivent être dans [1, len(evidence)]
        assert all(1 <= n <= len(evidence) for n in citations), (
            f"Citations inventées : {citations - set(range(1, len(evidence) + 1))}"
        )
