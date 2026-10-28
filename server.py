"""Point d'entrée du serveur — connecte l'index réel au backend FastAPI.

Usage :
    uvicorn server:app --reload

Ce module charge l'index FAISS + SQLite construit par Prompt 1 dans
``data/index/`` et le branche sur le moteur de recherche hybride de Prompt 2.
Le frontend (dossier ``frontend/``) est servi automatiquement par FastAPI.
"""
from __future__ import annotations

from pathlib import Path

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.api import create_app
from legal_rag.answer import FaithfulAnswerGenerator  # reranker mesuré non bénéfique, non branché
from legal_rag.postprocess import process_response


class PostProcessedRetriever:
    """Enveloppe un LegalRetriever et applique le post-traitement (nettoyage des
    numéros d'article, déduplication, re-classement par pertinence juridique)
    à chaque SearchResponse, sans modifier retrieval.py ni l'index.

    Expose ``.search()`` et ``.index`` pour rester compatible avec l'API
    FastAPI (qui appelle ``engine.search`` et ``engine.index.iter_chunks``).
    """

    def __init__(self, retriever: LegalRetriever) -> None:
        self._retriever = retriever
        # ``.index`` reste accessible pour les endpoints /api/documents
        self.index = retriever.index

    def search(self, query: str, *, strategy: str = "structural",
               retrieval_mode: str = "hybrid", limit: int = 10,
               generate_answer: bool = False):
        # On désactive la génération côté retrieval.py (generate_answer=False) :
        # on la fait NOUS-MÊMES après re-classement, pour garantir la cohérence
        # entre citations [n] et l'ordre final des passages.
        response = self._retriever.search(
            query, strategy=strategy, retrieval_mode=retrieval_mode,
            limit=limit, generate_answer=False,
        )
        cleaned = process_response(response, query)

        if generate_answer and self._retriever.answer_generator is not None and cleaned.passages:
            from dataclasses import replace
            regen = self._retriever.answer_generator.generate(query, evidence=cleaned.passages)
            cleaned = replace(cleaned, answer=regen)
        return cleaned

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INDEX_DIR  = BASE_DIR / "data" / "index"
CORPUS_DIR = BASE_DIR  # contient le PDF du recueil


def build_app():
    """Construit l'application FastAPI avec l'index réel.

    Si l'index n'existe pas encore (premier lancement sans indexation),
    l'application démarre quand même mais /api/search retourne 503 avec un
    message indiquant qu'il faut lancer l'indexation.
    """
    if not INDEX_DIR.is_dir() or not (INDEX_DIR / "structural" / "metadata.db").exists():
        print("[server] Aucun index trouvé dans data/index/.")
        print("[server] Lancez d'abord : python scripts/index_bam.py")
        print("[server] Démarrage sans index (l'API retournera 503).")
        return create_app(document_root=str(CORPUS_DIR))

    print(f"[server] Chargement de l'index : {INDEX_DIR}")
    index     = LegalIndex(INDEX_DIR)
    base_retriever = LegalRetriever(
        index,
        answer_generator=FaithfulAnswerGenerator(),  # synthèse fidèle avec citations [n]
    )
    retriever = PostProcessedRetriever(base_retriever)
    app       = create_app(retriever, document_root=str(CORPUS_DIR))
    print(f"[server] Index connecté. Frontend disponible sur http://127.0.0.1:8000")
    return app


app = build_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
