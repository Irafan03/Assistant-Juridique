"""Mesure avant/après : effet du reranker et du la génération de réponse fidèle."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import MetadataReranker, FaithfulAnswerGenerator, _key_terms, _overlap

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

QUESTIONS = [
    ("fonds-propres",
     "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?",
     "Dahir n°1-21-77"),  # doit contenir article 76 (liquidité, solvabilité, fonds propres)
    ("comptes-inactifs",
     "Quelles sont les règles relatives aux comptes inactifs ?",
     "comptes"),
    ("hors-corpus",
     "Quelle est la recette de la tarte aux pommes ?",
     None),  # aucune pertinence attendue
]

index = LegalIndex(INDEX)

# Préchauffage cache
for s in ("structural", "hierarchical", "parent_child"):
    _ = list(index.iter_chunks(strategy=s))

# Retriever SANS reranker (baseline)
retriever_base = LegalRetriever(index)
# Retriever AVEC reranker + générateur
retriever_fix  = LegalRetriever(index,
                                reranker=MetadataReranker(),
                                answer_generator=FaithfulAnswerGenerator())

print("=" * 78)
print("MESURE AVANT / APRÈS — strategy=structural, mode=hybrid, limit=10")
print("=" * 78)

for label, question, expected_doc in QUESTIONS:
    print(f"\n{'─'*78}")
    print(f"[{label}] {question[:70]}")
    print(f"{'─'*78}")
    key_terms = _key_terms(question)
    print(f"  termes clés : {key_terms[:8]}")

    for name, retriever, gen in [("AVANT (base)", retriever_base, False),
                                 ("APRÈS (reranker+gen)", retriever_fix, True)]:
        t0 = time.perf_counter()
        resp = retriever.search(question, strategy="structural",
                                retrieval_mode="hybrid", limit=10, generate_answer=gen)
        elapsed = time.perf_counter() - t0

        # Pertinence du top-1
        top = resp.passages[0] if resp.passages else None
        top_overlap = _overlap(top.chunk, key_terms) if top else 0
        top_art = top.chunk.article_number if top else "—"
        top_title = (top.chunk.title or "")[:45] if top else ""

        # Combien de passages dans le top 5 ont un chevauchement > 0 ?
        top5_relevant = sum(1 for p in resp.passages[:5] if _overlap(p.chunk, key_terms) > 0)

        # Article 76 present dans le top 5 ? (pour la question fonds propres)
        art76_rank = next((i+1 for i, p in enumerate(resp.passages)
                           if p.chunk.article_number == "76"
                           and "51-20" in (p.chunk.title or "")), None)

        print(f"  [{name}] {elapsed:.1f}s")
        print(f"    top-1 : art={top_art} overlap={top_overlap} | {top_title}")
        print(f"    top-5 pertinents (overlap>0) : {top5_relevant}/5")
        if label == "fonds-propres":
            print(f"    Article 76 (loi 51-20) rang : {art76_rank if art76_rank else 'absent du top 10'}")
        if gen and resp.answer:
            # Afficher un extrait de la réponse générée
            ans_preview = resp.answer[:300].replace("\n", " ")
            print(f"    RÉPONSE GÉNÉRÉE : {ans_preview}...")
        elif gen:
            print(f"    RÉPONSE : (masquée — {resp.warnings})")

index.close()
