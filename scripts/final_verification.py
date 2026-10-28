"""Vérification finale : question exacte du ticket sur l'index réel."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?"

index = LegalIndex(INDEX)
retriever = LegalRetriever(index, answer_generator=FaithfulAnswerGenerator())

for s in ("structural", "hierarchical", "parent_child"):
    _ = list(index.iter_chunks(strategy=s))

print("=" * 78)
print("QUESTION DE TEST (ticket)")
print("=" * 78)
print(QUESTION)
print()

for strategy in ("structural", "hierarchical", "parent_child"):
    for mode in ("hybrid", "semantic"):
        t0 = time.perf_counter()
        resp = retriever.search(QUESTION, strategy=strategy, retrieval_mode=mode,
                                limit=10, generate_answer=True)
        elapsed = time.perf_counter() - t0
        d = resp.to_dict()
        print(f"\n--- strategy={strategy} mode={mode}  ({elapsed:.1f}s) ---")
        print(f"passages={len(d['passages'])} circulaires={len(d['circulaires'])} "
              f"textes_associes={len(d['textes_associes'])} articles={len(d['articles'])}")
        if d['warnings']:
            for w in d['warnings']:
                print(f"  ⚠ {w}")
        if d['answer']:
            insufficient = "ne permet pas d'établir une réponse fiable" in d['answer']
            print(f"  RÉPONSE ({'insuffisante' if insufficient else 'synthèse'}):")
            print(d['answer'][:500])
        # Top 3 passages
        print(f"  Top 3 passages :")
        for i, p in enumerate(d['passages'][:3], 1):
            print(f"    [{i}] {p['citation_id']} {p['document_type']} art={p['article_number']} "
                  f"score={p['score']:.4f} rel={p['relation']} | {(p['title'] or '')[:45]}")

index.close()
