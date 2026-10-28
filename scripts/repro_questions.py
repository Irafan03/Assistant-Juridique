"""Investigation : comptes inactifs, hors-corpus, et structure JSON complète."""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

index = LegalIndex(INDEX)
retriever = LegalRetriever(index)

# Prechauffage
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

QUESTIONS = [
    ("fonds propres", "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?"),
    ("comptes inactifs", "Quelles sont les règles relatives aux comptes inactifs ?"),
    ("hors-corpus", "Quelle est la recette de la tarte aux pommes ?"),
]

for label, q in QUESTIONS:
    t0 = time.perf_counter()
    resp = retriever.search(q, strategy="structural", retrieval_mode="hybrid", limit=10)
    elapsed = time.perf_counter() - t0
    d = resp.to_dict()
    print(f"\n{'='*70}")
    print(f"[{label}] {q[:60]}")
    print(f"  {elapsed:.1f}s | passages={len(d['passages'])} circ={len(d['circulaires'])} "
          f"assoc={len(d['textes_associes'])} art={len(d['articles'])} warnings={len(d['warnings'])}")
    print(f"  answer={d['answer']!r}")
    for w in d['warnings']:
        print(f"  ⚠ {w}")
    print(f"  Top 3:")
    for i, p in enumerate(d['passages'][:3], 1):
        print(f"    [{i}] score={p['score']:.4f} type={p['document_type']} art={p['article_number']} "
              f"refs={len(p['references'])} | {(p['title'] or '')[:50]}")
        print(f"        {p['text'][:90].replace(chr(10),' ')}")

# Structure JSON complete d un passage
print(f"\n{'='*70}")
print("Structure JSON d'un passage (toutes les cles):")
d = retriever.search(QUESTIONS[0][1], strategy="structural", retrieval_mode="hybrid", limit=1).to_dict()
if d['passages']:
    print(json.dumps(d['passages'][0], ensure_ascii=False, indent=2)[:800])

index.close()
