"""Reproduction du bug : question sur les fonds propres et la solvabilité."""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?"

index = LegalIndex(INDEX)
retriever = LegalRetriever(index)

for strategy in ("structural",):
    for mode in ("semantic", "hybrid"):
        t0 = time.perf_counter()
        resp = retriever.search(QUESTION, strategy=strategy, retrieval_mode=mode, limit=10)
        elapsed = time.perf_counter() - t0
        d = resp.to_dict()
        print(f"\n{'='*70}")
        print(f"strategy={strategy} mode={mode} limit=10  ({elapsed:.1f}s)")
        print(f"  passages        : {len(d['passages'])}")
        print(f"  circulaires     : {len(d['circulaires'])}")
        print(f"  textes_associes : {len(d['textes_associes'])}")
        print(f"  articles        : {len(d['articles'])}")
        print(f"  answer          : {d['answer']!r}")
        print(f"  warnings        : {d['warnings']}")
        print(f"\n  --- Top 5 passages ---")
        for i, p in enumerate(d['passages'][:5], 1):
            print(f"  [{i}] {p['citation_id']} score={p['score']:.4f} type={p['document_type']} "
                  f"art={p['article_number']} rel={p['relation']}")
            print(f"      title: {(p['title'] or '')[:70]}")
            print(f"      text : {p['text'][:100].replace(chr(10),' ')}")

index.close()
