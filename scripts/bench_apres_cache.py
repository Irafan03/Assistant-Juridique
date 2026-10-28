"""Benchmark post-cache : mesure du gain réel sur retriever.search()."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUERY = "conditions agrément établissements de crédit"

index     = LegalIndex(INDEX)
retriever = LegalRetriever(index)

print("Benchmark post-cache (modele deja charge)\n")
print(f"{'Config':<30} {'1er appel':>10} {'2eme appel':>10} {'gain':>8}")
print("-" * 62)

for strategy in ("structural", "hierarchical", "parent_child"):
    for mode in ("semantic", "hybrid"):
        t0 = time.perf_counter()
        r1 = retriever.search(QUERY, strategy=strategy, retrieval_mode=mode, limit=5)
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2 = retriever.search(QUERY, strategy=strategy, retrieval_mode=mode, limit=5)
        t2 = time.perf_counter() - t0

        gain = t1 / max(t2, 0.001)
        label = f"{strategy}/{mode}"
        print(f"  {label:<28} {t1*1000:>9.0f}ms {t2*1000:>9.0f}ms {gain:>7.1f}x")
        if mode == "hybrid":
            break  # hybrid trop lent pour hierarchical/parent_child sans BM25 pre-indexe

index.close()
print("\nNote : le 1er appel charge le cache iter_chunks depuis SQLite.")
print("Le 2eme appel utilise le cache et montre le cout net de FAISS+BM25.")
