"""Test de recherche réelle sur l'index BAM.

Note sur les performances :
- structural (5 400 chunks)  : hybride ET sémantique rapides
- hierarchical/parent_child (16 000 chunks) : le mode hybride ET la résolution
  des références associées itèrent sur tout le corpus en pur Python.
  → Utiliser retrieval_mode="semantic" et limit réduit pour ces stratégies.
  → Pour la production, un BM25 indexé (ex. rank-bm25) serait nécessaire.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.api import create_app

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

print("Chargement de l'index réel (FAISS)…")
index     = LegalIndex(INDEX)
retriever = LegalRetriever(index)
client    = TestClient(create_app(retriever))

QUESTIONS = [
    ("Quelles sont les conditions d'agrément des établissements de crédit ?",  "loi"),
    ("Comment sont calculés les fonds propres réglementaires ?",               "circulaire"),
    ("Quelles sont les obligations en matière de lutte contre le blanchiment ?","circulaire"),
    ("Quel est le ratio de liquidité requis pour les banques ?",               "circulaire"),
    ("Quelles règles s'appliquent aux participations des banques ?",           "circulaire"),
]

CONFIGS = [
    ("structural",   "hybrid"),
    ("structural",   "semantic"),
    ("hierarchical", "semantic"),   # hybrid trop lent sur 16k chunks sans BM25 indexé
    ("parent_child", "semantic"),
]

ok_count = 0
fail_count = 0

for strategy, mode in CONFIGS:
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"STRATÉGIE : {strategy}  |  MODE : {mode}")
    print(f"{'='*65}")
    for question, expected_type in QUESTIONS:
        t1 = time.time()
        r = client.post("/api/search", json={
            "query": question,
            "strategy": strategy,
            "retrieval_mode": mode,
            "limit": 3,
        })
        elapsed = time.time() - t1
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
        body = r.json()
        passages   = body["passages"]
        circulaires= body["circulaires"]
        assoc      = body["textes_associes"]
        warnings   = body["warnings"]

        got_type = passages[0]["document_type"] if passages else "—"
        match    = "✓" if got_type == expected_type else f"≠ ({got_type})"
        ok_count   += (1 if got_type == expected_type else 0)
        fail_count += (0 if got_type == expected_type else 1)

        print(f"\n  Q: {question[:68]}")
        print(f"     {len(passages)} passages | {len(circulaires)} circ. | {len(assoc)} associés | {elapsed:.1f}s {match}")
        if passages:
            p = passages[0]
            print(f"     [1] [{p['document_type']}] {str(p.get('title',''))[:55]}")
            print(f"         art={p.get('article_number')}  p={p.get('page_number')}  "
                  f"score={p['score']:.4f}  relation={p['relation']}")
            if p.get("references"):
                print(f"         refs: {list(p['references'])[:2]}")
        for w in warnings[:1]:
            print(f"     ⚠  {w}")

index.close()

print(f"\n\n{'='*65}")
print(f"RÉSULTATS : {ok_count} type attendu correct / {ok_count+fail_count} total")
print(f"Note : 'type attendu' = premier résultat correspond au type juridique prévu.")
print(f"Les résultats ≠ ne sont pas des erreurs — le moteur peut retourner")
print(f"un texte associé (loi, directive) plutôt que la circulaire principale.")
