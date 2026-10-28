"""Évaluation comparative des 3 stratégies × 2 modes sur 8 questions vérifiées."""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

INDEX      = pathlib.Path(__file__).parent.parent / "data" / "index"
CASES_PATH = pathlib.Path(__file__).parent.parent / "evaluation" / "cases.json"
K = 5

cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
index = LegalIndex(INDEX)
retriever = LegalRetriever(index)

# Préchauffage cache iter_chunks pour toutes les stratégies
print("Préchauffage du cache…")
for s in ("structural", "hierarchical", "parent_child"):
    _ = list(index.iter_chunks(strategy=s))
    print(f"  {s}: cache prêt")

CONFIGS = [
    ("structural",   "semantic"),
    ("structural",   "hybrid"),
    ("hierarchical", "semantic"),
    ("parent_child", "semantic"),
]

print("\n" + "="*75)
print("ÉVALUATION COMPARATIVE — 8 questions × 4 configurations")
print("="*75)

results_table = {}

for strategy, mode in CONFIGS:
    label = f"{strategy}/{mode}"
    config_metrics = {"p@k": [], "r@k": [], "mrr": [], "ndcg@k": [], "times": []}

    for case in cases:
        relevant = set(case["relevant_articles"])
        if not relevant:
            continue

        t0 = time.perf_counter()
        resp = retriever.search(
            case["query"], strategy=strategy, retrieval_mode=mode, limit=K
        )
        elapsed = time.perf_counter() - t0

        ranked = [p.chunk.chunk_id for p in resp.passages]

        config_metrics["p@k"].append(precision_at_k(ranked, relevant, K))
        config_metrics["r@k"].append(recall_at_k(ranked, relevant, K))
        config_metrics["mrr"].append(reciprocal_rank(ranked, relevant))
        config_metrics["ndcg@k"].append(ndcg_at_k(ranked, relevant, K))
        config_metrics["times"].append(elapsed)

    n = len(config_metrics["p@k"])
    if n == 0:
        continue

    avg = {k: sum(v)/len(v) for k, v in config_metrics.items() if k != "times"}
    avg_time = sum(config_metrics["times"]) / len(config_metrics["times"])
    results_table[label] = {**avg, "avg_time_s": avg_time, "n": n}

# ── Affichage tableau ────────────────────────────────────────────────────────
print(f"\n{'Config':<25} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'nDCG@5':>8} {'moy(s)':>8}")
print("-" * 62)
for label, m in results_table.items():
    print(f"  {label:<23} {m['p@k']:>6.3f} {m['r@k']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@k']:>8.3f} {m['avg_time_s']:>8.1f}s")

# ── Détail question par question pour structural/hybrid ──────────────────────
print(f"\n{'='*75}")
print("DÉTAIL PAR QUESTION — structural/hybrid")
print(f"{'='*75}")

for case in cases:
    relevant = set(case["relevant_articles"])
    resp = retriever.search(case["query"], strategy="structural",
                            retrieval_mode="hybrid", limit=K)
    ranked = [p.chunk.chunk_id for p in resp.passages]
    p = precision_at_k(ranked, relevant, K)
    r = recall_at_k(ranked, relevant, K)

    print(f"\n  {case['id']} : {case['query'][:65]}")
    print(f"    P@{K}={p:.2f}  R@{K}={r:.2f}  ({case['note'][:50]})")
    for i, passage in enumerate(resp.passages[:3], 1):
        c = passage.chunk
        match = "✓" if c.chunk_id in relevant else " "
        print(f"    [{i}]{match} [{c.document_type}] {str(c.title)[:50]}  art={c.article_number}  rel={passage.relation}")
        if c.references:
            print(f"        refs: {list(c.references)[:2]}")

index.close()

# ── Limites de cette évaluation ──────────────────────────────────────────────
print(f"\n{'='*75}")
print("LIMITES DE CETTE ÉVALUATION")
print(f"{'='*75}")
print("""
  1. Jugements construits par mots-clés (non vérifiés manuellement) :
     Un chunk est marqué "pertinent" si son texte contient les mots-clés,
     pas parce qu'un expert juridique a confirmé sa pertinence.
     → Les métriques P@k/R@k mesurent la capacité à retrouver ces chunks,
       pas la qualité juridique réelle.

  2. Taille du jeu : 8 questions (besoins : 20 pour run_evaluation() de Prompt 2).
     → Les scores sont indicatifs, non statistiquement significatifs.

  3. Références attendues tirées de structural uniquement :
     Les chunk_ids varient entre stratégies (textes découpés différemment).
     → La comparaison cross-stratégie sur les mêmes chunk_ids est biaisée.
     
  4. Modèle non spécialisé juridique français :
     paraphrase-multilingual-MiniLM-L12-v2 est un modèle généraliste.
     Un modèle fine-tuné sur du droit marocain améliorerait les scores.

  5. Performance mesurée après préchauffage du cache :
     Le premier appel (cold start) est 3-10x plus lent.
""")
