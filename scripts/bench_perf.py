"""Benchmark 2 : mesure séparée des composants de la recherche.

Mesure :
  - Chargement du modèle
  - Encodage de la requête
  - Recherche vectorielle FAISS
  - iter_chunks (chargement BM25 pool)
  - BM25 pur
  - Résolution des références (assoc loop)
  - Temps total via API
"""
import sys, pathlib, time, sqlite3
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

from legal_rag.indexing import LegalIndex, EmbeddingModel, StrategyIndex
from legal_rag.retrieval import LegalRetriever, _tokens
from legal_rag.models import Chunk

QUERY = "conditions agrément établissements de crédit"
N_RUNS = 3

# ── Chargement ──────────────────────────────────────────────────────────────
print("=" * 70)
print("BENCHMARK PERFORMANCES — Legal RAG")
print("=" * 70)

t0 = time.perf_counter()
index = LegalIndex(INDEX)
t_load_index = time.perf_counter() - t0
print(f"\nChargement LegalIndex (sans modèle) : {t_load_index*1000:.0f} ms")

# Force le chargement du modèle
t0 = time.perf_counter()
_ = index.embedding_model.dimension
t_load_model = time.perf_counter() - t0
print(f"Chargement modèle sentence-transformers : {t_load_model*1000:.0f} ms (1ère fois uniquement)")

# ── Encodage requête ─────────────────────────────────────────────────────────
times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    vec = index.embedding_model.encode([QUERY])
    times.append(time.perf_counter() - t0)
print(f"\nEncodage requête ({N_RUNS} runs) : moy={np.mean(times)*1000:.1f} ms  min={min(times)*1000:.1f} ms")

# ── Recherche vectorielle FAISS (search uniquement, sans SQLite) ─────────────
print("\n--- Recherche vectorielle FAISS (pur, sans SQLite lookup) ---")
for strategy in ("structural", "hierarchical", "parent_child"):
    idx_obj = index._indices[strategy]
    faiss_idx = idx_obj._load_faiss()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _, rows = faiss_idx.search(vec, 20)
        times.append(time.perf_counter() - t0)
    n = faiss_idx.ntotal
    print(f"  {strategy:<15} ({n:>6} vecs) : moy={np.mean(times)*1000:.2f} ms")

# ── iter_chunks (chargement du pool BM25) ──────────────────────────────────
print("\n--- iter_chunks() — chargement du pool BM25 depuis SQLite ---")
for strategy in ("structural", "hierarchical", "parent_child"):
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        pool = list(index.iter_chunks(strategy=strategy))
        times.append(time.perf_counter() - t0)
    print(f"  {strategy:<15} ({len(pool):>6} chunks) : moy={np.mean(times)*1000:.0f} ms")

# ── BM25 pur (après chargement du pool) ─────────────────────────────────────
print("\n--- BM25 pur (tokenisation + scoring, pool déjà chargé en mémoire) ---")
retriever = LegalRetriever(index)
for strategy in ("structural", "hierarchical", "parent_child"):
    pool = list(index.iter_chunks(strategy=strategy))
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = retriever._lexical_search(QUERY, pool, 20)
        times.append(time.perf_counter() - t0)
    print(f"  {strategy:<15} ({len(pool):>6} chunks) : moy={np.mean(times)*1000:.0f} ms")

# ── Résolution des références (loop assoc dans retrieval.py) ─────────────────
print("\n--- Résolution des références associées (parcours pool post-RRF) ---")
# Simuler la boucle la plus coûteuse : iter sur pool pour trouver les refs matchantes
from legal_rag.references import extract_references, normalize_reference
import re

for strategy in ("structural", "hierarchical", "parent_child"):
    pool = list(index.iter_chunks(strategy=strategy))
    # Prendre un chunk circulaire avec des refs comme source
    circ_chunks = [c for c in pool if (c.document_type or "").lower() == "circulaire" and c.references]
    if not circ_chunks:
        print(f"  {strategy:<15} : pas de circulaire avec refs")
        continue
    test_circ = circ_chunks[0]
    cited = {normalize_reference(r) for r in (*test_circ.references, *extract_references(test_circ.text))}

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        # Reproduire exactement la boucle de retrieval.py
        associated_ids = set()
        for target in pool:
            if target.document_id == test_circ.document_id:
                continue
            if not re.search(r'loi|décret|decret|arrêté|arrete|texte|قانون|مرسوم|قرار',
                             (target.document_type or ""), re.I):
                continue
            target_refs = {normalize_reference(r) for r in (*target.references, *extract_references(target.text))}
            target_refs.update(normalize_reference(r) for r in extract_references(target.title or ""))
            if cited.intersection(target_refs):
                associated_ids.add(target.chunk_id)
        times.append(time.perf_counter() - t0)
    print(f"  {strategy:<15} ({len(pool):>6} chunks) : moy={np.mean(times)*1000:.0f} ms  ({len(associated_ids)} assoc trouvés)")

# ── Appel complet retriever.search() ────────────────────────────────────────
print("\n--- retriever.search() complet ---")
for strategy in ("structural", "hierarchical", "parent_child"):
    for mode in ("semantic", "hybrid"):
        t0 = time.perf_counter()
        resp = retriever.search(QUERY, strategy=strategy, retrieval_mode=mode, limit=5)
        elapsed = time.perf_counter() - t0
        print(f"  {strategy:<15} {mode:<8} : {elapsed*1000:.0f} ms  ({len(resp.passages)} passages, {len(resp.circulares)} circ., {len(resp.associated_texts)} assoc.)")

index.close()

# ── Identification précise du code lent ─────────────────────────────────────
print("\n" + "=" * 70)
print("IDENTIFICATION DU CODE RESPONSABLE DES PARCOURS COMPLETS")
print("=" * 70)
print("""
Fichier : legal_rag/retrieval.py  (code Personne 2 — non modifié)
Ligne   : pool = list(self.index.iter_chunks(strategy=strategy))

Cette ligne charge TOUS les chunks en mémoire pour :
  1. BM25 : _lexical_search(query, pool, limit)
     → tokenise et score CHAQUE chunk
     → O(n × |tokens|) : 16 000 chunks × ~100 tokens = ~1.6M opérations
  
  2. Résolution références : boucle sur pool pour trouver les textes associés
     → extrait les références de CHAQUE chunk via extract_references(chunk.text)
     → appelle des regex sur 16 000 textes
     → O(n) avec constante élevée (regex)

Coût mesuré sur structural (5 410 chunks) vs hierarchical (16 081 chunks) :
  - iter_chunks structural  : < 200 ms
  - BM25 structural         : < 500 ms
  - Résolution refs struct. : < 1 000 ms
  
  - iter_chunks hierarchical : > 500 ms
  - BM25 hierarchical        : > 2 000 ms
  - Résolution refs hier.    : > 5 000 ms

Solution compatible sans modifier retrieval.py :
  Option A : Pré-calculer et stocker les références normalisées dans une
             colonne SQLite indexée -> iter_chunks peut retourner des objets
             légers pour BM25 sans re-parser le texte.
  Option B : Ajouter une colonne "refs_normalized_json" dans SQLite et
             overrider iter_chunks pour retourner des Chunk avec refs déjà
             normalisées -> évite les regex répétées dans la boucle assoc.
  Option C (recommandée) : Ajouter un index BM25 persistant (rank-bm25)
             dans StrategyIndex, appelé depuis search() avant iter_chunks.
             L'interface IndexAdapter reste inchangée pour Personne 2.
""")
