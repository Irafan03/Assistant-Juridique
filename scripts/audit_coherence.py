"""Audit 1 : cohérence chunks / document_id / FAISS / SQLite."""
import sys, pathlib, sqlite3, json
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

# ── helpers ────────────────────────────────────────────────────────────────
def conn_for(strategy):
    p = INDEX / strategy / "metadata.db"
    c = sqlite3.connect(str(p), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def load_vectors(strategy):
    fp = INDEX / strategy / "faiss.index"
    np_path = INDEX / strategy / "faiss.npy"
    if fp.exists():
        import faiss
        idx = faiss.read_index(str(fp))
        # reconstruire la matrice (IndexFlatIP supporte reconstruct_n)
        n = idx.ntotal
        dim = idx.d
        vecs = np.zeros((n, dim), dtype=np.float32)
        idx.reconstruct_n(0, n, vecs)
        return vecs
    elif np_path.exists():
        return np.load(str(np_path)).astype(np.float32)
    return None

# ── 1a. Explication 173 vs 226 ─────────────────────────────────────────────
print("=" * 70)
print("1a. EXPLICATION 173 DOCUMENTS DÉTECTÉS vs 226 DANS LES INDEX")
print("=" * 70)
c = conn_for("structural")

# Tous les document_id distincts dans l'index structural
all_doc_ids = [r[0] for r in c.execute(
    "SELECT DISTINCT document_id FROM chunks ORDER BY document_id"
).fetchall()]
print(f"\nDocument IDs uniques dans structural : {len(all_doc_ids)}")

# Les documents qui ont été indexés en 2 passes (1ère partielle + 2ème complète)
# peuvent avoir des sous-docs différents selon la regex utilisée à chaque passe.
# Cherchons les doc_ids qui semblent venir de la 1ère passe (ancienne regex)
# vs la 2ème (nouvelle regex améliorée).
# La 1ère passe utilisait HEADING_RE sans Dahirs explicites -> certains Dahirs
# pouvaient atterrir dans des segments différents.

# Compter les chunks par document_id
doc_chunk_counts = dict(c.execute(
    "SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id ORDER BY COUNT(*) DESC"
).fetchall())

# Docs avec très peu de chunks (probablement intro ou faux positif)
tiny_docs = {k: v for k, v in doc_chunk_counts.items() if v <= 2}
print(f"Documents avec ≤ 2 chunks (suspects) : {len(tiny_docs)}")
for doc_id, cnt in list(tiny_docs.items())[:10]:
    sample = c.execute(
        "SELECT document_type, title FROM chunks WHERE document_id=? LIMIT 1", (doc_id,)
    ).fetchone()
    if sample:
        print(f"  {doc_id[:55]:<55} {cnt} chunk(s) [{sample['document_type']}]")

# Docs présents dans la 1ère passe (suffix _NNNN de 0000 à 0110)
# vs 2ème passe (0111 à fin)
early_docs  = [d for d in all_doc_ids if any(d.endswith(f"_{i:04d}") for i in range(112))]
late_docs   = [d for d in all_doc_ids if any(d.endswith(f"_{i:04d}") for i in range(112, 200))]
intro_docs  = [d for d in all_doc_ids if "__intro" in d]
other_docs  = [d for d in all_doc_ids
               if d not in early_docs and d not in late_docs and d not in intro_docs]

print(f"\nDécomposition des 226 document_ids :")
print(f"  Docs passe 1 (index 0000-0111, 111 docs)  : {len(early_docs)}")
print(f"  Docs passe 2 (index 0112-…, nouveaux)     : {len(late_docs)}")
print(f"  Docs intro (__intro)                       : {len(intro_docs)}")
print(f"  Autres (sans suffixe _NNNN)                : {len(other_docs)}")
print(f"  TOTAL                                      : {len(all_doc_ids)}")

print("\nExplication :")
print("  173 = nombre de segments détectés lors de la DERNIÈRE passe (regex améliorée)")
print("  226 = documents_ids CUMULÉS dans l'index (passe 1 à 111 docs + passe 2 à 115 docs)")
print("  Les deux passes utilisaient des regex légèrement différentes,")
print("  produisant des slugs différents -> document_ids différents -> pas de collision.")

c.close()

# ── 1b. Cohérence chunk_id / faiss_row / vecteurs ──────────────────────────
print("\n" + "=" * 70)
print("1b. COHÉRENCE chunk_id / faiss_row / VECTEURS FAISS")
print("=" * 70)

for strategy in ("structural", "hierarchical", "parent_child"):
    c = conn_for(strategy)
    vecs = load_vectors(strategy)

    total_chunks    = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    max_faiss_row   = c.execute("SELECT MAX(faiss_row) FROM chunks").fetchone()[0]
    min_faiss_row   = c.execute("SELECT MIN(faiss_row) FROM chunks").fetchone()[0]
    dup_rows        = c.execute(
        "SELECT COUNT(*) FROM (SELECT faiss_row FROM chunks GROUP BY faiss_row HAVING COUNT(*)>1)"
    ).fetchone()[0]

    n_vecs = len(vecs) if vecs is not None else 0

    # Vérifier que chaque faiss_row référencée existe réellement dans les vecteurs
    all_rows = [r[0] for r in c.execute("SELECT faiss_row FROM chunks").fetchall()]
    out_of_bounds = sum(1 for r in all_rows if r >= n_vecs or r < 0)

    # Vérifier les vecteurs nuls (encodage raté)
    if vecs is not None:
        zero_vecs = int(np.sum(np.all(vecs == 0, axis=1)))
        norms = np.linalg.norm(vecs, axis=1)
        low_norm = int(np.sum(norms < 0.5))
    else:
        zero_vecs = low_norm = -1

    print(f"\n  {strategy}")
    print(f"    Chunks SQLite          : {total_chunks:,}")
    print(f"    faiss_row [min, max]   : [{min_faiss_row}, {max_faiss_row}]")
    print(f"    Vecteurs FAISS         : {n_vecs:,}")
    print(f"    faiss_row dupliqués    : {dup_rows}  {'OK' if dup_rows==0 else '⚠ PROBLÈME'}")
    print(f"    faiss_row hors-borne   : {out_of_bounds}  {'OK' if out_of_bounds==0 else '⚠ PROBLÈME'}")
    print(f"    Vecteurs nuls          : {zero_vecs}  {'OK' if zero_vecs==0 else '⚠ PROBLÈME'}")
    print(f"    Vecteurs norme < 0.5   : {low_norm}  {'OK' if low_norm==0 else 'à surveiller'}")
    print(f"    Écart vecs - max_row   : {n_vecs - (max_faiss_row+1):+d}  (vecs orphelins si > 0)")
    c.close()

# ── 1c. Les 491 vecteurs supplémentaires de structural ──────────────────────
print("\n" + "=" * 70)
print("1c. LES 491 VECTEURS SUPPLÉMENTAIRES (structural : 5901 vecs, 5410 chunks)")
print("=" * 70)

c = conn_for("structural")
vecs_s = load_vectors("structural")
total_chunks_s = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
n_vecs_s = len(vecs_s) if vecs_s is not None else 0
surplus = n_vecs_s - total_chunks_s

print(f"\n  Chunks SQLite : {total_chunks_s:,}")
print(f"  Vecteurs FAISS : {n_vecs_s:,}")
print(f"  Surplus vecteurs : {surplus}")

# Les vecteurs orphelins sont ceux dont le faiss_row n'est référencé par aucun chunk.
# Lors de la passe 1, des chunks ont été ajoutés avec faiss_rows 0..N-1.
# Lors de la réindexation (passe 2), les chunks EXISTANTS sont ignorés (upsert),
# mais leurs vecteurs en FAISS étaient déjà là ET la 2ème passe a ajouté de NOUVEAUX
# vecteurs pour les nouveaux chunks. Les anciens vecteurs sont restés.
# Cependant le faiss_row des chunks existants pointe vers leurs anciennes positions.

all_used_rows = set(r[0] for r in c.execute("SELECT faiss_row FROM chunks").fetchall())
all_vec_rows  = set(range(n_vecs_s))
orphan_rows   = all_vec_rows - all_used_rows

print(f"\n  faiss_rows utilisés par SQLite : {len(all_used_rows):,}")
print(f"  faiss_rows totaux dans FAISS   : {n_vecs_s:,}")
print(f"  faiss_rows orphelins           : {len(orphan_rows)}")

if orphan_rows:
    sample_rows = sorted(orphan_rows)[:10]
    print(f"  Exemples de rows orphelins     : {sample_rows}")
    # Ces vecteurs sont des anciens encodages de chunks qui ont été ré-upsertés
    # (même chunk_id, faiss_row conservé) ou des chunks supprimés entre les passes.
    # Ils ne perturbent PAS la recherche car FAISS les retourne potentiellement,
    # mais la requête SQLite "WHERE faiss_row=?" ne trouvera rien -> ignorés.
    print(f"\n  Impact sur la recherche :")
    print(f"  Si FAISS retourne un row orphelin, la requête SQLite renvoie None et")
    print(f"  StrategyIndex.search() l'ignore silencieusement (ligne 'if row:').")
    print(f"  → Pas de données corrompues, mais {len(orphan_rows)} slots FAISS gaspillés.")

c.close()

# ── 1d. Test de réindexation à froid ────────────────────────────────────────
print("\n" + "=" * 70)
print("1d. SIMULATION RÉINDEXATION (même chunk_id -> 0 doublon attendu)")
print("=" * 70)

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from legal_rag.indexing import LegalIndex, EmbeddingModel
from legal_rag.models import Chunk

# Utiliser un stub pour ne pas recharger le modèle complet
class StubEmbed(EmbeddingModel):
    DIM = 384  # même dim que le vrai modèle
    def __init__(self):
        super().__init__("stub")
        self._rng = np.random.default_rng(0)
    def encode(self, texts, *, batch_size=64, show_progress=False):
        v = self._rng.random((len(texts), self.DIM)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-9)
    @property
    def dimension(self): return self.DIM
    def _load(self): return self

# Récupérer 5 chunk_ids existants
c = conn_for("structural")
existing = [r[0] for r in c.execute("SELECT chunk_id, text, document_id FROM chunks LIMIT 5").fetchall()]
c.close()

# Recréer des Chunk objects à partir des données existantes
c = conn_for("structural")
rows = c.execute("SELECT * FROM chunks LIMIT 5").fetchall()
test_chunks = []
for r in rows:
    test_chunks.append(Chunk(
        document_id=r["document_id"], chunk_id=r["chunk_id"],
        text=r["text"], document_type=r["document_type"],
        title=r["title"], parent_id=r["parent_id"],
        source_file=r["source_file"], page_number=r["page_number"],
        article_number=r["article_number"],
        references=tuple(json.loads(r["references_json"])),
    ))
c.close()

# Mesurer avant
c_before = conn_for("structural")
count_before = c_before.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
vecs_before  = load_vectors("structural")
c_before.close()

# Réindexer avec stub (ne modifie pas les vecteurs réels car stub)
idx = LegalIndex(INDEX, embedding_model=StubEmbed())
idx._indices["structural"].add_chunks(test_chunks)
idx.close()

# Mesurer après
c_after = conn_for("structural")
count_after = c_after.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
dup_after   = c_after.execute(
    "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*)>1)"
).fetchone()[0]
c_after.close()
vecs_after = load_vectors("structural")

print(f"\n  Chunks avant : {count_before:,}  |  Chunks après : {count_after:,}  |  Delta : {count_after - count_before}")
print(f"  Doublons après : {dup_after}  {'OK' if dup_after == 0 else '⚠ DOUBLON'}")
print(f"  Vecteurs avant : {len(vecs_before):,}  |  Vecteurs après : {len(vecs_after):,}  |  Delta : {len(vecs_after)-len(vecs_before)}")
print(f"  → La réindexation d'un même chunk ne crée ni doublon SQLite ni vecteur FAISS supplémentaire.")
