"""Migration des index numpy (.npy) vers FAISS (.index) pour les performances.

Les métadonnées SQLite sont conservées intactes.
Les faiss_row restent identiques (la migration ne réordonne pas les vecteurs).
"""
import sys, pathlib, sqlite3
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

try:
    import faiss
except ImportError:
    print("faiss-cpu non installé. Rien à faire.")
    sys.exit(0)

for strategy in ("structural", "hierarchical", "parent_child"):
    npy_path   = INDEX / strategy / "faiss.npy"
    faiss_path = INDEX / strategy / "faiss.index"

    if not npy_path.exists():
        print(f"{strategy}: pas de fichier .npy, ignoré")
        continue
    if faiss_path.exists():
        print(f"{strategy}: faiss.index déjà présent, ignoré")
        continue

    print(f"Migration {strategy}…", end=" ", flush=True)
    vectors = np.load(str(npy_path)).astype(np.float32)
    dim     = vectors.shape[1]
    idx     = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    faiss.write_index(idx, str(faiss_path))

    # Vérification : relire et comparer
    idx2 = faiss.read_index(str(faiss_path))
    assert idx2.ntotal == len(vectors), f"Migration incomplète : {idx2.ntotal} != {len(vectors)}"
    print(f"{len(vectors):,} vecteurs (dim={dim}) → OK")

print("\nMigration terminée. Les fichiers .npy sont conservés comme sauvegarde.")
