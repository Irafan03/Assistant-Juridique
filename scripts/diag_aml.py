"""Diagnostic Étape 1 : analyse complète sans modifier le code."""
import sys, pathlib, sqlite3, re, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

def norm(s):
    import unicodedata
    return re.sub(r"[\u0300-\u036f]", "", unicodedata.normalize("NFKD", (s or "").casefold()))

# ─────────────────────────────────────────────────────────────────────────────
# 1a) Chunks contenant les termes AML — par stratégie
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 78)
print("1a) CHUNKS CONTENANT LES TERMES AML (par stratégie)")
print("=" * 78)

TERMS = {
    "vigilance": "vigilance",
    "soupçon":   "soupcon",
    "bénéficiaire effectif": "beneficiaire effectif",
    "conservation": "conservation",
    "identification du client": "identification du client|connaissance du client",
    "déclaration": "declaration",
}

for strategy in ("structural", "hierarchical", "parent_child"):
    db = sqlite3.connect(str(INDEX / strategy / "metadata.db"))
    db.row_factory = sqlite3.Row
    print(f"\n--- Stratégie : {strategy} ---")
    for label, pat in TERMS.items():
        pat_n = norm(pat)
        # Chercher dans le texte normalisé (on recharge et normalise en Python)
        rows = db.execute("SELECT chunk_id, document_id, document_type, title, "
                          "article_number, page_number, text FROM chunks").fetchall()
        hits = []
        for r in rows:
            if re.search(pat_n, norm(r["text"]) + " " + norm(r["title"] or "")):
                hits.append(r)
        # Document distincts / pages
        docs = {}
        for r in hits:
            docs.setdefault(r["document_id"], []).append(r["page_number"])
        print(f"  [{label:<22}] {len(hits):>4} chunks | {len(docs):>3} documents")
        # Afficher les 3 premiers avec leur page/article
        for r in hits[:3]:
            print(f"        art={r['article_number']!r:<8} p={r['page_number']} "
                  f"type={r['document_type']:<10} | {(r['title'] or '')[:45]}")
    db.close()
