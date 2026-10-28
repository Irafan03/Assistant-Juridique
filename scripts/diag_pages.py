"""Diagnostic 1b : texte brut des pages 638-640 et chunks produits."""
import sys, pathlib, sqlite3, re
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.extraction import extract_document
from legal_rag.chunking import chunk_document

PDF   = pathlib.Path(__file__).parent.parent / "recueil-des-textes-legislatifs-et-reglementaires-de-bam-vf.pdf"
INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

# ─────────────────────────────────────────────────────────────────────────────
# 1b) Texte brut extrait des pages 638-640
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 78)
print("1b) TEXTE BRUT DES PAGES 638-640")
print("=" * 78)
doc = extract_document(PDF)
for page in doc.pages:
    if 636 <= page.page_number <= 642:
        print(f"\n──── Page {page.page_number} (len={len(page.text)}) ────")
        print(page.text[:1200])
        print("..." if len(page.text) > 1200 else "")

# ─────────────────────────────────────────────────────────────────────────────
# Chunks produits par chaque stratégie pour la circulaire 5/W/2017 (page 638)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("CHUNKS PRODUITS POUR LA CIRCULAIRE 5/W/2017 (page 638) — par stratégie")
print("=" * 78)

# Identifier le document_id de la circulaire 5/W/2017 dans l'index
db = sqlite3.connect(str(INDEX / "structural" / "metadata.db"))
db.row_factory = sqlite3.Row
rows = db.execute(
    "SELECT DISTINCT document_id FROM chunks WHERE title LIKE '%5/W/2017%' LIMIT 5"
).fetchall()
print(f"document_id candidats : {[r['document_id'] for r in rows]}")
db.close()

# Re-découper le document complet et filtrer la circulaire 5/W/2017
from scripts.index_bam import split_recueil  # réutiliser la segmentation
sub_docs, _ = split_recueil(doc)
circ517 = [d for d in sub_docs if "5/w/2017" in (d.metadata.get("heading") or "").lower()
           or "circulaire n" in (d.metadata.get("heading") or "").lower()
           and "5" in (d.metadata.get("heading") or "")][:1]

if circ517:
    target = circ517[0]
    print(f"\nDocument ciblé : {target.document_id}")
    print(f"  heading : {target.metadata.get('heading')}")
    print(f"  type    : {target.detected_type}")
    print(f"  pages   : {[p.page_number for p in target.pages]}")
    print(f"  texte complet (3000 premiers chars) :")
    print(repr(target.full_text[:3000]))

    for strategy in ("structural", "hierarchical", "parent_child"):
        chunks = chunk_document(target, strategy)
        print(f"\n--- {strategy} : {len(chunks)} chunks ---")
        for i, c in enumerate(chunks[:8], 1):
            print(f"  [{i}] chunk_id={c.chunk_id[:20]}... art={c.article_number!r} "
                  f"p={c.page_number} len={len(c.text)}")
            print(f"      {c.text[:140].replace(chr(10), ' | ')!r}")
else:
    print("\n[!] Circulaire 5/W/2017 non trouvée dans la segmentation")
    # Chercher par page 638
    for d in sub_docs:
        if any(636 <= p.page_number <= 640 for p in d.pages):
            print(f"\nDoc contenant page 638 : {d.document_id}")
            print(f"  heading : {d.metadata.get('heading')}")
