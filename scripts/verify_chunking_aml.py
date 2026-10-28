"""Vérifier que le nouveau chunking découpe les articles AML de la circulaire 5/W/2017."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.extraction import extract_document, RawDocument, RawPage
from legal_rag.chunking import chunk_document
from scripts.index_bam import split_recueil

PDF = pathlib.Path(__file__).parent.parent / "recueil-des-textes-legislatifs-et-reglementaires-de-bam-vf.pdf"

doc = extract_document(PDF)
sub_docs, _ = split_recueil(doc)

# Trouver la circulaire 5/W/2017 (page 636-642)
for d in sub_docs:
    if any(636 <= p.page_number <= 642 for p in d.pages) and "5" in (d.metadata.get("heading") or ""):
        target = d
        break
else:
    # Fallback : le doc contenant la page 638
    target = next(d for d in sub_docs if any(636 <= p.page_number <= 642 for p in d.pages))

print(f"Document : {target.document_id}")
print(f"  heading : {target.metadata.get('heading')}")
print(f"  pages   : {[p.page_number for p in target.pages]}")

for strategy in ("structural", "hierarchical", "parent_child"):
    chunks = chunk_document(target, strategy)
    # Filtrer les chunks des pages 636-642
    aml_chunks = [c for c in chunks if c.page_number and 636 <= c.page_number <= 642]
    print(f"\n--- {strategy} : {len(aml_chunks)} chunks AML (pages 636-642) ---")
    for c in aml_chunks[:10]:
        print(f"  art={c.article_number!r:<8} p={c.page_number} len={len(c.text):<5} "
              f"| {c.text[:75].replace(chr(10), ' ')!r}")
