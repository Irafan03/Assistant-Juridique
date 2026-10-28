"""Script d'exploration du recueil BAM.
Etape 1 : extraction et apercu.
Etape 2 : analyse de la structure (types de documents detectes dans le texte).
Etape 3 : chunking apercu par strategie.
Etape 4 : indexation complete.
"""
import sys, pathlib, re, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.extraction import extract_document
from legal_rag.chunking import chunk_document
from legal_rag.references import extract_references

PDF = pathlib.Path(__file__).parent.parent / "recueil-des-textes-legislatifs-et-reglementaires-de-bam-vf.pdf"

print("=" * 70)
print("ETAPE 1 — EXTRACTION")
print("=" * 70)
doc = extract_document(PDF)
print(f"Document ID   : {doc.document_id}")
print(f"Pages         : {doc.metadata['num_pages']}")
print(f"Total chars   : {len(doc.full_text):,}")
print(f"Type detecte  : {doc.detected_type}  (auto — recueil multi-types)")

print("\n" + "=" * 70)
print("ETAPE 2 — ANALYSE DE LA STRUCTURE")
print("=" * 70)

# Compter les articles detectes
articles = re.findall(
    r'(?:^|\n)(?:Article|art\.)\s*(\d+(?:\s*bis|\s*ter)?)',
    doc.full_text, re.I | re.MULTILINE
)
print(f"Articles detectes        : {len(articles)}")

# Compter les chapitres/sections
sections = re.findall(
    r'(?:^|\n)(?:Chapitre|Section|Titre|Partie)\s+[\dIVXivx]+',
    doc.full_text, re.I | re.MULTILINE
)
print(f"Chapitres/Sections       : {len(sections)}")

# Extraire les references juridiques (echantillon des 5000 premiers chars par page)
sample_text = "\n".join(p.text[:3000] for p in doc.pages[:30])
refs = extract_references(sample_text)
print(f"References (30 premieres pages) : {len(refs)}")
for r in refs[:10]:
    print(f"  {r!r}")

# Detecter les types de documents dans le texte complet
patterns_type = {
    "Circulaire": r'\bCirculaire\s+n[°º]?\s*\d+',
    "Loi":        r'\bloi\s+n[°º]\s*\d+',
    "Decret":     r'\bdécret\s+n[°º]\s*\d+',
    "Arrete":     r'\barrêté\s+(?:du|n[°º])',
    "Directive":  r'\bdirective\b',
    "Lettre circulaire": r'\blettre\s+circulaire\b',
    "Dahir":      r'\bdahir\b',
}
print("\nTypes de documents identifies dans le texte :")
for label, pat in patterns_type.items():
    count = len(re.findall(pat, doc.full_text, re.I))
    if count:
        print(f"  {label:<20} : {count} occurrences")

print("\n" + "=" * 70)
print("ETAPE 3 — CHUNKING (apercu)")
print("=" * 70)

for strategy in ("structural", "hierarchical", "parent_child"):
    chunks = chunk_document(doc, strategy)
    sizes = [len(c.text) for c in chunks]
    with_articles = sum(1 for c in chunks if c.article_number)
    with_refs     = sum(1 for c in chunks if c.references)
    doc_types     = collections.Counter(c.document_type for c in chunks)
    print(f"\nStrategie : {strategy}")
    print(f"  Chunks totaux    : {len(chunks):,}")
    print(f"  Taille min/moy/max : {min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)} chars")
    print(f"  Avec article_number: {with_articles:,}")
    print(f"  Avec references    : {with_refs:,}")
    print(f"  document_type      : {dict(doc_types)}")
    # Echantillon de 3 chunks
    print("  --- 3 premiers chunks ---")
    for c in chunks[:3]:
        print(f"    [{c.chunk_id[:20]}...] art={c.article_number!r} p={c.page_number} "
              f"refs={len(c.references)} : {c.text[:80].replace(chr(10),' ')!r}")
