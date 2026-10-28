"""Analyse de la structure interne du recueil pour identifier les delimiteurs
de documents individuels (lois, decrets, circulaires...).
"""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.extraction import extract_document

PDF = pathlib.Path(__file__).parent.parent / "recueil-des-textes-legislatifs-et-reglementaires-de-bam-vf.pdf"
doc = extract_document(PDF)
text = doc.full_text

# Chercher les marqueurs de debut de document individuel
# Pattern typique: "LOI N° XX-YY" ou "CIRCULAIRE N° X/W/YYYY" en debut de bloc
heading_re = re.compile(
    r'(?:^|\n)('
    r'(?:LOI|Loi)\s+[Nn][°º]\s*\d+[-\d]*'
    r'|(?:DECRET|Décret|décret)\s+[Nn][°º]\s*\d+[-\d]*'
    r'|(?:CIRCULAIRE|Circulaire)\s+[Nn][°º]\s*\d+[\w/.-]*'
    r'|(?:ARRETE|Arrêté|arrêté)\s+(?:du|[Nn][°º])[^.]{0,60}'
    r'|(?:DIRECTIVE|Directive)\s+[Nn][°º]\s*\d+'
    r'|(?:DAHIR|Dahir)\s+(?:portant|[Nn][°º]|\w+\s+loi)[^.]{0,60}'
    r'|(?:LETTRE CIRCULAIRE|Lettre circulaire)\s+[Nn][°º]\s*\d+'
    r')',
    re.MULTILINE
)

matches = list(heading_re.finditer(text))
print(f"Marqueurs de documents identifies : {len(matches)}")
print("\nPremiers 20 marqueurs :")
for m in matches[:20]:
    title = m.group(1).strip().replace('\n', ' ')
    print(f"  offset {m.start():>8} | {title[:90]}")

print("\nDerniers 5 marqueurs :")
for m in matches[-5:]:
    title = m.group(1).strip().replace('\n', ' ')
    print(f"  offset {m.start():>8} | {title[:90]}")

# Compter par type
from collections import Counter
types = Counter()
for m in matches:
    t = m.group(1).strip().lower()
    if t.startswith('loi'):
        types['loi'] += 1
    elif t.startswith('d'):
        types['decret'] += 1
    elif t.startswith('c') and 'lettre' not in t:
        types['circulaire'] += 1
    elif t.startswith('a'):
        types['arrete'] += 1
    elif t.startswith('dir'):
        types['directive'] += 1
    elif t.startswith('dahir'):
        types['dahir'] += 1
    elif 'lettre' in t:
        types['lettre_circulaire'] += 1
    else:
        types['autre'] += 1

print(f"\nRepartition par type de document :")
for k, v in types.most_common():
    print(f"  {k:<20} : {v}")
