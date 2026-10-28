"""Audit de l'index : en-têtes, chunks courts, artefacts de numéros d'article."""
import sys, pathlib, re, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

db = sqlite3.connect(str(pathlib.Path("data/index/structural/metadata.db")))
db.row_factory = sqlite3.Row

HEADER_RE = re.compile(
    r"RECUEIL DES TEXTES LEGISLATIFS|MIS.?À.?JOUR|ORGANISMES ASSIMILES"
    r"|PUBLIC\s*$|Mis à jour à fin",
    re.I | re.MULTILINE
)
SHORT_MIN = 80

rows = db.execute(
    "SELECT chunk_id, document_id, document_type, title, article_number, text, page_number "
    "FROM chunks"
).fetchall()

headers  = [r for r in rows if HEADER_RE.search((r["text"] or "")[:300])]
short_ch = [r for r in rows if len((r["text"] or "").strip()) < SHORT_MIN]
art_bad  = [r for r in rows if r["article_number"] and len(str(r["article_number"])) > 5]

print(f"Total chunks structural : {len(rows):,}")
print(f"En-têtes détectés       : {len(headers)}")
print(f"Chunks < {SHORT_MIN} chars    : {len(short_ch)}")
print(f"Art. suspects (len>5)   : {len(art_bad)}")
print()

print("Exemples en-têtes :")
for r in headers[:5]:
    print(f"  chunk={r['chunk_id'][:28]} art={r['article_number']!r} p={r['page_number']}")
    print(f"    {(r['text'] or '')[:110].replace(chr(10),' ')!r}")

print("\nExemples chunks courts :")
for r in short_ch[:5]:
    print(f"  len={len((r['text'] or '').strip())} art={r['article_number']!r}")
    print(f"    {(r['text'] or '').strip()[:80]!r}")

print("\nExemples art. suspects :")
for r in art_bad[:5]:
    print(f"  art={r['article_number']!r} p={r['page_number']}")
    print(f"    {(r['text'] or '')[:80].replace(chr(10),' ')!r}")

# Doublons de loi (même texte légal, plusieurs versions/dahirs)
print("\nDoublons de texte légal (même contenu normalisé) :")
from collections import Counter
def norm(t):
    return re.sub(r"\s+", " ", (t or "").casefold()[:120].strip())
counts = Counter(norm(r["text"]) for r in rows)
for text_norm, cnt in counts.most_common(5):
    if cnt > 1:
        print(f"  {cnt}x : {text_norm[:80]!r}")

db.close()
