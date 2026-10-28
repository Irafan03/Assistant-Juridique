"""Vérification du nouvel index après réindexation."""
import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.postprocess import is_header_chunk, _strip_header_lines
from legal_rag.models import Chunk

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

db = sqlite3.connect(str(INDEX / "structural" / "metadata.db"))
db.row_factory = sqlite3.Row

# 1. Articles AML de la circulaire 5/W/2017 (pages 636-650)
rows = db.execute(
    "SELECT article_number, page_number, text FROM chunks "
    "WHERE title LIKE '%5/W/2017%' AND page_number BETWEEN 636 AND 650 "
    "ORDER BY page_number, article_number"
).fetchall()
print(f"Chunks Circulaire 5/W/2017 (pages 636-650) : {len(rows)}")
for r in rows:
    snippet = (r["text"] or "")[:70].replace("\n", " ")
    print(f"  art={r['article_number']!r:<6} p={r['page_number']} | {snippet!r}")

print()

# 2. En-têtes restants dans le top-10 de la recherche AML
from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.postprocess import filter_passages

index = LegalIndex(INDEX)
_ = list(index.iter_chunks(strategy="structural"))
retr = LegalRetriever(index)
resp = retr.search(
    "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?",
    strategy="structural", retrieval_mode="hybrid", limit=10
)

print(f"\nTop-10 AVANT filtre :")
for i, c in enumerate(resp.passages, 1):
    chunk = c.chunk
    h = is_header_chunk(chunk)
    cleaned = _strip_header_lines(chunk.text or "")
    flag = " [HEADER]" if h else ""
    print(f"  [{i}] is_header={h} art={chunk.article_number!r:<6} p={chunk.page_number}"
          f" len_raw={len(chunk.text or ''):<5} len_clean={len(cleaned):<5}{flag}")
    print(f"       {(chunk.text or '')[:80].replace(chr(10),' ')!r}")

kept, rejected = filter_passages(resp.passages)
print(f"\nFiltrage : {len(kept)} gardes, {len(rejected)} rejetes")
for r in rejected:
    print(f"  REJETE: art={r.chunk.article_number!r} p={r.chunk.page_number}")
    print(f"    {(r.chunk.text or '')[:80].replace(chr(10),' ')!r}")

db.close()
index.close()
