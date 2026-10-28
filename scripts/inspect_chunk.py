"""Inspecter le contenu exact des chunks problématiques."""
import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.postprocess import is_header_chunk, _strip_header_lines
from legal_rag.models import Chunk

db = sqlite3.connect(str(pathlib.Path("data/index/structural/metadata.db")))
db.row_factory = sqlite3.Row

print("=== Chunks Circulaire 5/W/2017 page 638 ===")
rows = db.execute(
    "SELECT chunk_id, text, page_number, article_number, title FROM chunks "
    "WHERE page_number = 638 LIMIT 5"
).fetchall()
for r in rows:
    text = r["text"] or ""
    c = Chunk("d", r["chunk_id"], text, article_number=r["article_number"],
              title=r["title"])
    cleaned = _strip_header_lines(text)
    is_h = is_header_chunk(c)
    print(f"\nchunk_id : {r['chunk_id'][:35]}")
    print(f"art      : {r['article_number']!r}")
    print(f"is_header: {is_h}")
    print(f"len_raw  : {len(text)}")
    print(f"len_clean: {len(cleaned)}")
    print(f"text_raw : {text[:200]!r}")
    print(f"cleaned  : {cleaned[:200]!r}")

db.close()
