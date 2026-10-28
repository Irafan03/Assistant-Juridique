"""Inspecter tous les chunks de la page 638 et comprendre le chunk en-tête."""
import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.postprocess import is_header_chunk, _strip_header_lines, _HEADER_RE
from legal_rag.models import Chunk

db = sqlite3.connect(str(pathlib.Path("data/index/structural/metadata.db")))
db.row_factory = sqlite3.Row

rows = db.execute("SELECT * FROM chunks WHERE page_number=638").fetchall()
print(f"Chunks page 638 : {len(rows)}")
for row in rows:
    text = row["text"] or ""
    c = Chunk(row["document_id"], row["chunk_id"], text,
              article_number=str(row["article_number"] or ""), title=row["title"])
    cleaned = _strip_header_lines(text)
    is_h = is_header_chunk(c)
    has_hdr = bool(_HEADER_RE.search(text))
    lines_clean = [l.strip() for l in cleaned.splitlines() if l.strip() and not l.strip().isdigit()]
    first = lines_clean[0][:80] if lines_clean else ""
    first_is_upper = first == first.upper()
    print(f"\n  art={row['article_number']!r:<6} is_h={is_h} has_hdr={has_hdr} "
          f"len_raw={len(text):<5} len_clean={len(cleaned):<5}")
    print(f"  first_real_line={first!r} is_upper={first_is_upper}")
    print(f"  raw[:100]={text[:100].replace(chr(10),' ')!r}")

db.close()
