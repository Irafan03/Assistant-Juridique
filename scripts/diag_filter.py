"""Diagnostic 1d : combien le filtre supprime et pourquoi."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.postprocess import (process_response, filter_passages,
                                    is_header_chunk, is_content_too_short)

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?"

index = LegalIndex(INDEX)
retriever = LegalRetriever(index)
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

resp = retriever.search(QUESTION, strategy="structural", retrieval_mode="hybrid",
                        limit=10, generate_answer=False)
print(f"Passages AVANT filtre : {len(resp.passages)}")
print()
print("Détail du filtre (limit=10) :")
kept, rejected = filter_passages(resp.passages)
for i, cand in enumerate(resp.passages, 1):
    c = cand.chunk
    is_h = is_header_chunk(c)
    is_s = is_content_too_short(c)
    verdict = "GARDE" if (not is_h and not is_s) else "SUPPRIME"
    reason = []
    if is_h: reason.append("en-tete")
    if is_s: reason.append("trop court")
    print(f"  [{i}] {verdict:<8} {'/'.join(reason) or '-':<20} "
          f"art={c.article_number!r:<8} p={c.page_number} "
          f"len={len((c.text or '').strip())}")
    if verdict == "SUPPRIME":
        print(f"        {(c.text or '')[:90].replace(chr(10),' ')!r}")

print(f"\nTotal : {len(kept)} gardes / {len(rejected)} supprimes sur {len(resp.passages)} initiaux")

# Après process_response complet
cleaned = process_response(resp, QUESTION)
print(f"\nApres process_response : {len(cleaned.passages)} passages affiches")
print(f"Avertissements : {cleaned.warnings}")
index.close()
