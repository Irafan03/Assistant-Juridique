"""Diagnostic 1c : top-20 avant post-traitement avec scores vectoriel/BM25/RRF/boost titre."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.postprocess import is_header_chunk, clean_article_number
from legal_rag.answer import _key_terms, _overlap, _fold
from legal_rag.concepts import expand_query_terms

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?"

index = LegalIndex(INDEX)
retriever = LegalRetriever(index)

# Préchauffage
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

# Recherche brute (avant postprocess) — on appelle directement pour récupérer les passages
print("=" * 78)
print("1c) TOP-20 AVANT POST-TRAITEMENT (structural / hybrid)")
print("=" * 78)
resp = retriever.search(QUESTION, strategy="structural", retrieval_mode="hybrid",
                        limit=20, generate_answer=False)

# Score vectoriel approximé : 1/(60+rank_sem), BM25 : 1/(60+rank_lex)
# On recalcule les composantes pour le diagnostic
pool = list(index.iter_chunks(strategy="structural"))
semantic = list(index.search(QUESTION, strategy="structural", limit=80))
lexical = retriever._lexical_search(QUESTION, pool, 80)
sem_rank = {c.chunk_id: r for r, c in enumerate(semantic, 1)}
lex_rank = {c.chunk_id: r for r, (c, _) in enumerate(lexical, 1)}

base_terms = _key_terms(QUESTION)
expanded, is_domain, _ = expand_query_terms(QUESTION, base_terms)

print(f"\nquestion: {QUESTION}")
print(f"terme cles (base)   : {base_terms[:8]}")
print(f"terme cles (etendus): {expanded[:12]} (is_domain={is_domain})")
print(f"pool chunks         : {len(pool)}")
print(f"passages retournes  : {len(resp.passages)}")

print(f"\n{'#':>2} {'chunk':<26} {'type':<10} {'art':<8} {'p':>4} "
      f"{'sem':>7} {'bm25':>7} {'rrf':>7} {'title':>6} {'text':>5}")
print("-" * 96)
for i, cand in enumerate(resp.passages[:20], 1):
    c = cand.chunk
    sem = 1/(60+sem_rank.get(c.chunk_id, 999)) if c.chunk_id in {x.chunk_id for x in semantic} else 0
    bm = 1/(60+lex_rank.get(c.chunk_id, 999)) if c.chunk_id in {x.chunk_id for x,_ in lexical} else 0
    title_hay = _fold(c.title or "")
    text_hay = _fold(c.text or "")
    title_hit = sum(1 for t in expanded if __import__('re').search(r"\b"+__import__('re').escape(t)+r"\b", title_hay))
    text_hit = sum(1 for t in expanded if __import__('re').search(r"\b"+__import__('re').escape(t)+r"\b", text_hay))
    art = clean_article_number(c.article_number)
    is_h = is_header_chunk(c)
    flag = "[H]" if is_h else "   "
    print(f"{i:>2} {c.chunk_id[:24]:<26} {(c.document_type or '')[:9]:<10} "
          f"{str(art):<8} {str(c.page_number):>4} {sem:>7.4f} {bm:>7.4f} "
          f"{cand.score:>7.4f} {title_hit:>6} {text_hit:>5} {flag}")
    print(f"     {(c.title or '')[:60]}")
