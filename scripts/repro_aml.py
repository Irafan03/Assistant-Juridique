"""Reproduction exacte du problème AML — pipeline PRODUCTION (avec PostProcessedRetriever)."""
import sys, pathlib, json, re
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator
from legal_rag.postprocess import process_response, is_header_chunk

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?"

index = LegalIndex(INDEX)
gen   = FaithfulAnswerGenerator()
base  = LegalRetriever(index, answer_generator=gen)

# Préchauffage cache
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

# --- Pipeline PRODUCTION (identique à PostProcessedRetriever dans server.py)
resp_raw  = base.search(QUESTION, strategy="structural", retrieval_mode="hybrid",
                        limit=10, generate_answer=False)
resp      = process_response(resp_raw, QUESTION)   # ← filtrage + dédup + rerank
answer    = gen.generate(QUESTION, evidence=resp.passages)
d         = resp.to_dict()

HEADER_RE = re.compile(
    r"RECUEIL DES TEXTES|MIS À JOUR|ORGANISMES ASSIMILES|"
    r"ETABLISSEMENTS DE CREDIT ET ORGANISMES|PUBLIC\s*$",
    re.I
)

print("=" * 78)
print("QUESTION :", QUESTION)
print("=" * 78)
print(f"passages={len(d['passages'])} circ={len(d['circulaires'])} "
      f"assoc={len(d['textes_associes'])} art={len(d['articles'])}")

# --- Anomalie 2 : doublons de contenu
print("\n--- DOUBLONS DE CONTENU (top-10 passages) ---")
seen_text = {}
for i, p in enumerate(d["passages"], 1):
    snippet = p["text"][:80].replace("\n", " ")
    if snippet in seen_text:
        print(f"  [!] [{i}] = [{seen_text[snippet]}] doublon exact : {snippet!r}")
    else:
        seen_text[snippet] = i

# --- Anomalie 3 : chunks en-tête / titre seul
print("\n--- CHUNKS EN-TÊTE / TITRE SEUL ---")
for i, p in enumerate(d["passages"], 1):
    text = p["text"].strip()
    is_header = bool(HEADER_RE.search(text))
    is_short  = len(text) < 80
    if is_header or is_short:
        print(f"  [!] [{i}] is_header={is_header} is_short={is_short} "
              f"art={p['article_number']!r} len={len(text)}")
        print(f"      {text[:100]!r}")

# --- Anomalie 4 : article_number vide / doublon de loi
print("\n--- ARTICLES VIDES ET DOUBLONS DE LOI ---")
doc_ids_seen = {}
for i, p in enumerate(d["passages"], 1):
    if not p["article_number"]:
        print(f"  [!] [{i}] article_number vide | {(p['title'] or '')[:60]}")
    if p["document_id"] in doc_ids_seen:
        print(f"  [!] [{i}] document_id déjà vu : {p['document_id'][:50]}")
    doc_ids_seen[p["document_id"]] = i

# --- Anomalie 5 : compteurs
total_passage_ids = {p["chunk_id"] for cat in
    (d["passages"], d["circulaires"], d["textes_associes"], d["articles"])
    for p in cat}
total_doc_ids = {p["document_id"] for cat in
    (d["passages"], d["circulaires"], d["textes_associes"], d["articles"])
    for p in cat}
total_refs = {r for cat in
    (d["passages"], d["circulaires"], d["textes_associes"], d["articles"])
    for p in cat for r in (p.get("references") or [])}
print(f"\n--- COMPTEURS ---")
print(f"  passages uniques  : {len(total_passage_ids)}")
print(f"  documents uniques : {len(total_doc_ids)}")
print(f"  références uniques: {len(total_refs)}")

print("\n--- TOP 10 PASSAGES ---")
for i, p in enumerate(d["passages"], 1):
    print(f"[{i}] {p['citation_id']} type={p['document_type']} "
          f"art={p['article_number']!r} page={p['page_number']} score={p['score']:.4f}")
    print(f"    title : {(p['title'] or '')[:65]}")
    print(f"    text  : {p['text'][:100].replace(chr(10), ' ')!r}")

print("\n--- RÉPONSE GÉNÉRÉE ---")
print(answer or "(aucune)")

index.close()
