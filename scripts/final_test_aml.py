"""Test final : question exacte du ticket, pipeline production complet."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator, _key_terms, _overlap
from legal_rag.postprocess import process_response, clean_article_number, is_header_chunk
from legal_rag.concepts import expand_query_terms

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?"

index = LegalIndex(INDEX)
gen   = FaithfulAnswerGenerator()
base  = LegalRetriever(index, answer_generator=gen)

# Préchauffage cache
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

# Pipeline production (identique à PostProcessedRetriever dans server.py)
t0 = time.perf_counter()
resp_raw = base.search(QUESTION, strategy="structural", retrieval_mode="hybrid",
                       limit=10, generate_answer=False)
resp     = process_response(resp_raw, QUESTION)
answer   = gen.generate(QUESTION, evidence=resp.passages)
elapsed  = time.perf_counter() - t0

d = resp.to_dict()

print("=" * 78)
print("QUESTION TEST (ticket) :", QUESTION)
print(f"Temps total : {elapsed:.1f}s")
print("=" * 78)
print(f"passages={len(d['passages'])} circ={len(d['circulaires'])} "
      f"assoc={len(d['textes_associes'])} art={len(d['articles'])}")
print(f"warnings : {d['warnings']}")

# Compteurs affichés (basés sur passages uniquement, comme le frontend)
chunk_ids = {p['chunk_id'] for p in d['passages']}
doc_ids   = {p['document_id'] for p in d['passages']}
refs      = {r for p in d['passages'] for r in (p.get('references') or [])}
print(f"\nCompteurs affiches : {len(chunk_ids)} passages | "
      f"{len(doc_ids)} documents | {len(refs)} references juridiques")

# Vérification : aucun en-tête, aucun doublon
print("\n--- VERIFICATIONS ---")
header_leak = [p for p in d['passages'] if 'RECUEIL DES TEXTES' in (p.get('text') or '')]
print(f"En-tetes dans les passages : {len(header_leak)} (attendu 0)")

from collections import Counter
text_prefixes = [p['text'][:80] for p in d['passages']]
dup = Counter(text_prefixes)
dups = {k: v for k, v in dup.items() if v > 1}
print(f"Doublons de contenu      : {len(dups)} (attendu 0)")

# Top passages
print("\n--- TOP PASSAGES (apres postprocess) ---")
for i, p in enumerate(d['passages'], 1):
    art = clean_article_number(p['article_number'])
    print(f"  [{i}] {p['citation_id']} {p['document_type']:<10} art={art!r:<6} "
          f"p={p['page_number']} score={p['score']:.4f}")
    print(f"      {(p['title'] or '')[:55]}")
    print(f"      {p['text'][:95].replace(chr(10),' ')}")

# Réponse générée
print("\n--- REPONSE GENEREE ---")
print(answer or "(aucune)")

# Vérifier la présence d'obligations réelles
print("\n--- QUALITE DE LA REPONSE ---")
if answer:
    has_obligation = any(w in answer.lower() for w in
                         ['doit', 'sont tenus', 'est tenu', 'doivent', 'sont obliges'])
    has_citation = '[' in answer and ']' in answer
    has_themes = '■' in answer
    no_header = 'RECUEIL DES TEXTES' not in answer
    print(f"  Contient un verbe d'obligation : {has_obligation}")
    print(f"  Contient des citations [n]      : {has_citation}")
    print(f"  Organise par themes (■)        : {has_themes}")
    print(f"  Aucun en-tete                  : {no_header}")
    # Articles AML reels presents ?
    aml_arts = [p for p in d['passages']
                if p['page_number'] and 636 <= p['page_number'] <= 650
                and p['document_type'] == 'circulaire']
    print(f"  Passages de la circulaire 5/W/2017 (pages 636-650) : {len(aml_arts)}")

index.close()
