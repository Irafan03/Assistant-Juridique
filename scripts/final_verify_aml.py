"""Vérification finale propre : question ticket après pipeline complet."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator
from legal_rag.postprocess import process_response

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
QUESTION = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment ?"

index = LegalIndex(INDEX)
gen   = FaithfulAnswerGenerator()
base  = LegalRetriever(index, answer_generator=gen)

for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

t0       = time.perf_counter()
resp_raw = base.search(QUESTION, strategy="structural", retrieval_mode="hybrid",
                       limit=10, generate_answer=False)
resp     = process_response(resp_raw, QUESTION)
answer   = gen.generate(QUESTION, evidence=resp.passages)
elapsed  = time.perf_counter() - t0

d = resp.to_dict()

print(f"Temps : {elapsed:.1f}s | "
      f"passages={len(d['passages'])} circ={len(d['circulaires'])} "
      f"assoc={len(d['textes_associes'])} art={len(d['articles'])}")
print(f"Avertissements : {d['warnings']}")

# Vérifications strictes
header_in  = [p for p in d['passages'] if 'RECUEIL DES TEXTES' in (p.get('text') or '')]
from collections import Counter
dup        = [k for k, v in Counter(p['text'][:80] for p in d['passages']).items() if v > 1]
art_clean  = all(p['article_number'] is None or len(str(p['article_number'])) <= 4
                 for p in d['passages'])

# Compteurs
chunk_ids = {p['chunk_id'] for p in d['passages']}
doc_ids   = {p['document_id'] for p in d['passages']}
refs      = {r for p in d['passages'] for r in (p.get('references') or [])}

print(f"\nCompteurs affichés : {len(chunk_ids)} passages | {len(doc_ids)} documents | {len(refs)} refs")
print(f"En-têtes dans passages : {len(header_in)}  (cible=0)")
print(f"Doublons de contenu    : {len(dup)}  (cible=0)")
print(f"Art. propres (≤4 car.)  : {art_clean}  (cible=True)")

# Circulaire 5/W/2017 dans les passages ?
circ517 = [p for p in d['passages']
           if '5/W/2017' in (p.get('title') or '') or '5/w/2017' in (p.get('title') or '').lower()]
print(f"Passages circulaire 5/W/2017 : {len(circ517)}")

print("\n--- Passages ---")
for i, p in enumerate(d['passages'], 1):
    art = p['article_number'] or '—'
    print(f"  [{i}] {p['document_type']:<10} art={art:<4} p={p['page_number']}")
    print(f"       {(p['title'] or '')[:55]}")
    print(f"       {p['text'][:90].replace(chr(10),' ')}")

print("\n--- Réponse générée ---")
print(answer or "(aucune)")

# Qualité
if answer:
    has_obligation_verb = any(w in answer.lower() for w in
                              ['doit', 'sont tenus', 'doivent', 'est tenu', 'sont obliges', 'nomme'])
    has_art50   = 'Art. 50' in answer or 'article 50' in answer.lower()
    has_art18   = 'Art. 18' in answer or 'soupçon' in answer.lower() or 'soupcon' in answer.lower()
    no_header   = 'RECUEIL DES TEXTES' not in answer
    print(f"\n--- Qualité ---")
    print(f"  Verbe obligation           : {has_obligation_verb}")
    print(f"  Art. 50 (responsable AML)  : {has_art50}")
    print(f"  Soupçon cité               : {has_art18}")
    print(f"  Aucun en-tête              : {no_header}")

index.close()
