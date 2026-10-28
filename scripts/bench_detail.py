"""Mesure avant/après corrections AML sur l'index réel."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator, _key_terms, _overlap
from legal_rag.postprocess import process_response, clean_article_number
from legal_rag.concepts import is_aml_query, expand_query_terms

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"

QUESTION_AML = "Quelles sont les obligations d'une banque en matière de lutte contre le blanchiment d'argent ?"
QUESTION_FONDS = "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?"

index = LegalIndex(INDEX)
for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

gen = FaithfulAnswerGenerator()
retriever_base = LegalRetriever(index)
retriever_gen  = LegalRetriever(index, answer_generator=gen)

print("=" * 78)
print("MESURES SUR L'INDEX RÉEL — strategy=structural mode=hybrid limit=10")
print("=" * 78)

for question, label in [
    (QUESTION_AML,   "AML/blanchiment"),
    (QUESTION_FONDS, "fonds propres"),
]:
    print(f"\n{'─'*78}")
    print(f"[{label}] {question[:70]}")
    print(f"{'─'*78}")

    # ── AVANT (base, sans postprocess) ────────────────────────────────────────
    t0 = time.perf_counter()
    resp_base = retriever_base.search(question, strategy="structural",
                                      retrieval_mode="hybrid", limit=10)
    t_base = time.perf_counter() - t0

    bad_arts_before = sum(1 for p in resp_base.passages
                          if clean_article_number(p.chunk.article_number) is None
                          and p.chunk.article_number is not None)
    key_terms = _key_terms(question)
    expanded, _, _ = expand_query_terms(question, key_terms)
    top1_overlap_before = _overlap(resp_base.passages[0].chunk, expanded) if resp_base.passages else 0
    top5_rel_before = sum(1 for p in resp_base.passages[:5] if _overlap(p.chunk, expanded) > 0)

    print(f"\n  AVANT ({t_base:.1f}s)")
    print(f"    Numéros d'article aberrants : {bad_arts_before}/10")
    print(f"    Top-1 overlap               : {top1_overlap_before}")
    print(f"    Top-5 pertinents (overlap>0): {top5_rel_before}/5")
    print(f"    Top 3 passages :")
    for i, p in enumerate(resp_base.passages[:3], 1):
        print(f"      [{i}] {p.chunk.document_type} art={p.chunk.article_number!r} "
              f"score={p.score:.4f} | {(p.chunk.title or '')[:45]}")

    # ── APRÈS (avec postprocess + génération) ─────────────────────────────────
    t0 = time.perf_counter()
    resp_raw = retriever_gen.search(question, strategy="structural",
                                    retrieval_mode="hybrid", limit=10,
                                    generate_answer=False)
    resp_after = process_response(resp_raw, question)
    # Générer après re-classement
    answer = gen.generate(question, evidence=resp_after.passages)
    t_after = time.perf_counter() - t0

    bad_arts_after = sum(1 for p in resp_after.passages
                         if clean_article_number(p.chunk.article_number) is None
                         and p.chunk.article_number is not None)
    top1_overlap_after = _overlap(resp_after.passages[0].chunk, expanded) if resp_after.passages else 0
    top5_rel_after = sum(1 for p in resp_after.passages[:5] if _overlap(p.chunk, expanded) > 0)

    print(f"\n  APRÈS ({t_after:.1f}s)")
    print(f"    Numéros d'article aberrants : {bad_arts_after}/10")
    print(f"    Top-1 overlap               : {top1_overlap_after}")
    print(f"    Top-5 pertinents (overlap>0): {top5_rel_after}/5")
    print(f"    Top 3 passages :")
    for i, p in enumerate(resp_after.passages[:3], 1):
        print(f"      [{i}] {p.chunk.document_type} art={p.chunk.article_number!r} "
              f"score={p.score:.4f} | {(p.chunk.title or '')[:45]}")
    is_insuf = "ne permet pas d'établir une réponse fiable" in (answer or "")
    print(f"    Réponse : {'insuffisante' if is_insuf else 'synthèse structurée'}")
    if answer and not is_insuf:
        print(f"    --- Extrait réponse (200 car.) ---")
        print(f"    {answer[:200].replace(chr(10), ' ')}")

    # ── Comparaison 3 stratégies (sémantique, moins coûteux) ──────────────────
    print(f"\n  Comparaison stratégies (mode=semantic) :")
    for strategy in ("structural", "hierarchical", "parent_child"):
        t0 = time.perf_counter()
        r = retriever_base.search(question, strategy=strategy,
                                  retrieval_mode="semantic", limit=5)
        elapsed = time.perf_counter() - t0
        r_pp = process_response(r, question)
        top1_ov = _overlap(r_pp.passages[0].chunk, expanded) if r_pp.passages else 0
        bad_a = sum(1 for p in r_pp.passages
                    if clean_article_number(p.chunk.article_number) is None
                    and p.chunk.article_number is not None)
        print(f"    {strategy:<15} {elapsed:.1f}s | "
              f"top1_overlap={top1_ov} bad_art={bad_a}/5 "
              f"| {(r_pp.passages[0].chunk.title or '')[:40] if r_pp.passages else '—'}")

index.close()
