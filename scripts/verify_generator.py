"""Vérification du générateur fidèle après corrections (word-boundary + accents)."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legal_rag.indexing import LegalIndex
from legal_rag.retrieval import LegalRetriever
from legal_rag.answer import FaithfulAnswerGenerator, _key_terms, _overlap

INDEX = pathlib.Path(__file__).parent.parent / "data" / "index"
index = LegalIndex(INDEX)
retriever = LegalRetriever(index, answer_generator=FaithfulAnswerGenerator())

for s in ("structural",):
    _ = list(index.iter_chunks(strategy=s))

QUESTIONS = [
    ("fonds-propres", "Quelles règles prudentielles encadrent les fonds propres et la solvabilité des établissements de crédit ?"),
    ("comptes-inactifs", "Quelles sont les règles relatives aux comptes inactifs ?"),
    ("hors-corpus", "Quelle est la recette de la tarte aux pommes ?"),
    ("accent-test", "Quelles sont les regles prudentielles pour les fonds propres et la solvabilite ?"),  # sans accents
]

for label, q in QUESTIONS:
    resp = retriever.search(q, strategy="structural", retrieval_mode="hybrid",
                            limit=10, generate_answer=True)
    key_terms = _key_terms(q)
    print(f"\n{'='*72}")
    print(f"[{label}] termes: {key_terms[:6]}")
    print(f"  answer non vide: {resp.answer is not None}")
    if resp.answer:
        is_insufficient = "ne permet pas d'établir une réponse fiable" in (resp.answer or "")
        print(f"  reponse insuffisante signalee: {is_insufficient}")
        print(f"  --- APERCU REPONSE ---")
        print((resp.answer or "")[:400].replace("\n", " "))
    if resp.warnings:
        print(f"  warnings: {resp.warnings}")

index.close()
