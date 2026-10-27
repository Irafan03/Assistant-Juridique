"""Evaluation harness for corpus-verified judgments; no synthetic labels included."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import evaluate
from .retrieval import LegalRetriever


def run_evaluation(
    configurations: dict[str, tuple[LegalRetriever, str, str]],
    cases_path: str | Path,
    *,
    k: int = 5,
) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if len(cases) != 20:
        raise ValueError(f"Le jeu de référence doit contenir exactement 20 questions annotées; trouvé: {len(cases)}")
    rows: list[dict[str, Any]] = []
    for case in cases:
        query = case["query"]
        for label, (retriever, strategy, mode) in configurations.items():
            response = retriever.search(query, strategy=strategy, retrieval_mode=mode, limit=max(k, 10))
            for target, items, field in (
                ("circulaires", response.circulares, "relevant_circulars"),
                ("textes_associes", response.associated_texts, "relevant_laws"),
                ("articles", response.articles, "relevant_articles"),
            ):
                expected = set(case.get(field, []))
                ranked = [item.chunk.chunk_id for item in items]
                rows.append({"question_id": case["id"], "configuration": label, "strategy": strategy, "retrieval_mode": mode, "target": target, "metrics": evaluate(ranked, expected, k)})
    return {"questions": len(cases), "k": k, "results": rows}


def main() -> None:
    raise SystemExit("Configurez les IndexAdapter dans votre projet, puis appelez run_evaluation() avec vos annotations vérifiées.")
