from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(item in relevant for item in ranked[:k]) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    return next((1 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = sum((1 / math.log2(rank + 1)) for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(ranked: Sequence[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
    }
