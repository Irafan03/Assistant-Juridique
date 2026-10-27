from legal_rag.metrics import evaluate, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    ranked, relevant = ["a", "x", "b"], {"b", "a"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 3) > 0.8
    assert set(evaluate(ranked, relevant)) == {"precision@5", "recall@5", "mrr", "ndcg@5"}
