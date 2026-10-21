"""Evaluation metrics calculations for RAG benchmarks."""

import numpy as np
from typing import List, Dict, Any


def compute_retrieval_metrics(results: List[Dict[str, Any]], target_doc_id: str, k: int = 5) -> Dict[str, float]:
    """Computes Recall@k and Reciprocal Rank for a single query."""
    if not target_doc_id:
        return {"recall_at_k": 1.0, "reciprocal_rank": 1.0}

    top_k_results = results[:k]
    doc_ids = [item.get("doc_id") for item in top_k_results]

    is_recalled = 1.0 if target_doc_id in doc_ids else 0.0

    reciprocal_rank = 0.0
    for rank, item in enumerate(results, start=1):
        if item.get("doc_id") == target_doc_id:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_k": is_recalled,
        "reciprocal_rank": reciprocal_rank,
    }


def aggregate_benchmark_metrics(eval_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates individual query evaluations into a comprehensive scorecard."""
    if not eval_runs:
        return {}

    recalls = [r["recall_at_5"] for r in eval_runs]
    reciprocal_ranks = [r["reciprocal_rank"] for r in eval_runs]
    faithfulness_scores = [r["faithfulness"] for r in eval_runs]
    citation_precisions = [r["citation_precision"] for r in eval_runs]
    latencies = [r["total_latency_ms"] for r in eval_runs]

    mean_recall = float(np.mean(recalls))
    mrr = float(np.mean(reciprocal_ranks))
    mean_faithfulness = float(np.mean(faithfulness_scores))
    mean_precision = float(np.mean(citation_precisions))

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    avg_latency = float(np.mean(latencies))

    # Cost model: average 800 prompt tokens + 200 completion tokens per query
    # GPT-4o-mini pricing: $0.15 / 1M prompt, $0.60 / 1M completion
    prompt_tokens = 800
    completion_tokens = 200
    cost_per_query = (prompt_tokens * 0.15 / 1_000_000) + (completion_tokens * 0.60 / 1_000_000)
    cost_per_1k_queries = round(cost_per_query * 1000, 4)

    return {
        "total_test_cases": len(eval_runs),
        "recall_at_5": round(mean_recall, 4),
        "mrr": round(mrr, 4),
        "mean_faithfulness": round(mean_faithfulness, 4),
        "mean_citation_precision": round(mean_precision, 4),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "cost_per_1k_queries_usd": cost_per_1k_queries,
    }
