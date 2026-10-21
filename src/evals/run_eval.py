"""CLI runner for automated RAG evaluation harness."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from src.ingestion.loader import DocumentLoader
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import ReRanker
from src.generation.generator import GroundedGenerator
from src.evals.metrics import compute_retrieval_metrics, aggregate_benchmark_metrics


def run_evaluation() -> Dict[str, Any]:
    """Runs end-to-end evaluation benchmark over the golden test set."""
    print("=" * 70)
    print("  PRODUCTION RAG EVALUATION HARNESS — BENCHMARK RUNNER")
    print("=" * 70)

    # 1. Ingest Knowledge Base
    loader = DocumentLoader()
    chunks = loader.load_seed_documents()
    print(f"[*] Loaded {len(chunks)} chunks from technical knowledge base.")

    dense = DenseRetriever()
    dense.index_chunks(chunks)
    print("[*] Indexed chunks into Qdrant vector database.")

    sparse = SparseRetriever()
    sparse.index_chunks(chunks)
    print("[*] Built BM25 lexical inverted index.")

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    reranker = ReRanker()
    generator = GroundedGenerator()

    # 2. Load Golden Dataset
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"[*] Evaluating {len(golden_set)} test cases (including negative controls)...\n")

    eval_runs: List[Dict[str, Any]] = []

    for item in golden_set:
        qid = item["id"]
        question = item["question"]
        target_doc = item["target_doc_id"]
        is_neg = item.get("is_negative", False)

        t0 = time.perf_counter()
        # Hybrid search
        search_res = hybrid.search(question, dense_top_k=15, sparse_top_k=15, fused_top_k=15)
        candidates = search_res["fused_results"]

        # Rerank
        top_chunks = reranker.rerank(question, candidates, top_k=5)
        retrieval_latency = (time.perf_counter() - t0) * 1000

        # Generate
        rag_response = generator.generate(question, top_chunks, retrieval_latency_ms=retrieval_latency)

        # Compute metrics
        retrieval_metrics = compute_retrieval_metrics(top_chunks, target_doc, k=5)

        run_record = {
            "id": qid,
            "question": question,
            "target_doc_id": target_doc,
            "recall_at_5": retrieval_metrics["recall_at_k"],
            "reciprocal_rank": retrieval_metrics["reciprocal_rank"],
            "faithfulness": rag_response.faithfulness_score,
            "citation_precision": rag_response.citation_precision,
            "retrieval_latency_ms": rag_response.retrieval_latency_ms,
            "total_latency_ms": rag_response.total_latency_ms,
            "answer_preview": rag_response.answer[:80] + "...",
            "citations_count": len(rag_response.citations),
            "is_negative": is_neg,
        }
        eval_runs.append(run_record)
        status_flag = "PASS" if run_record["recall_at_5"] >= 1.0 or is_neg else "WARN"
        print(f"[{status_flag}] {qid} | R@5: {run_record['recall_at_5']} | Faithfulness: {run_record['faithfulness']} | Citations: {run_record['citations_count']} | Latency: {run_record['total_latency_ms']}ms")

    # 3. Aggregate Scorecard
    scorecard = aggregate_benchmark_metrics(eval_runs)

    print("\n" + "=" * 70)
    print("  AGGREGATE BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"  Total Test Cases Evaluated: {scorecard['total_test_cases']}")
    print(f"  Recall@5:                   {scorecard['recall_at_5'] * 100:.1f}%")
    print(f"  Mean Reciprocal Rank (MRR): {scorecard['mrr']:.3f}")
    print(f"  Context Faithfulness:       {scorecard['mean_faithfulness'] * 100:.1f}%")
    print(f"  Citation Precision:         {scorecard['mean_citation_precision'] * 100:.1f}%")
    print(f"  p50 End-to-End Latency:     {scorecard['p50_latency_ms']:.1f} ms")
    print(f"  p95 End-to-End Latency:     {scorecard['p95_latency_ms']:.1f} ms")
    print(f"  Est. Cost per 1k Queries:   ${scorecard['cost_per_1k_queries_usd']:.4f}")
    print("=" * 70)

    # Save to file
    output_path = Path("eval_results.json")
    final_output = {
        "timestamp": time.time(),
        "scorecard": scorecard,
        "runs": eval_runs,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[+] Detailed benchmark report saved to {output_path.absolute()}")
    return final_output


if __name__ == "__main__":
    run_evaluation()
