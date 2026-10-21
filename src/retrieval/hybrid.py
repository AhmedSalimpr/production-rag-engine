"""Hybrid retrieval engine combining Dense and BM25 search via Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict, Any, Optional
from src.config import settings
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Orchestrates multi-index dense + sparse search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()
        self.rrf_k = rrf_k or settings.rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Calculates RRF score: sum(weight / (k + rank)) across ranking channels."""
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rank_breakdowns: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Rankings
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
                rank_breakdowns[cid] = {"dense_rank": None, "sparse_rank": None, "dense_score": 0.0, "sparse_score": 0.0}

            rank_breakdowns[cid]["dense_rank"] = rank
            rank_breakdowns[cid]["dense_score"] = item["score"]
            rrf_val = dense_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_val

        # 2. Process Sparse BM25 Rankings
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
                rank_breakdowns[cid] = {"dense_rank": None, "sparse_rank": None, "dense_score": 0.0, "sparse_score": 0.0}

            rank_breakdowns[cid]["sparse_rank"] = rank
            rank_breakdowns[cid]["sparse_score"] = item["score"]
            rrf_val = sparse_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_val

        # Sort by total fused RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_candidates: List[Dict[str, Any]] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            base_item = dict(chunk_map[cid])
            base_item["rrf_score"] = float(fused_scores[cid])
            base_item["fused_rank"] = rank
            base_item["dense_rank"] = rank_breakdowns[cid]["dense_rank"]
            base_item["sparse_rank"] = rank_breakdowns[cid]["sparse_rank"]
            base_item["dense_score"] = rank_breakdowns[cid]["dense_score"]
            base_item["sparse_score"] = rank_breakdowns[cid]["sparse_score"]
            final_candidates.append(base_item)

        return final_candidates

    def search(
        self,
        query: str,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        fused_top_k: int = 15,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Runs parallel dense and sparse search, then fuses via RRF."""
        dense_results = self.dense.search(query, top_k=dense_top_k)
        sparse_results = self.sparse.search(query, top_k=sparse_top_k)

        fused_results = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=fused_top_k,
        )

        return {
            "query": query,
            "fused_results": fused_results,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
        }
