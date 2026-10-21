"""Re-ranking engine to rescore and prune candidate chunks for LLM context."""

import re
from typing import List, Dict, Any, Optional
from src.config import settings


class ReRanker:
    """Rescores retrieved candidates to isolate top high-relevance chunks."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k or settings.rerank_top_k

    def _compute_relevance_score(self, query: str, content: str, base_rrf: float) -> float:
        """Heuristic cross-scoring modeling token coverage, bigram matching, and title density."""
        query_words = set(re.findall(r"\b[a-z0-9_]+\b", query.lower()))
        if not query_words:
            return base_rrf

        content_lower = content.lower()
        exact_matches = sum(1 for w in query_words if w in content_lower)
        coverage_ratio = exact_matches / len(query_words)

        # Check for multi-word phrase matching
        phrase_bonus = 0.2 if query.lower() in content_lower else 0.0

        # Composite normalized score
        normalized_rrf = min(1.0, base_rrf * 30.0)
        final_score = (0.5 * coverage_ratio) + (0.3 * normalized_rrf) + phrase_bonus
        return round(final_score, 4)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reranks candidate chunks and truncates to top_k items."""
        limit = top_k or self.top_k
        if not candidates:
            return []

        rescored: List[Dict[str, Any]] = []
        for rank, item in enumerate(candidates, start=1):
            base_rrf = item.get("rrf_score", 0.0)
            score = self._compute_relevance_score(query, item["content"], base_rrf)

            item_copy = dict(item)
            item_copy["rerank_score"] = score
            item_copy["pre_rerank_rank"] = rank
            rescored.append(item_copy)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)

        for i, item in enumerate(rescored[:limit], start=1):
            item["final_rank"] = i

        return rescored[:limit]
