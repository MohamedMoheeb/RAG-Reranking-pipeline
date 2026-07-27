from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest


class FlashRankReranker:
    """Re-ranks retrieved candidates using FlashRank cross-encoder ONNX models."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Expects candidates as a list of dicts with 'id' and 'text'.
        Returns top_n re-ranked results with relevance scores.
        """
        if not candidates:
            return []

        rerank_request = RerankRequest(query=query, passages=candidates)
        results = self.ranker.rerank(rerank_request)
        return results[:top_n]
