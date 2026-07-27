from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest


class FlashRankReranker:
    """Re-ranks retrieved documents using FlashRank cross-encoder models."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Re-ranks candidate passages and returns top-n matches."""
        if not candidates:
            return []

        passages = [{"id": item["id"], "text": item["text"]} for item in candidates]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # FlashRank outputs sorted dictionary list with relevance scores
        return results[:top_n]