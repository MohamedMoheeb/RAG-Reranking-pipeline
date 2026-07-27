from src.indexer import DocumentIndexer
from src.retriever import VectorRetriever
from src.reranker import FlashRankReranker


def run_pipeline():
    # 1. Sample Data Setup
    docs = [
        "Retrieval-Augmented Generation (RAG) improves LLM responses using dynamic external search.",
        "FlashRank is an ONNX-backed, ultra-lightweight re-ranking engine designed for low latency.",
        "Sentence Transformers provide dense embedding representations for text similarity tasks.",
        "ChromaDB is an open-source, embedded vector store optimized for developers."
    ]

    # 2. Index Documents
    indexer = DocumentIndexer()
    indexer.add_documents(documents=docs)

    # 3. Dense Retrieval (Stage 1: Top-K)
    retriever = VectorRetriever()
    query = "How does FlashRank speed up search re-ranking?"
    candidates = retriever.retrieve(query=query, top_k=4)

    # 4. Cross-Encoder Re-Ranking (Stage 2: Top-N)
    reranker = FlashRankReranker()
    reranked_results = reranker.rerank(query=query, candidates=candidates, top_n=2)

    # Output Top Contexts
    print(f"\nQuery: {query}\n" + "-"*40)
    for idx, item in enumerate(reranked_results, start=1):
        print(f"Rank {idx} | Score: {item.get('score', 0):.4f} | Text: {item['text']}")


if __name__ == "__main__":
    run_pipeline()