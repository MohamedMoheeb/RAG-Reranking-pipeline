import os
from src import ParentChildIndexer, VectorRetriever, FlashRankReranker, GeminiGenerator


def run_pipeline():
    # 1. Sample Data with Parent-Child Structure
    data_pairs = [
        {
            "parent_id": "parent_1",
            "parent_text": "Retrieval-Augmented Generation (RAG) significantly reduces LLM hallucinations by dynamically fetching verified context from an external vector database before generating an answer. This architectural pattern is widely used in enterprise search engines.",
            "child_id": "child_1",
            "child_text": "RAG reduces LLM hallucinations by dynamically fetching external context."
        },
        {
            "parent_id": "parent_2",
            "parent_text": "FlashRank is a highly efficient cross-encoder re-ranking engine backed by ONNX runtime. It is specifically tailored for ultra-fast, low-latency search re-ranking in resource-constrained environments.",
            "child_id": "child_2",
            "child_text": "FlashRank uses ONNX runtime for ultra-fast, low-latency search re-ranking."
        }
    ]

    # 2. Step 1: Index Chunks
    print("--- Step 1: Indexing Chunks ---")
    indexer = ParentChildIndexer()
    indexer.add_parent_child_documents(data_pairs)

    # 3. Step 2: Dense Similarity Retrieval (Child level)
    print("\n--- Step 2: Dense Retrieval ---")
    retriever = VectorRetriever(collection_name="rag_parent_child")
    query = "How does FlashRank speed up search re-ranking?"
    child_candidates = retriever.retrieve(query=query, top_k=2)

    # Map retrieved Child chunks back to their richer Parent context
    parent_candidates = [
        {
            "id": item.get("metadata", {}).get("parent_id", item["id"]),
            "text": item.get("metadata", {}).get("parent_text", item["text"])
        }
        for item in child_candidates
    ]
    print(f"Retrieved {len(parent_candidates)} candidate parent contexts.")

    # 4. Step 3: Cross-Encoder Re-Ranking (Parent level)
    print("\n--- Step 3: Re-Ranking with FlashRank ---")
    reranker = FlashRankReranker()
    reranked_results = reranker.rerank(query=query, candidates=parent_candidates, top_n=1)
    print(f"Top re-ranked passage selected (Score: {reranked_results[0].get('score', 0.0):.4f})")

    # 5. Step 4: Grounded Generation with Gemini
    print("\n--- Step 4: Generating Answer with Gemini ---")
    generator = GeminiGenerator()
    final_answer = generator.generate_answer(query=query, context_chunks=reranked_results)

    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("="*50)
    print(f"GEMINI ANSWER:\n{final_answer}")
    print("="*50)


if __name__ == "__main__":
    run_pipeline()
