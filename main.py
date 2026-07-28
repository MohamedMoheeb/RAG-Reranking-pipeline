import os
from src import ParentChildIndexer, VectorRetriever, FlashRankReranker, GeminiGenerator


def run_pipeline():
    # 1. Sample Data with Parent-Child Structure
    data_pairs = [
        {
            "storage_uri": "s3://central-bank-vault/policies/risk_matrix_2026.txt",
            "title": "2026 Internal Risk Mitigation and Infrastructure Architecture Protocols",
            "acl_permitted_roles": ["Compliance_Officer", "Audit_Team", "System_Admin"],
            "content": (
                "This document dictates infrastructure safety criteria. When processing capital transfers, "
                "systems must cross-reference core ledger systems. Below is the active asset classification matrix:\n\n"
                "| Component ID | Subsystem Target | Risk Vulnerability Level |\n"
                "| :--- | :--- | :--- |\n"
                "| AUTH-881 | User Credentials Layer | Medium |\n"
                "| RISK-992-ALPHA | High-Frequency Transaction Pipeline | CRITICAL |\n"
                "| STOR-004 | In-Memory Vault Mirroring | High |\n\n"
                "Warning: Any structural unexpected downtime on asset RISK-992-ALPHA requires immediate rolling "
                "hot-swaps to secondary nodes to prevent international reconciliation clearing failures."
            )
        },
        {
            "storage_uri": "s3://central-bank-vault/hr/executive_compensation_q1.txt",
            "title": "Q1 Executive Leadership Compensations and Active Payroll Tier Schema",
            "acl_permitted_roles": ["Executive_Board", "HR_Director"],
            "content": (
                "Private Document - Strictly Restricted. Executive Salary distribution schedules for the fiscal year 2026. "
                "Chief Executive Officer base compensation is set at $450,000 per quarter. Chief Technology Officer "
                "base tier calculation stands at $380,000 per quarter. Discretionary performance bonuses are bound "
                "by the asset allocation thresholds specified under HR-COMP-2026."
            )
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
