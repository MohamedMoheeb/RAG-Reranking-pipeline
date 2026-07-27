from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions


class VectorRetriever:
    """Performs top-K dense vector similarity retrieval against ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_parent_child",
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        candidates = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

            for doc, doc_id, meta in zip(docs, ids, metadatas):
                candidates.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {}
                })

        return candidates
