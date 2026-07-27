from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions


class VectorRetriever:
    """Retrieves candidate document chunks from ChromaDB using dense embeddings."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
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
        """Queries ChromaDB and formats top-k retrieved contexts."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        candidates = []
        if results["documents"]:
            for doc, doc_id in zip(results["documents"][0], results["ids"][0]):
                candidates.append({"id": doc_id, "text": doc})

        return candidates