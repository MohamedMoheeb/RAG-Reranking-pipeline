from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions


class ParentChildIndexer:
    """Indexes child chunks in ChromaDB while linking them to larger parent contexts."""

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
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_parent_child_documents(self, parent_child_pairs: List[Dict[str, str]]):
        """
        Expects a list of dicts:
        [{"parent_id": "p1", "parent_text": "...", "child_id": "c1", "child_text": "..."}, ...]
        """
        documents = [item["child_text"] for item in parent_child_pairs]
        ids = [item["child_id"] for item in parent_child_pairs]
        metadatas = [
            {"parent_id": item["parent_id"], "parent_text": item["parent_text"]}
            for item in parent_child_pairs
        ]

        # Use upsert to avoid duplicate key errors when re-running scripts
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully indexed/updated {len(documents)} child chunks linked to parent contexts.")
