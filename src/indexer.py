import json
import re
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions


class ParentChildIndexer:
    """
    Parses S3-style document dictionaries, creates parent-child mappings using
    regex line-splitting and artifact filtering, and indexes the chunks into ChromaDB.
    """

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

    def process_and_index_s3_documents(
        self, raw_s3_bucket: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs your exact notebook parsing logic:
        1. Builds parent_document_store keyed by storage_uri.
        2. Extracts child chunks via regex split with length filtering.
        3. Serializes metadata (ACLs, title) for ChromaDB storage.
        """
        parent_document_store = {}
        all_child_chunks = []

        for doc in raw_s3_bucket:
            # Primary unique key across distributed landscape
            parent_id = doc["storage_uri"]

            # Store the pristine, complete raw string as Parent Context
            parent_document_store[parent_id] = {
                "text": doc["content"],
                "title": doc["title"],
                "acl": doc["acl_permitted_roles"]
            }

            # Parser Layer: Extract child rows/sentences
            raw_lines = re.split(r'(?<=[.!?])\s+|\n', doc["content"])

            for index, line in enumerate(raw_lines):
                clean_line = line.strip()
                # Filter out raw layout artifact noise (< 15 chars)
                if len(clean_line) < 15:
                    continue

                all_child_chunks.append({
                    "child_id": f"{parent_id}#chunk_{index}",
                    "parent_id": parent_id,
                    "text": clean_line,
                    "parent_text": doc["content"],  # Store full context in metadata for retrieval fallback
                    "metadata": {
                        "parent_id": parent_id,
                        "source_title": doc["title"],
                        "permitted_roles": json.dumps(doc["acl_permitted_roles"])
                    }
                })

        print(f"Constructed {len(all_child_chunks)} atomic child nodes linked back to parents.")

        # Upsert into ChromaDB
        if all_child_chunks:
            documents = [chunk["text"] for chunk in all_child_chunks]
            ids = [chunk["child_id"] for chunk in all_child_chunks]
            metadatas = [
                {
                    "parent_id": chunk["parent_id"],
                    "parent_text": chunk["parent_text"],
                    "source_title": chunk["metadata"]["source_title"],
                    "permitted_roles": chunk["metadata"]["permitted_roles"]
                }
                for chunk in all_child_chunks
            ]

            self.collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully indexed {len(all_child_chunks)} chunks into ChromaDB.")

        return parent_document_store
