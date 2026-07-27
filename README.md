# RAG-Reranking-pipeline
# Production-Grade RAG Pipeline with Two-Stage Re-Ranking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)
![FlashRank](https://img.shields.io/badge/Re--Ranking-FlashRank%2FRankify-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to minimize hallucinations and optimize context retrieval accuracy using vector search combined with a ultra-fast re-ranking stage.

---

## Architecture Overview

Standard RAG architectures often suffer from low retrieval precision when relying solely on cosine similarity over dense vector embeddings. This repository implements a **Two-Stage Retrieval Pipeline**:
[ User Query ] ──► [ Dense Retrieval (ChromaDB + Sentence-Transformers) ]
│
▼  (Top-K Documents)
[ Re-Ranking Stage (FlashRank / Rankify) ]
│
▼  (Top-N Ultra-Relevant Contexts)
[ LLM Context Window Generation ]
1. **Stage 1 (Dense Vector Search)**: Quickly retrieves the top candidate documents using `sentence-transformers` and `ChromaDB`.
2. **Stage 2 (Re-Ranking)**: Applies cross-encoder models (`FlashRank` / `Rankify`) to compute fine-grained semantic relevance scores, filtering out irrelevant chunks before passing context to the LLM.

---

## Key Features

* **High-Precision Context Retrieval**: Combines semantic embedding search with cross-encoder re-ranking.
* **Low Latency Re-Ranking**: Integrated ONNX-optimized `FlashRank` for real-time applications without heavy GPU overhead.
* **Modular & Scalable**: Embedded `ChromaDB` setup allowing seamless transition to production vector storages.

---

## Tech Stack

* **Language**: Python 3.10+
* **Vector Store**: [ChromaDB](https://www.trychroma.com/)
* **Embeddings**: [Sentence-Transformers](https://sbert.net/)
* **Re-Ranking Engines**: [FlashRank](https://github.com/Prithivida/FlashRank), [Rankify](https://github.com/rankify-ai/rankify)

---

## Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed.

```bash
git clone [https://github.com/YOUR_USERNAME/rag-rerank-pipeline.git](https://github.com/YOUR_USERNAME/rag-rerank-pipeline.git)
cd rag-rerank-pipeline
