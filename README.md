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

```text
[ User Query ]
       │
       ▼
[ Stage 1: Dense Retrieval (ChromaDB + MiniLM) ] ──► Fetches Top-K Child Chunks
       │
       ▼ (Maps Child Chunks to Parent Contexts)
[ Stage 2: Cross-Encoder Re-Ranking (FlashRank) ] ──► Scores & Ranks Top-N Parent Passages
       │
       ▼
[ Stage 3: Grounded Synthesis (Google Gemini) ]   ──► Generates Hallucination-Free Answer
```

1. **Stage 1 (Dense Vector Search)**: Quickly retrieves the top candidate documents using `sentence-transformers` and `ChromaDB`.
2. **Stage 2 (Re-Ranking)**: Applies cross-encoder models (`FlashRank` / `Rankify`) to compute fine-grained semantic relevance scores, filtering out irrelevant chunks before passing context to the LLM.

---

## Key Features

* **Parent-Child Chunking**: Embeds granular child chunks (100–200 tokens) for high-precision vector search while attaching larger parent contexts (500–1000 tokens) for richer LLM context windows.
* **Two-Stage Search Optimization**: Combines ultra-fast dense similarity search with lightweight ONNX-backed cross-encoders (`ms-marco-MiniLM-L-12-v2`) via FlashRank.
* **Grounded Answer Generation**: Integrates Google Gemini (`gemini-2.5-flash`) via the official `google-genai` SDK to synthesize verified answers strictly bound to retrieved sources.
* **Clean & Modular OOP Design**: Completely decoupled modules for indexing, retrieval, re-ranking, and LLM generation.

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

## Quickstart Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/MohamedMoheeb/RAG-Reranking-pipeline.git](https://github.com/MohamedMoheeb/RAG-Reranking-pipeline.git)
cd RAG-Reranking-pipeline

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Your Gemini API Key
To use the generation stage, obtain an API key from [Google AI Studio](https://aistudio.google.com/) and set it as an environment variable:

* **Linux / macOS:**
  ```bash
  export GEMINI_API_KEY="your_actual_api_key_here"
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="your_actual_api_key_here"
  ```
* **Windows (CMD):**
  ```cmd
  set GEMINI_API_KEY="your_actual_api_key_here"
  ```

### 4. Run the Pipeline
```bash
python main.py
```
---
## How It Works Under the Hood

1. **`ParentChildIndexer` (`src/indexer.py`)**: Stores small child chunks alongside parent text in ChromaDB metadata using `upsert` operations.
2. **`VectorRetriever` (`src/retriever.py`)**: Queries ChromaDB for top child matches and extracts associated parent metadata.
3. **`FlashRankReranker` (`src/reranker.py`)**: Re-ranks parent candidate passages using ONNX cross-encoders for precise relevance scoring.
4. **`GeminiGenerator` (`src/generator.py`)**: Passes re-ranked context into `gemini-2.5-flash` with a strict grounding prompt to prevent hallucinations.

---

## License
Distributed under the **MIT License**. See `LICENSE` for details.
