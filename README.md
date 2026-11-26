# Enterprise RAG System

A secure, multi-organization **Retrieval-Augmented Generation (RAG)** system built for enterprise knowledge management, combining multi-format document ingestion, permission-aware hybrid retrieval, verifiable source citations, and role-based access control (RBAC).

---

## Core Architecture & Features
- **Multi-Format Document Ingestion:** Extracts, normalizes, and recursively chunks `.pdf`, `.docx`, `.csv`, and `.txt` documents while attaching organization, role, source filename, and page metadata.
- **Permission-Aware Hybrid Retrieval:** Combines dense vector similarity search (`PostgreSQL` + `pgvector`) with lexical keyword matching (`BM25` fused via Reciprocal Rank Fusion), enforcing organization and role filters **prior** to LLM context construction.
- **Grounded Generation & Traceable Citations:** Uses local LLMs via `Ollama` (`nomic-embed-text` for embeddings and `qwen2.5:3b` for generation) with strict grounding rules to eliminate hallucinations and cite exact source passages.
- **Full-Stack Enterprise Delivery:** `FastAPI` REST backend secured with JWT authentication, paired with a `React` + `Tailwind CSS` web interface and an automated evaluation benchmark suite.

---

## Project Structure
```text
enterprise-rag-system/
├── backend/
│   ├── api/            # FastAPI routes, JWT authentication & RBAC enforcement
│   ├── ingestion/      # Multi-format loaders, recursive chunker & metadata tagger
│   └── rag/            # pgvector store, hybrid retriever & citation generator
├── frontend/           # React + Tailwind CSS user interface
├── evaluation/         # Retrieval relevance, faithfulness & RBAC test suite
├── docs/               # Sample organizational documents
├── docker-compose.yml  # PostgreSQL + pgvector container setup
├── .env.example        # Environment variable template
└── requirements.txt    # Python dependencies