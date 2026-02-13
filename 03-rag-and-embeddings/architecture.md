# RAG Architecture

## End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  Documents → Parse → Chunk → Embed → Store in Vector DB        │
│                                        + metadata               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
│                                                                 │
│  User Query                                                     │
│      ↓                                                          │
│  (Optional) Query Transformation                                │
│      ↓                                                          │
│  Embed Query                                                    │
│      ↓                                                          │
│  Retrieve (Vector + optional Keyword/Hybrid)                    │
│      ↓                                                          │
│  (Optional) Rerank                                              │
│      ↓                                                          │
│  Build Prompt (system + context chunks + query)                 │
│      ↓                                                          │
│  LLM Generation                                                 │
│      ↓                                                          │
│  (Optional) Citation Extraction / Validation                    │
│      ↓                                                          │
│  Response to User                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Vector Databases

### Comparison

| Database | Type | Key Strengths | Best For |
|---|---|---|---|
| **Pinecone** | Managed cloud | Zero ops, scales automatically, metadata filtering | Teams wanting minimal infra |
| **Weaviate** | Open source / cloud | Hybrid search built-in, GraphQL API, modules | Hybrid search use cases |
| **Qdrant** | Open source / cloud | Rust-based (fast), rich filtering, payload storage | Performance-critical apps |
| **Milvus** | Open source / cloud | GPU support, massive scale | Very large datasets |
| **ChromaDB** | Open source | Simple API, great DX, embeds in-process | Prototyping, small-scale |
| **pgvector** | PostgreSQL extension | Uses existing Postgres, ACID transactions | Teams already on Postgres |
| **FAISS** | Library (Meta) | Extremely fast, GPU support | Research, custom pipelines |

### Key Features to Evaluate

- **Metadata filtering:** Filter by document type, date, user, etc. alongside vector search
- **Hybrid search:** Built-in keyword + vector search fusion
- **Scalability:** Max vectors, query latency at scale
- **Operational model:** Managed vs. self-hosted
- **Cost:** Storage + query pricing

### Choosing a Vector DB

```
Need managed + zero ops?        → Pinecone
Already using Postgres?         → pgvector
Need hybrid search built-in?    → Weaviate
Need maximum performance?       → Qdrant or Milvus
Prototyping / small scale?      → ChromaDB
Custom research pipeline?       → FAISS
```

---

## Ingestion Pipeline Design

### Document Processing

```
Raw Documents
    ↓
┌─────────────┐
│   Parsing    │  PDF → text, HTML → text, Docx → text
└─────────────┘
    ↓
┌─────────────┐
│  Cleaning    │  Remove headers/footers, fix encoding, normalize whitespace
└─────────────┘
    ↓
┌─────────────┐
│  Chunking    │  Split into appropriately-sized pieces (see concepts.md)
└─────────────┘
    ↓
┌─────────────┐
│  Enrichment  │  Add metadata: source, title, section, date, author
└─────────────┘
    ↓
┌─────────────┐
│  Embedding   │  Generate vectors for each chunk
└─────────────┘
    ↓
┌─────────────┐
│   Storage    │  Insert vectors + metadata + original text into vector DB
└─────────────┘
```

### Metadata Strategy

Store metadata alongside vectors to enable filtering at query time:

```python
{
    "vector": [0.1, 0.02, ...],         # embedding
    "text": "chunk content...",          # original text for prompt injection
    "metadata": {
        "source": "docs/api-guide.md",  # document source
        "title": "API Authentication",  # section/document title
        "chunk_index": 3,               # position in document
        "total_chunks": 12,             # total chunks in document
        "last_updated": "2024-01-15",   # for freshness filtering
        "category": "api",              # for category filtering
    }
}
```

### Ingestion Considerations

- **Incremental updates:** Don't re-embed everything when one document changes. Track document hashes, update only changed chunks.
- **Embedding batch size:** Most APIs support batching (e.g., 100 texts per call). Batch for throughput.
- **Deduplication:** Detect and remove duplicate/near-duplicate chunks.
- **Versioning:** Track which embedding model version was used. If you change models, you need to re-embed everything (vectors aren't compatible across models).

---

## Hybrid Search Architecture

```
User Query
    ↓
┌───────────────────────────┐
│    Query Processing        │
│    - Embed for vector      │
│    - Tokenize for BM25     │
└───────────────────────────┘
    ↓                    ↓
┌──────────┐      ┌──────────┐
│  Vector   │      │  BM25    │
│  Search   │      │  Search  │
│  (top 20) │      │  (top 20)│
└──────────┘      └──────────┘
    ↓                    ↓
┌───────────────────────────┐
│   Reciprocal Rank Fusion   │
│   or weighted scoring      │
│   → merged top 20          │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│   Cross-Encoder Reranker   │
│   → final top 5            │
└───────────────────────────┘
    ↓
  Prompt Assembly → LLM
```

### Reciprocal Rank Fusion (RRF)

Simple algorithm for merging ranked lists:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i

where k = 60 (constant), rank_i = rank from search method i
```

Documents that rank highly in multiple methods get boosted. No tuning needed — the constant k works well in practice.

---

## Advanced Patterns

### Multi-Index RAG

Different content types may need different embeddings or retrieval strategies:

```
Query → ┬→ Code Index (code-optimized embeddings) → code chunks
        ├→ Docs Index (general embeddings) → doc chunks
        └→ API Index (structured search) → API specs

        All results → Merge → Rerank → LLM
```

### Contextual Retrieval

Add context to chunks before embedding:

```
Original chunk: "The function returns null if the user is not found."

Enriched chunk: "In the UserService authentication module, when looking up
users by email: The function returns null if the user is not found."
```

Adding document/section context to each chunk before embedding improves retrieval accuracy because the chunk becomes self-contained.

### Parent-Child Chunking

Embed small chunks for precise retrieval, but return the parent (larger) chunk for context:

```
Document
├── Section (parent chunk — returned to LLM)
│   ├── Paragraph 1 (child chunk — used for search)
│   ├── Paragraph 2 (child chunk — used for search)
│   └── Paragraph 3 (child chunk — used for search)
```

This gives you precision in retrieval and context richness in generation.

### Conversational RAG

Maintain conversation context across turns:

```
Turn 1: "What's the refund policy?" → retrieve refund docs → answer
Turn 2: "What about digital products?"
  → Rewrite query using history: "What is the refund policy for digital products?"
  → Retrieve with rewritten query
```

The key: use the LLM to rewrite the follow-up query into a standalone query that includes context from the conversation.

---

## Evaluation Framework

### Retrieval Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Recall@k** | % of relevant docs in top-k results | > 0.8 |
| **Precision@k** | % of top-k results that are relevant | > 0.6 |
| **MRR** | How high the first relevant result ranks | > 0.7 |
| **NDCG@k** | Quality of ranking order | > 0.7 |

### Generation Metrics

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Does the answer only use information from the retrieved context? |
| **Answer relevance** | Does the answer address the question? |
| **Context relevance** | Are the retrieved chunks relevant to the question? |

### Tools

- **RAGAS:** Open-source framework for RAG evaluation (faithfulness, answer relevance, context metrics)
- **LLM-as-judge:** Use a powerful LLM to evaluate another LLM's outputs
- **Human evaluation:** Gold standard but expensive; use for calibration

---

## Common Failure Modes

| Problem | Symptom | Solution |
|---|---|---|
| Wrong chunks retrieved | Answer is off-topic | Improve chunking, add hybrid search, reranking |
| Relevant info not in top-k | "I don't have information about that" | Increase k, try query expansion, check embedding model |
| Model ignores context | Answer contradicts retrieved docs | Strengthen system prompt, reduce noise in context |
| Model hallucinates beyond context | Answer includes ungrounded facts | Ask model to cite sources, add "only use provided context" instruction |
| Stale information | Answer is outdated | Implement incremental ingestion, add date filtering |
| Chunks too fragmented | Answer lacks coherence | Increase chunk size, use parent-child chunking |
