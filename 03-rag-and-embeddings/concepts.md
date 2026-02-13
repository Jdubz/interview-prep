# RAG & Embeddings — Core Concepts

## Embeddings

### What Are Embeddings?

Embeddings are dense vector representations of text in a continuous high-dimensional space. The key property: **semantically similar text maps to nearby vectors**.

```
embed("How do I reset my password?")  ≈  embed("I forgot my login credentials")
  → cosine similarity: ~0.89 (high)

embed("How do I reset my password?")  ≈  embed("What's the weather today?")
  → cosine similarity: ~0.12 (low)
```

### Embedding Models vs. LLMs

| | Embedding Models | LLMs |
|---|---|---|
| **Output** | Fixed-length vector (e.g., 1536-dim) | Variable-length text |
| **Purpose** | Represent meaning as numbers | Generate text |
| **Cost** | Very cheap (~$0.02/1M tokens) | Much more expensive |
| **Speed** | Fast, can batch thousands | Slower, sequential generation |
| **Examples** | OpenAI `text-embedding-3-small`, Cohere `embed-v3`, open-source `bge`, `e5` | GPT-4o, Claude, Llama |

### Popular Embedding Models

| Model | Dimensions | Notes |
|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 | Good balance of quality/cost |
| OpenAI `text-embedding-3-large` | 3072 | Higher quality, supports dimension reduction |
| Cohere `embed-v3` | 1024 | Strong multilingual, supports different input types |
| `bge-large-en-v1.5` | 1024 | Open source, competitive quality |
| `e5-mistral-7b-instruct` | 4096 | Instruction-tuned, high quality |

### Similarity Metrics

**Cosine similarity** (most common):
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: -1 to 1 (1 = identical direction, 0 = orthogonal, -1 = opposite)
- Ignores magnitude, focuses on direction
- Standard choice for text embeddings

**Dot product:** Faster to compute. Equivalent to cosine when vectors are normalized (most embedding models normalize output).

**Euclidean distance:** Less common for text. Sensitive to magnitude.

---

## Vector Similarity Search

### How It Works

1. **Index time:** Embed all your documents, store vectors in a vector database
2. **Query time:** Embed the user's query, find the k nearest vectors

### Approximate Nearest Neighbor (ANN)

Exact nearest neighbor search is O(n) — too slow for large datasets. ANN algorithms trade perfect accuracy for speed:

| Algorithm | How It Works | Used By |
|---|---|---|
| **HNSW** (Hierarchical Navigable Small Worlds) | Graph-based, multilayer navigation | Pinecone, Weaviate, pgvector |
| **IVF** (Inverted File Index) | Cluster vectors, search only nearby clusters | FAISS, Milvus |
| **Product Quantization** | Compress vectors for faster comparison | Combined with IVF |

**HNSW** is the most popular — great balance of speed, accuracy, and simplicity. Typical recall@10 > 95% with proper tuning.

---

## Chunking Strategies

How you split documents into chunks is one of the most impactful RAG decisions.

### Why Chunk?

- Embedding models have token limits (typically 512–8192 tokens)
- Smaller chunks = more precise retrieval (less noise)
- Larger chunks = more context per retrieval (fewer chunks needed)

### Strategies

#### Fixed-Size Chunking
```
Split every N tokens (e.g., 500) with M token overlap (e.g., 50)
```
- **Pros:** Simple, predictable chunk size
- **Cons:** Splits mid-sentence, mid-paragraph, mid-thought
- **When to use:** Quick baseline, uniform-structure documents

#### Recursive / Structural Chunking
```
Try to split on: paragraphs → sentences → words → characters
Respect a max size, split at the highest-level boundary that fits
```
- **Pros:** Respects natural text structure
- **Cons:** Variable chunk sizes
- **When to use:** Default recommendation for most use cases

#### Semantic Chunking
```
Compute embeddings for sentences, group consecutive sentences
with high similarity, split where similarity drops
```
- **Pros:** Chunks align with topic boundaries
- **Cons:** More complex, requires embedding each sentence at ingestion
- **When to use:** When retrieval quality matters more than simplicity

#### Document-Aware Chunking
```
Split on structural markers: headings, code blocks, tables, etc.
Preserve metadata (section title, page number, etc.)
```
- **Pros:** Maintains document structure, great for technical docs
- **Cons:** Requires parsing document structure
- **When to use:** Structured documents (docs sites, PDFs with headings)

### Chunk Size Guidelines

| Chunk Size | Retrieval Precision | Context Richness | Best For |
|---|---|---|---|
| 100–200 tokens | High | Low | FAQ matching, short answers |
| 300–500 tokens | Balanced | Balanced | General-purpose RAG |
| 500–1000 tokens | Lower | High | Complex topics, code |
| Full document | Lowest | Highest | Small corpus, long-context models |

**Overlap:** 10–20% overlap between adjacent chunks prevents information loss at boundaries.

---

## Retrieval Pipelines

### Basic Retrieval

```
Query → Embed → Vector Search → Top-K Chunks → LLM Prompt
```

Simple and effective for many use cases.

### Hybrid Search

Combine vector (semantic) and keyword (lexical) search for better recall:

```
Query → ┬→ Vector Search → Semantic Results ─┐
        └→ Keyword Search (BM25) → Lexical Results ─┤→ Merge + Rank → Top-K
```

**Why hybrid?**
- Vector search excels at meaning ("happy" finds "joyful")
- Keyword search excels at exact matches (product IDs, error codes, names)
- Together they cover both cases

**Reciprocal Rank Fusion (RRF):** Simple, effective algorithm for merging ranked lists from different search methods.

### Reranking

After initial retrieval, use a more powerful (but slower) model to re-score results:

```
Initial Top-20 → Cross-Encoder Reranker → Reranked Top-5 → LLM Prompt
```

**Cross-encoders** (e.g., Cohere Rerank, `bge-reranker`) process (query, document) pairs jointly, producing more accurate relevance scores than embedding similarity alone. They're too slow for first-stage search but great for refining a small candidate set.

### Query Transformation

Sometimes the user's query isn't ideal for retrieval:

- **Query expansion:** Generate multiple query variations, retrieve for each, merge results
- **HyDE (Hypothetical Document Embeddings):** Ask the LLM to generate a hypothetical answer, embed *that* for retrieval (the hypothetical answer is often closer to the actual document than the question is)
- **Step-back prompting:** Rephrase the query at a higher level of abstraction

---

## Key Decisions in RAG Design

| Decision | Options | Guidance |
|---|---|---|
| Chunk size | 200–1000 tokens | Start at 500, tune based on eval |
| Chunks retrieved (k) | 3–20 | Start at 5, increase if recall is low |
| Search type | Vector / Keyword / Hybrid | Start vector, add keyword if exact match matters |
| Reranking | None / Cross-encoder | Add if precision matters and latency allows |
| Embedding model | See table above | Match your language, cost, and quality needs |
| Vector DB | See architecture.md | Depends on scale and infrastructure |
