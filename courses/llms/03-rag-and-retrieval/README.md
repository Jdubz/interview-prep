# Module 03: RAG & Retrieval -- Core Interview Knowledge

This is the primary reference for RAG and retrieval concepts you need to know
for applied AI engineering interviews. It builds on the existing `concepts.md`
and `architecture.md` files in this directory, consolidating and extending them
into interview-ready depth.

---

## 1. Embeddings

### What Embeddings Are

An embedding is a learned mapping from discrete input (text, images, code) to a
dense, fixed-length vector in continuous space. The training objective forces
semantically similar inputs to cluster together.

```
embed("database connection timeout")  ->  [0.12, -0.34, 0.56, ..., 0.08]   (1536 dims)
embed("DB conn timed out")            ->  [0.11, -0.33, 0.55, ..., 0.09]   (1536 dims)
                                           ^^ very close in vector space

embed("best pizza in Brooklyn")       ->  [-0.72, 0.15, -0.03, ..., 0.44]  (1536 dims)
                                           ^^ far away
```

Key properties:
- **Fixed dimensionality**: Every input maps to the same-length vector (e.g., 1536-d)
- **Semantic proximity**: Similar meaning = nearby vectors
- **Compositionality**: Arithmetic on vectors sometimes captures relationships
  (king - man + woman ~ queen, though this is more reliable in older word2vec models)
- **Task-agnostic**: A single embedding can power search, classification, clustering, and dedup

### How Embedding Models Encode Meaning

Modern text embedding models are transformer-based. The typical architecture:

```
Input text
    |
    v
Tokenizer (text -> token IDs)
    |
    v
Transformer encoder (multiple layers of self-attention)
    |
    v
Pooling (mean of token embeddings, [CLS] token, or learned pooling)
    |
    v
Optional: projection + normalization
    |
    v
Fixed-length vector
```

Training uses contrastive learning: given (query, positive_doc, negative_doc) triples,
the model learns to push query and positive_doc embeddings together while pushing
query and negative_doc embeddings apart.

### Embedding Model Comparison

| Model | Provider | Dims | Max Tokens | Cost (per 1M tokens) | Notes |
|---|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | 8191 | ~$0.02 | Good default, supports dim reduction via `dimensions` param |
| `text-embedding-3-large` | OpenAI | 3072 | 8191 | ~$0.13 | Higher quality, also supports dim reduction |
| `text-embedding-ada-002` | OpenAI | 1536 | 8191 | ~$0.10 | Legacy, replaced by 3-small |
| `embed-v3` | Cohere | 1024 | 512 | ~$0.10 | Supports `input_type` (query vs doc), strong multilingual |
| `bge-large-en-v1.5` | BAAI (OSS) | 1024 | 512 | Free (self-host) | Top open-source model, competitive quality |
| `e5-mistral-7b-instruct` | Microsoft (OSS) | 4096 | 32768 | Free (self-host) | Instruction-tuned, high quality, large model |
| `gte-large-en-v1.5` | Alibaba (OSS) | 1024 | 8192 | Free (self-host) | Strong MTEB benchmark scores |
| `voyage-3` | Voyage AI | 1024 | 32000 | ~$0.06 | Optimized for code + retrieval |
| `nomic-embed-text-v1.5` | Nomic (OSS) | 768 | 8192 | Free (self-host) | Matryoshka dims, competitive |

**Interview insight**: Know that OpenAI's `text-embedding-3-*` models support Matryoshka
Representation Learning -- you can truncate the vector to fewer dimensions (e.g., 256)
with graceful quality degradation instead of a cliff. This matters for cost/latency
tradeoffs in production.

### Dimensionality

Higher dimensions capture more nuance but cost more to store and search:

```
Dimensions    Storage per 1M vectors (float32)    Search speed
   256              ~1 GB                          Fastest
   768              ~3 GB                          Fast
  1536              ~6 GB                          Moderate
  3072             ~12 GB                          Slower
```

**Matryoshka embeddings** let you train once at high dimensionality and truncate at
inference time. OpenAI's v3 models and several open-source models (nomic, gte) support
this natively.

### Normalization

Most modern embedding models output **L2-normalized vectors** (unit length). This means:
- Cosine similarity = dot product (since ||A|| = ||B|| = 1)
- You can use the faster dot product operation in your vector DB
- If your model does NOT normalize, you must choose cosine similarity or normalize yourself

**Interview question**: "Why does Pinecone recommend dot product metric with OpenAI embeddings?"
Answer: Because OpenAI embeddings are already L2-normalized, so dot product and cosine
similarity produce identical rankings, but dot product is cheaper to compute.

---

## 2. Similarity Metrics

### Cosine Similarity

```
cos(A, B) = (A . B) / (||A|| * ||B||)

Range: [-1, 1]
  1.0 = identical direction (same meaning)
  0.0 = orthogonal (unrelated)
 -1.0 = opposite direction (rare in practice for text)
```

- **Direction-only**: Ignores vector magnitude, compares orientation
- **Default choice** for text embeddings
- When vectors are normalized: cosine similarity == dot product

### Dot Product (Inner Product)

```
dot(A, B) = sum(A_i * B_i)

Range: (-inf, +inf) for unnormalized; [-1, 1] for normalized
```

- Faster to compute (no normalization step)
- For normalized vectors: identical to cosine similarity
- For unnormalized vectors: magnitude matters, which may or may not be desirable
- Use when your embedding model normalizes output (most do)

### Euclidean Distance (L2)

```
L2(A, B) = sqrt(sum((A_i - B_i)^2))

Range: [0, +inf)
  0 = identical
  larger = more different
```

- Sensitive to magnitude (a long vector and short vector pointing the same direction
  will have high L2 distance)
- Less common for text retrieval; more common in image/audio domains
- Relationship: for normalized vectors, minimizing L2 is equivalent to maximizing
  cosine similarity

### When to Use Which

| Metric | Use When | Avoid When |
|---|---|---|
| Cosine similarity | Default for text, especially if unsure about normalization | You need magnitude to matter |
| Dot product | Vectors are normalized (most production systems) | Vectors are NOT normalized |
| Euclidean (L2) | Image similarity, anomaly detection | Text search with varying doc lengths |

**Interview tip**: The answer to "which similarity metric?" is almost always "cosine
similarity, or dot product if embeddings are normalized" for text retrieval. If someone
asks about L2, know that it's monotonically related to cosine for normalized vectors
but is more common in image retrieval.

---

## 3. Vector Databases

### Why You Need One

You cannot do O(n) linear scan over millions of vectors for every query. Vector
databases provide:
- **Approximate Nearest Neighbor (ANN) indexing**: Sub-linear search
- **Metadata filtering**: Filter by attributes alongside vector search
- **Persistence**: Don't re-embed your entire corpus on restart
- **Scalability**: Shard, replicate, handle concurrent queries

### Comparison

| Database | Type | Language | ANN Algorithm | Metadata Filtering | Hybrid Search | Max Scale |
|---|---|---|---|---|---|---|
| **Pinecone** | Managed SaaS | - | Proprietary | Yes (rich) | No (use separately) | Billions |
| **Weaviate** | OSS / Cloud | Go | HNSW | Yes | Built-in BM25+vector | Billions |
| **Qdrant** | OSS / Cloud | Rust | HNSW | Yes (payload filters) | Built-in sparse+dense | Billions |
| **Milvus** | OSS / Cloud | Go/C++ | IVF, HNSW, DiskANN | Yes | Sparse vector support | Tens of billions |
| **ChromaDB** | OSS | Python | HNSW (hnswlib) | Yes (basic) | No | Millions |
| **pgvector** | PG extension | C | HNSW, IVFFlat | Yes (full SQL) | Via pg full-text search | Tens of millions |
| **FAISS** | Library | C++/Python | IVF, HNSW, PQ | No (DIY) | No | Billions (in-memory) |

### Architecture Differences

**Managed (Pinecone)**:
- Zero infrastructure management
- Pay per vector stored + queries
- Vendor lock-in, limited customization
- Best for: teams that want to focus on application logic, not infra

**Self-hosted open-source (Qdrant, Weaviate, Milvus)**:
- Full control over hardware, tuning, data residency
- Operational burden: backups, scaling, upgrades
- Often cheaper at scale
- Best for: teams with infra capability and scale/compliance requirements

**Postgres extension (pgvector)**:
- Uses your existing Postgres instance
- ACID transactions, joins with relational data
- Performance ceiling lower than purpose-built vector DBs
- Best for: moderate scale (<10M vectors), teams already on Postgres, need relational + vector

**In-process library (FAISS, ChromaDB)**:
- No server, runs in your application process
- No metadata filtering (FAISS) or limited filtering
- Best for: prototyping, batch processing, research

### Indexing and Performance

The ANN index is the core data structure that makes vector search fast:

```
Exact search:   O(n * d)     -- scan every vector, compute similarity
HNSW:           O(log n * d) -- navigate graph layers
IVF:            O(n/k * d)   -- search only nearby clusters (k = num clusters)
```

Index build time and memory are the tradeoffs:
- **HNSW** uses more memory (graph overhead) but has excellent query performance
- **IVF** is more memory-efficient but requires training (k-means clustering)
- **PQ** compresses vectors (lossy) for massive memory savings at cost of recall

### Metadata Filtering

Two approaches:

**Pre-filtering** (filter, then search):
```
1. Apply metadata filter (e.g., category = "engineering")
2. Search vectors only within filtered set
```
- Guarantees filter compliance
- Can be slow if filter is very selective (small result set, index less effective)

**Post-filtering** (search, then filter):
```
1. Search all vectors, get top-N candidates
2. Apply metadata filter to candidates
3. Return top-K from filtered results
```
- Can return fewer than K results if filter removes many candidates
- Generally faster search, but wasteful if most results are filtered out

Most production vector DBs now do **integrated filtering** -- the ANN search
and metadata filter happen together, combining both approaches.

### Choosing a Vector Database -- Decision Framework

```
Do you need a fully managed service?
  YES --> Do you need hybrid search built in?
            YES --> Weaviate Cloud / Qdrant Cloud
            NO  --> Pinecone
  NO  --> Are you already on Postgres?
            YES --> Is your scale < 10M vectors?
                      YES --> pgvector
                      NO  --> Purpose-built DB (Qdrant, Milvus)
            NO  --> Do you need GPU-accelerated search?
                      YES --> Milvus
                      NO  --> Are you prototyping?
                                YES --> ChromaDB
                                NO  --> Qdrant or Weaviate (self-hosted)
```

---

## 4. Chunking Strategies

Chunking is how you split documents into units for embedding and retrieval.
It is one of the highest-leverage decisions in RAG quality.

### Why Chunk?

1. **Embedding model token limits**: Most models cap at 512-8192 tokens
2. **Retrieval precision**: Smaller chunks = less noise per retrieval hit
3. **LLM context budget**: You need to fit multiple chunks + instructions in the prompt
4. **Embedding quality**: Embedding models produce better vectors for focused text

### Strategy Comparison

| Strategy | How It Works | Chunk Size Consistency | Semantic Coherence | Complexity |
|---|---|---|---|---|
| Fixed-size | Split every N tokens with overlap | Uniform | Poor | Trivial |
| Recursive character | Split on hierarchy of separators (\n\n, \n, ., space) | Variable (bounded) | Good | Low |
| Semantic | Embed sentences, group by similarity, split at drop-offs | Variable | Excellent | High |
| Document-aware | Split on structural markers (headers, code fences) | Variable | Good | Medium |
| Sentence-level | One sentence per chunk | Uniform (small) | High per chunk | Low |

### Fixed-Size Chunking

```python
def fixed_chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    # Split every `size` characters with `overlap` character overlap
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks
```

- Simple baseline. Use for homogeneous text without clear structure.
- **Overlap** (typically 10-20% of chunk size) prevents information loss at boundaries.
- Splits mid-sentence, mid-paragraph -- information gets fragmented.

### Recursive Character Chunking (LangChain default)

```
Separators (tried in order): ["\n\n", "\n", ". ", " ", ""]

1. Try to split on double newline (paragraph breaks)
2. If chunk still too large, split on single newline
3. If still too large, split on sentence boundaries
4. Last resort: split on spaces or characters
```

- Respects natural text boundaries at decreasing granularity
- The de facto standard for general-purpose RAG
- Variable chunk sizes, but bounded by max_chunk_size

### Semantic Chunking

```
1. Split document into sentences
2. Embed each sentence
3. Compute similarity between consecutive sentences
4. Split where similarity drops below threshold

Sentence similarities:
  S1-S2: 0.92  (same topic, keep together)
  S2-S3: 0.88  (same topic, keep together)
  S3-S4: 0.45  <-- topic shift, SPLIT HERE
  S4-S5: 0.91  (same topic, keep together)
```

- Produces semantically coherent chunks aligned with topic boundaries
- Requires embedding every sentence at ingestion time (expensive)
- Works well for documents with many topic shifts (meeting transcripts, long articles)

### Document-Aware Chunking

```
Markdown input:
  # Section A           <-- split boundary
  Content about A...
  ## Subsection A.1     <-- split boundary
  More content...
  ```python             <-- keep code blocks intact
  def foo():
      pass
  ```
  # Section B           <-- split boundary
```

- Preserves structural units: sections, code blocks, tables, lists
- Attaches metadata (section heading, nesting level) to each chunk
- Essential for technical documentation, codebases, structured content

### Chunk Size Tradeoffs

```
Small chunks (100-200 tokens)        Large chunks (500-1000 tokens)
  + High retrieval precision           + More context per chunk
  + Less noise per result              + Fewer retrieval calls needed
  - May lack context                   - More noise per result
  - Need more chunks to answer         - Lower retrieval precision
  - Higher storage/indexing cost        - Embedding quality may degrade

Sweet spot for most use cases: 300-500 tokens
```

**Interview question**: "How would you determine the right chunk size?"
Answer: Start with 500 tokens as baseline. Evaluate retrieval quality (recall@k,
precision@k) on a representative query set. Tune based on:
- If answers are incomplete: increase chunk size or retrieve more chunks
- If retrieved chunks contain irrelevant content: decrease chunk size
- If domain has short, self-contained units (FAQ, API docs): smaller chunks
- If domain has long, interconnected explanations (textbooks): larger chunks

### Overlap Strategies

- **Sliding window**: Fixed overlap (e.g., 50 tokens). Simple, predictable.
- **Sentence-boundary overlap**: Overlap at sentence boundaries to avoid mid-sentence splits.
- **No overlap**: Acceptable when using semantic or document-aware chunking (boundaries are natural).

---

## 5. RAG Pipeline Architecture

### End-to-End Flow

```
                         INGESTION (offline)
    ================================================================

    Raw Documents
         |
         v
    +-----------+     +----------+     +-----------+     +----------+
    |   Parse   | --> |  Chunk   | --> |   Embed   | --> |  Store   |
    |  (extract |     | (split   |     | (vectorize|     | (vector  |
    |   text)   |     |  text)   |     |  chunks)  |     |  DB +    |
    +-----------+     +----------+     +-----------+     | metadata)|
                                                         +----------+

                         QUERY (online)
    ================================================================

    User Query
         |
         v
    +------------------+
    | Query Transform  |  (optional: rewrite, expand, HyDE)
    +------------------+
         |
         v
    +------------------+
    | Embed Query      |  (same embedding model as ingestion)
    +------------------+
         |
         v
    +------------------+
    | Retrieve         |  vector search + optional keyword (hybrid)
    | (top-K chunks)   |  + optional metadata filters
    +------------------+
         |
         v
    +------------------+
    | Rerank           |  (optional: cross-encoder reranker)
    | (top-N chunks)   |  N << K
    +------------------+
         |
         v
    +------------------+
    | Prompt Assembly  |  system prompt + retrieved chunks + user query
    +------------------+
         |
         v
    +------------------+
    | LLM Generation   |  generate answer grounded in retrieved context
    +------------------+
         |
         v
    +------------------+
    | Post-processing  |  (optional: citation extraction, fact-check)
    +------------------+
         |
         v
    Response to User
```

### Critical Design Decisions

| Decision | Default | When to Change |
|---|---|---|
| Embedding model | `text-embedding-3-small` | Need multilingual, code-specific, or higher quality |
| Chunk size | 500 tokens | See chunk size tradeoffs above |
| Top-K retrieval | 5-10 | Low recall: increase. Noisy context: decrease. |
| Reranking | Off | Turn on when precision matters and latency allows |
| Hybrid search | Off | Turn on when exact keyword matches matter |
| Query transformation | Off | Turn on for conversational or complex queries |

### Prompt Assembly Pattern

```python
def build_rag_prompt(query: str, chunks: list[str], system: str = "") -> list[dict]:
    context = "\n---\n".join(chunks)
    return [
        {"role": "system", "content": system or (
            "Answer the user's question using ONLY the provided context. "
            "If the context doesn't contain the answer, say so. "
            "Cite the relevant section when possible."
        )},
        {"role": "user", "content": (
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )},
    ]
```

The system prompt is critical. Common instructions:
- "Only use information from the provided context" (reduces hallucination)
- "If the context doesn't contain the answer, say you don't know"
- "Cite which context section supports your answer"
- "If sources conflict, note the discrepancy"

---

## 6. Hybrid Search

### Why Pure Semantic Search Isn't Enough

Semantic (vector) search excels at meaning but fails at:
- **Exact matches**: Product IDs, error codes, version numbers, proper nouns
- **Rare terms**: Highly specific jargon the embedding model may not distinguish
- **Acronyms**: "HNSW" might not embed distinctly from other abbreviations
- **Negation/absence**: "NOT python" is hard to capture in embedding space

Keyword (BM25) search excels at exact token matching but fails at:
- **Synonyms**: "automobile" won't match "car"
- **Paraphrasing**: Different phrasings of the same question
- **Conceptual similarity**: Related but differently worded content

**Hybrid search combines both** for better recall across all query types.

### Architecture

```
User Query
     |
     +---> Embed query ---------> Vector search (top-20) ----+
     |                                                        |
     +---> Tokenize query ------> BM25 search (top-20) ------+
                                                              |
                                                              v
                                                    Reciprocal Rank Fusion
                                                         (top-20)
                                                              |
                                                              v
                                                    Cross-Encoder Rerank
                                                         (top-5)
                                                              |
                                                              v
                                                       Prompt Assembly
```

### Reciprocal Rank Fusion (RRF)

```
RRF_score(doc) = SUM over all rankers r:  1 / (k + rank_r(doc))

where k = 60 (standard constant, rarely needs tuning)
```

Example:
```
Doc X: rank 1 in vector search, rank 5 in BM25
  RRF = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

Doc Y: rank 3 in vector search, rank 2 in BM25
  RRF = 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200

Doc Y wins (higher combined score) -- boosted by ranking well in BOTH methods.
```

Why RRF works:
- No need to normalize scores across different search methods
- Documents that appear in multiple result sets get boosted
- Robust -- the k=60 constant works well empirically without tuning
- Simple to implement and reason about

### Alternative Fusion Methods

| Method | Description | When to Use |
|---|---|---|
| RRF | Rank-based, no score normalization | Default, works well without tuning |
| Weighted linear | `alpha * vector_score + (1-alpha) * bm25_score` | When you want control over weight |
| Convex combination | Normalize scores then combine | When scores are calibrated |
| Distribution-based | Normalize to same distribution | Research settings |

---

## 7. Reranking

### Why Rerank?

The initial retrieval stage uses **bi-encoder** models: query and document are embedded
independently. This is fast (O(1) per stored document via ANN index) but less accurate
because the query and document never "see" each other.

**Cross-encoder rerankers** process the (query, document) pair jointly through a
transformer, computing a relevance score with full attention between query and document
tokens. This is much more accurate but too slow for first-stage retrieval.

```
Bi-encoder (retrieval):          Cross-encoder (reranking):

  query  -->  [Encoder] --> vec     query + doc --> [Encoder] --> relevance score
  doc    -->  [Encoder] --> vec
  similarity = dot(q, d)           Processes the pair jointly -- much richer signal
```

### The Two-Stage Pipeline

```
Stage 1 (Retrieval): Bi-encoder retrieves top-50 candidates
  - Fast: O(log n) via ANN index
  - Lower precision acceptable (cast a wide net)

Stage 2 (Reranking): Cross-encoder scores 50 candidates
  - Slow: O(50) forward passes, but 50 is manageable
  - High precision (full query-document attention)
  - Return top-5 to LLM
```

This is analogous to database query optimization: fast index scan for candidates,
expensive filter on small set.

### Reranker Models

| Model | Provider | Type | Notes |
|---|---|---|---|
| Cohere Rerank v3 | Cohere (API) | Cross-encoder | Production-grade, multilingual |
| `bge-reranker-v2-m3` | BAAI (OSS) | Cross-encoder | Strong open-source, multilingual |
| `ms-marco-MiniLM` | Microsoft (OSS) | Cross-encoder | Small, fast, English-focused |
| Jina Reranker v2 | Jina (API/OSS) | Cross-encoder | Good balance of speed/quality |
| ColBERT v2 | Stanford (OSS) | Late interaction | Token-level matching, different tradeoff |

### Reranking Latency Budget

In production, you have a latency budget. Example breakdown:

```
Total target: 2000ms

Query embedding:      50ms
Vector search:       100ms
BM25 search:          50ms
RRF fusion:            5ms
Reranking (50 docs): 200ms   <-- cross-encoder
Prompt assembly:      10ms
LLM generation:    1200ms
Post-processing:      50ms
Overhead/network:    335ms
```

If latency is tight, reduce candidates to rerank (50 -> 20) or use a smaller
reranker model. The quality gain from reranking typically justifies the cost.

---

## 8. Metadata Filtering

### Strategy

Store structured metadata alongside every chunk:

```python
{
    "id": "doc-123-chunk-5",
    "vector": [0.12, -0.34, ...],
    "text": "The API rate limit is 100 requests per minute...",
    "metadata": {
        "source": "docs/api-reference.md",
        "title": "Rate Limiting",
        "section_path": "API Reference > Rate Limiting > Defaults",
        "doc_type": "api_docs",
        "language": "en",
        "last_updated": "2025-01-15",
        "version": "3.2",
        "tenant_id": "acme-corp",        # multi-tenant
        "access_level": "public",         # access control
        "chunk_index": 5,
        "parent_chunk_id": "doc-123-section-2",
    }
}
```

### Common Filtering Patterns

| Pattern | Filter Expression | Use Case |
|---|---|---|
| Recency | `last_updated > "2025-01-01"` | Prefer fresh content |
| Tenant isolation | `tenant_id == "acme-corp"` | Multi-tenant SaaS |
| Document type | `doc_type IN ["api_docs", "guides"]` | Scope retrieval |
| Access control | `access_level IN user.permissions` | Security |
| Language | `language == "en"` | Multilingual corpus |
| Source scoping | `source LIKE "docs/v3/*"` | Version-specific retrieval |

### Pre-filter vs Post-filter vs Integrated

```
Pre-filter:   Filter metadata first, then vector search within filtered set
              + Guarantees all results match filter
              - Smaller search space may reduce ANN effectiveness
              - Can be slow if filter is very selective

Post-filter:  Vector search first (top-N), then filter results
              + Fast vector search on full index
              - May return fewer than K results
              - Wasteful if most candidates are filtered out

Integrated:   Filter and search happen together in the index
              + Best of both worlds
              - Requires vector DB support (Pinecone, Qdrant, Weaviate do this)
```

**Interview answer**: "Most production vector databases now support integrated
filtering, where metadata constraints are applied during the ANN search. This
avoids the problems of both pre- and post-filtering. If I had to choose between
the two simpler approaches, I'd use pre-filtering for highly selective filters
and post-filtering for broad filters."

---

## 9. RAG vs Fine-Tuning vs Long Context

### Decision Framework

```
+------------------------------------------------------------------+
| Question                          | RAG | Fine-tune | Long ctx   |
|-----------------------------------|-----|-----------|------------|
| Data changes frequently?          | YES |    no     |   maybe    |
| Need to cite sources?             | YES |    no     |   maybe    |
| Domain-specific behavior/style?   | no  |    YES    |    no      |
| Corpus fits in context window?    | n/a |    n/a    |    YES     |
| Need factual grounding?           | YES |    no     |   maybe    |
| Cost per query matters?           | med |    low    |   HIGH     |
| Setup complexity?                 | med |    high   |    low     |
| Latency sensitive?                | med |    low    |   high     |
+------------------------------------------------------------------+
```

### When RAG Wins

- **Dynamic knowledge**: Documents change frequently (support docs, product catalogs)
- **Large corpus**: Too much data to fit in any context window
- **Traceability**: Need to cite sources, show provenance
- **Multi-tenant**: Different users access different document sets
- **Cost at scale**: Only retrieve what's needed per query

### When Fine-Tuning Wins

- **Behavioral changes**: Specific tone, style, output format
- **Consistent patterns**: Always respond in a certain way
- **Domain adaptation**: Medical, legal, financial language understanding
- **Latency**: No retrieval overhead at query time
- **Small, stable knowledge**: Facts that rarely change

### When Long Context Wins

- **Small corpus**: Everything fits in the context window (< 100K tokens)
- **Simplicity**: No chunking, embedding, vector DB infrastructure
- **Prototyping**: Quick to test, no pipeline to build
- **Cross-document reasoning**: All information available simultaneously

### Combining Approaches

In practice, these are not mutually exclusive:

```
Fine-tuned model (domain behavior)
  + RAG (dynamic knowledge retrieval)
  + Long context (fit more retrieved chunks)
  = Production-grade system
```

Example: A legal AI assistant might use a fine-tuned model (for legal writing style)
with RAG (to retrieve relevant case law) and a long context window (to fit multiple
lengthy case excerpts).

---

## 10. Interview Questions You Should Be Ready For

### Conceptual Questions

1. **"Walk me through a RAG pipeline end to end."**
   Cover: ingestion (parse, chunk, embed, store) and query (embed, retrieve, rerank,
   prompt, generate). Mention metadata, hybrid search, evaluation.

2. **"How would you evaluate RAG quality?"**
   Two levels: retrieval quality (recall@k, precision@k, MRR) and generation quality
   (faithfulness, answer relevance). Tools: RAGAS, LLM-as-judge, human eval.

3. **"When would you use RAG vs fine-tuning?"**
   Use the decision framework above. Key differentiator: RAG for dynamic/factual
   knowledge, fine-tuning for behavioral/stylistic changes.

4. **"How do you handle a multi-tenant RAG system?"**
   Metadata filtering with tenant_id on every chunk. Pre-filter or integrated filter
   to ensure tenant isolation. Consider separate indexes for very large tenants.

5. **"What's the biggest failure mode you've seen in RAG?"**
   Irrelevant retrieval cascading into confident-sounding but wrong answers. Root
   causes: bad chunking, wrong embedding model, no reranking, weak system prompt.

### Design Questions

6. **"Design a RAG system for customer support over 10K product documents."**
   Key decisions: chunking (document-aware, respect product boundaries), metadata
   (product_id, version, category), hybrid search (error codes need keyword matching),
   reranking, incremental updates as docs change.

7. **"How would you reduce hallucination in a RAG system?"**
   Multi-layered: improve retrieval (more relevant chunks), reranking (less noise),
   strong system prompt ("only use provided context"), post-generation fact-checking,
   citation extraction, faithfulness evaluation.

8. **"Your RAG system retrieves the right chunks but the LLM still gives wrong answers."**
   Debug: Are chunks positioned well in the prompt? (Lost-in-the-middle problem.)
   Is the system prompt strong enough? Is the LLM ignoring context in favor of
   parametric knowledge? Try: reorder chunks, strengthen instructions, use a more
   instruction-following model.

---

## Further Reading

- `deep-dive.md` in this directory -- Advanced algorithms, agentic RAG, evaluation
- `cheat-sheet.md` -- Quick reference tables for interviews
- `examples.py` -- Runnable code patterns
- `exercises.py` -- Practice problems
- `concepts.md` -- Original concept notes (foundation for this module)
- `architecture.md` -- Original architecture notes (foundation for this module)
