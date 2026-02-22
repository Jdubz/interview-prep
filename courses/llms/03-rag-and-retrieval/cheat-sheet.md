# Module 03: RAG & Retrieval -- Cheat Sheet

Quick reference tables for interviews and system design. Bookmark this.

---

## Vector Database Comparison

| Database | Type | ANN Algo | Metadata Filter | Hybrid Search | Max Dims | Pricing Model | Self-Host |
|---|---|---|---|---|---|---|---|
| **Pinecone** | Managed | Proprietary | Rich (pre+post) | No (DIY) | 20000 | Per vector + query | No |
| **Weaviate** | OSS/Cloud | HNSW | Yes | BM25+vector built-in | 65535 | Cloud: usage-based | Yes |
| **Qdrant** | OSS/Cloud | HNSW | Payload filters | Sparse+dense | 65535 | Cloud: usage-based | Yes |
| **Milvus** | OSS/Cloud | IVF/HNSW/DiskANN | Yes | Sparse vectors | 32768 | Zilliz Cloud: CU-based | Yes |
| **ChromaDB** | OSS | HNSW (hnswlib) | Basic (where) | No | No hard limit | Free (self-host only) | Yes |
| **pgvector** | PG Extension | HNSW/IVFFlat | Full SQL | Via pg_trgm/tsvector | 2000 | Postgres hosting | Yes |
| **FAISS** | Library | IVF/HNSW/PQ | No (DIY) | No | No limit | Free | N/A (library) |

**Quick decision**:
- Zero ops, just works: **Pinecone**
- Already on Postgres, < 10M vectors: **pgvector**
- Need built-in hybrid search: **Weaviate** or **Qdrant**
- Massive scale with GPU: **Milvus**
- Prototyping: **ChromaDB**
- Research / custom pipeline: **FAISS**

---

## Embedding Model Comparison

| Model | Provider | Dims | Max Tokens | Matryoshka | Cost/1M tok | MTEB Avg | Notes |
|---|---|---|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | 8191 | Yes | $0.02 | ~62 | Default choice |
| `text-embedding-3-large` | OpenAI | 3072 | 8191 | Yes | $0.13 | ~64 | Higher quality |
| `embed-v3` | Cohere | 1024 | 512 | No | $0.10 | ~65 | Input-type aware |
| `voyage-3` | Voyage AI | 1024 | 32000 | No | $0.06 | ~67 | Code-optimized variant |
| `bge-large-en-v1.5` | BAAI (OSS) | 1024 | 512 | No | Free | ~64 | Top OSS English |
| `bge-m3` | BAAI (OSS) | 1024 | 8192 | No | Free | ~65 | Multilingual, dense+sparse |
| `e5-mistral-7b-instruct` | Microsoft (OSS) | 4096 | 32768 | No | Free | ~66 | Large, high quality |
| `gte-large-en-v1.5` | Alibaba (OSS) | 1024 | 8192 | No | Free | ~65 | Strong MTEB |
| `nomic-embed-text-v1.5` | Nomic (OSS) | 768 | 8192 | Yes | Free | ~62 | Lightweight |

**Quick decision**:
- API, budget-friendly: `text-embedding-3-small`
- API, best quality: `voyage-3` or `text-embedding-3-large`
- Self-hosted, English: `bge-large-en-v1.5` or `gte-large-en-v1.5`
- Self-hosted, multilingual: `bge-m3`
- Long documents: `e5-mistral-7b-instruct` or `nomic-embed-text-v1.5`

---

## Chunking Decision Matrix

| Document Type | Recommended Strategy | Chunk Size | Overlap | Notes |
|---|---|---|---|---|
| Plain text (articles, blogs) | Recursive character | 400-600 tokens | 50-100 tokens | Default choice |
| Technical docs (MDX, RST) | Document-aware (headers) | 300-500 tokens | 0 (natural boundaries) | Preserve section structure |
| Code files | Document-aware (functions/classes) | Whole function/class | 0 | Keep logical units intact |
| PDFs (structured) | Document-aware + OCR | 400-600 tokens | 50 tokens | Parse structure first |
| PDFs (scanned/unstructured) | Fixed-size after OCR | 300-500 tokens | 50-100 tokens | OCR quality is bottleneck |
| FAQ / Q&A pairs | Per question-answer pair | 1 pair per chunk | 0 | Natural atomic units |
| Chat transcripts | Semantic (topic shifts) | Variable (1-5 turns) | 0 | Group by conversation topic |
| Legal contracts | Document-aware (clauses) | Per clause/section | 0 | Clause boundaries matter |
| API documentation | Document-aware (endpoints) | Per endpoint | 0 | Keep request/response together |
| Meeting notes | Semantic | Variable | 0 | Topic boundaries matter |
| Spreadsheet/CSV | Per row or row group | 1-10 rows | 0 | Preserve row context |

---

## Similarity Metric Quick Reference

| Metric | Formula | Range | Best For | Notes |
|---|---|---|---|---|
| Cosine similarity | `(A.B) / (\|\|A\|\| * \|\|B\|\|)` | [-1, 1] | Text (default) | Direction only, ignores magnitude |
| Dot product | `SUM(A_i * B_i)` | (-inf, inf) | Normalized text embeddings | Faster; = cosine when normalized |
| Euclidean (L2) | `sqrt(SUM((A_i-B_i)^2))` | [0, inf) | Images, anomaly detection | Magnitude-sensitive |
| Manhattan (L1) | `SUM(\|A_i - B_i\|)` | [0, inf) | Sparse data | Less sensitive to outlier dimensions |

**Rule of thumb**: Use cosine similarity unless you know your vectors are normalized
(then use dot product for speed) or you have a specific reason to use L2.

---

## RAG Evaluation Metrics

### Retrieval Metrics

| Metric | Formula | What It Measures | Target |
|---|---|---|---|
| Recall@K | `\|relevant in top-K\| / \|total relevant\|` | Did we find the relevant docs? | > 0.8 |
| Precision@K | `\|relevant in top-K\| / K` | Are top-K results mostly relevant? | > 0.6 |
| MRR | `1 / rank_of_first_relevant` | How high is the first relevant result? | > 0.7 |
| NDCG@K | `DCG@K / ideal_DCG@K` | Is the ranking order good? | > 0.7 |
| Hit Rate@K | `1 if any relevant in top-K else 0` | Did we find anything relevant? | > 0.9 |

### Generation Metrics (RAGAS)

| Metric | What It Measures | How Computed | Target |
|---|---|---|---|
| Faithfulness | Answer grounded in context? | Claims in answer supported by context / total claims | > 0.9 |
| Answer Relevance | Answer addresses the question? | Similarity of generated-from-answer questions to original | > 0.8 |
| Context Precision | Retrieved chunks relevant? | Weighted precision at each rank position | > 0.7 |
| Context Recall | Context covers the answer? | Ground-truth claims found in context / total GT claims | > 0.8 |

### Quick Diagnostic

```
Low recall@K        -> Retrieval missing documents -> fix chunking, embedding, or add hybrid search
Low precision@K     -> Too much noise in results  -> add reranking, reduce chunk size
Low faithfulness    -> LLM hallucinating           -> strengthen prompt, add citation requirement
Low answer relevance -> Answer off-topic           -> check retrieval + prompt structure
Low context recall  -> Context incomplete          -> increase K, improve chunking, check corpus
```

---

## ANN Algorithm Quick Reference

| Algorithm | Type | Memory | Build Time | Query Time | Recall | Best For |
|---|---|---|---|---|---|---|
| Flat (exact) | Brute force | Low | None | O(n*d) | 100% | < 100K vectors |
| HNSW | Graph | High (+30-50%) | Slow | O(log n) | 95-99% | General purpose |
| IVF | Cluster | Medium | Medium (k-means) | O(n/k * nprobe) | 90-98% | Large scale |
| PQ | Compression | Very low | Medium | Fast (lookup tables) | 80-95% | Memory constrained |
| IVF-PQ | Cluster+compress | Low | Slow | Fast | 85-95% | Billions of vectors |
| DiskANN | Disk-based graph | Low (RAM) | Slow | Medium (disk I/O) | 95-99% | Larger than RAM |

---

## Reranker Comparison

| Model | Provider | Type | Latency (50 docs) | Quality | Notes |
|---|---|---|---|---|---|
| Rerank v3 | Cohere (API) | Cross-encoder | ~200ms | High | Production-grade, multilingual |
| `bge-reranker-v2-m3` | BAAI (OSS) | Cross-encoder | ~150ms (GPU) | High | Best open-source |
| `ms-marco-MiniLM` | Microsoft (OSS) | Cross-encoder | ~50ms (GPU) | Medium | Small and fast |
| Jina Reranker v2 | Jina (API/OSS) | Cross-encoder | ~100ms | Medium-High | Good balance |
| ColBERT v2 | Stanford (OSS) | Late interaction | ~30ms | Medium-High | Different architecture |
| FlashRank | OSS | Cross-encoder | ~80ms (CPU) | Medium | CPU-friendly |

---

## Common Failure Modes and Fixes

| Failure Mode | Symptoms | Quick Fix | Proper Fix |
|---|---|---|---|
| Empty retrieval | "No information found" | Increase K, lower threshold | Hybrid search, query expansion |
| Irrelevant chunks | Off-topic answers | Add reranking | Improve chunking, try different embedding model |
| Lost in the middle | Ignores relevant mid-context | Put best chunks first/last | Reduce chunk count, summarize |
| Conflicting sources | Contradictory answers | Add recency filter | Version metadata, source authority ranking |
| Hallucination despite context | Makes up facts beyond context | Lower temperature | Stronger prompt, citation requirement, faithfulness eval |
| Chunking artifacts | Partial/broken information | Increase overlap | Document-aware chunking, parent-child |
| Query-doc mismatch | Right docs exist but not retrieved | Try HyDE or query expansion | Contextual chunk headers, hybrid search |
| Stale content | Outdated answers | Date filter in metadata | Incremental indexing pipeline |
| Context overflow | Too many chunks for context window | Reduce K | Reranking + context window packing |

---

## RAG vs Fine-Tuning vs Long Context -- Quick Decision

```
Need to ADD knowledge?           -> RAG
Need to CHANGE behavior/style?   -> Fine-tuning
Corpus < 100K tokens?            -> Long context (simplest)
Data changes frequently?         -> RAG
Need source citations?           -> RAG
Need lowest latency?             -> Fine-tuning (no retrieval overhead)
Need lowest cost per query?      -> Fine-tuning (no retrieval cost)
Need fastest setup?              -> Long context (no pipeline)
```

---

## Reciprocal Rank Fusion (RRF) Formula

```
RRF_score(doc) = SUM over all rankers:  1 / (k + rank(doc))

k = 60 (standard constant)

Example with 2 rankers:
  doc_id | vector_rank | bm25_rank | RRF_score
  -------|-------------|-----------|----------
  A      | 1           | 5         | 1/61 + 1/65 = 0.0318
  B      | 3           | 2         | 1/63 + 1/62 = 0.0320  <-- winner
  C      | 2           | 10        | 1/62 + 1/70 = 0.0304
  D      | 10          | 1         | 1/70 + 1/61 = 0.0307
```

---

## Prompt Template for RAG

```
System: You are a helpful assistant. Answer the user's question using ONLY
the provided context. If the context does not contain enough information to
answer, say "I don't have enough information to answer that." Cite the
relevant source when possible.

User:
Context:
---
[Chunk 1 - Source: {source_1}]
{chunk_text_1}
---
[Chunk 2 - Source: {source_2}]
{chunk_text_2}
---
[Chunk 3 - Source: {source_3}]
{chunk_text_3}
---

Question: {user_query}
```

---

## Key Numbers to Remember

| Metric | Typical Value |
|---|---|
| OpenAI embedding cost | ~$0.02 / 1M tokens (3-small) |
| Embedding latency | 10-50ms per batch |
| Vector search latency (HNSW) | 1-10ms for 1M vectors |
| Cross-encoder reranking | 100-300ms for 50 documents |
| Chunk size sweet spot | 300-500 tokens |
| Overlap | 10-20% of chunk size |
| Top-K retrieval default | 5-10 |
| Candidates for reranking | 20-50 |
| RRF constant k | 60 |
| HNSW M parameter | 16 (default) |
| HNSW ef_construction | 200 (default) |
| Min eval queries | 50-200 |
