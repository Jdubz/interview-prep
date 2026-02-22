# Module 03: RAG & Retrieval -- Deep Dive

Advanced topics that go beyond core interview knowledge. Study this material
when preparing for senior/staff-level roles or when you need to demonstrate
depth in system design discussions.

---

## 1. Approximate Nearest Neighbor (ANN) Algorithms

At scale, exact nearest neighbor search is O(n * d) per query -- intractable for
millions of vectors. ANN algorithms trade perfect recall for dramatically faster search.

### HNSW (Hierarchical Navigable Small World)

The most widely used ANN algorithm in production vector databases.

**Core idea**: Build a multi-layer graph where each layer is a "small world" network.
Higher layers have fewer nodes and longer edges (coarse navigation), lower layers
have more nodes and shorter edges (fine navigation).

```
Layer 3 (fewest nodes):    A -------- D
                           |          |
Layer 2:                   A --- B -- D --- F
                           |    |     |     |
Layer 1:                   A-B--C--D--E--F--G
                           | |  |  |  |  |  |
Layer 0 (all nodes):       A-B-BC-CD-DE-EF-FG-G
                                 (full graph)

Search: Start at top layer, greedily descend to nearest neighbors,
        refine at each layer until reaching layer 0.
```

**Key parameters**:
- `M`: Max edges per node (higher = better recall, more memory). Default: 16
- `ef_construction`: Search width during build (higher = better graph, slower build). Default: 200
- `ef_search`: Search width during query (higher = better recall, slower query). Default: 100

**Tradeoffs**:
```
                    Memory        Build Time     Query Time     Recall@10
Low M (8)           Low           Fast           Fast           ~90%
Default M (16)      Medium        Medium         Medium         ~95%
High M (32)         High          Slow           Slightly slow  ~99%
```

**Interview insight**: HNSW is the default recommendation for most workloads because
it offers excellent recall (>95%) with logarithmic query time and doesn't require
training. The main downside is memory -- the graph structure adds ~30-50% overhead
on top of vector storage.

### IVF (Inverted File Index)

**Core idea**: Cluster vectors into K partitions using k-means. At query time, only
search the `nprobe` nearest clusters.

```
Training phase (offline):
  Run k-means on all vectors -> K cluster centroids

Index structure:
  Cluster 0: [vec_14, vec_203, vec_891, ...]
  Cluster 1: [vec_7, vec_42, vec_555, ...]
  ...
  Cluster K: [vec_33, vec_78, vec_904, ...]

Query:
  1. Find nprobe nearest cluster centroids to query vector
  2. Search only vectors in those clusters
  3. Return top-k
```

**Key parameters**:
- `nlist` (K): Number of clusters. sqrt(n) to 4*sqrt(n) is typical.
- `nprobe`: Number of clusters to search. Higher = better recall, slower.

**Tradeoffs**:
- Requires training (k-means), which takes time and needs representative data
- More memory-efficient than HNSW (no graph overhead)
- Query time scales with nprobe, not log(n) -- less predictable latency
- Often combined with PQ for compression

### Product Quantization (PQ)

**Core idea**: Compress vectors by splitting them into sub-vectors and quantizing
each sub-vector to the nearest centroid in a learned codebook.

```
Original vector (1536 dims, float32):  6144 bytes

Split into 96 sub-vectors of 16 dims each:
  [sub_0] [sub_1] ... [sub_95]

Each sub-vector quantized to nearest centroid (8-bit code):
  [code_0] [code_1] ... [code_95]

Compressed: 96 bytes (64x compression!)
```

**How distance is computed**:
- Precompute distances from query sub-vectors to all centroids (lookup tables)
- Approximate full vector distance by summing sub-vector distances from tables
- This is extremely fast (table lookups instead of floating point math)

**Tradeoffs**:
- Massive memory savings (10-100x compression)
- Lossy -- recall drops, especially for high-precision tasks
- Usually combined with IVF: IVF narrows candidates, PQ compresses stored vectors
- The IVF-PQ combination is how FAISS handles billion-scale datasets

### Algorithm Selection Guide

```
Dataset size        Recommended               Reasoning
< 100K              Flat (exact) or HNSW      Small enough for exact, HNSW for speed
100K - 10M          HNSW                      Best recall/speed tradeoff
10M - 100M          HNSW or IVF-PQ            Memory constrained -> IVF-PQ
100M - 1B           IVF-PQ or DiskANN         Need compression, possible disk-based
> 1B                IVF-PQ + sharding         Must compress + distribute
```

### DiskANN (Bonus)

Microsoft Research algorithm used in Bing. Stores vectors on SSD instead of RAM,
using a graph-based index optimized for disk access patterns. Milvus supports this.
Key advantage: handle datasets larger than RAM. Key cost: higher query latency
due to disk I/O (still sub-100ms with NVMe).

---

## 2. Query Transformation

The user's raw query is often suboptimal for retrieval. Query transformation
techniques rewrite or expand the query before embedding.

### Query Rewriting

Use the LLM to rewrite the query for better retrieval:

```
Original query: "why is it slow?"

Context (conversation history):
  User: "I'm building a RAG pipeline with pgvector"
  User: "why is it slow?"

Rewritten query: "Why is my pgvector RAG pipeline slow? What causes
                  performance issues with pgvector vector search?"
```

Essential for **conversational RAG** where follow-up queries reference prior context.
Without rewriting, "why is it slow?" would retrieve generic results about slowness.

### HyDE (Hypothetical Document Embeddings)

Instead of embedding the question, generate a hypothetical answer and embed that.
The intuition: the hypothetical answer is linguistically closer to the actual
document than the question is.

```
Query: "How does HNSW handle deletions?"

Step 1 -- LLM generates hypothetical answer:
  "HNSW handles deletions by marking nodes as deleted in the graph
   without removing them immediately. The deleted nodes are skipped
   during search. Periodic rebuilds compact the graph..."

Step 2 -- Embed the hypothetical answer (not the original query)

Step 3 -- Retrieve using that embedding
```

**When HyDE helps**:
- Questions where the answer looks very different from the question
- Technical queries where the answer uses different terminology
- When there's an "asymmetry" between query language and document language

**When HyDE hurts**:
- Factoid questions ("What year was X founded?") -- the LLM might hallucinate a year
- When the hypothetical answer is wrong, it retrieves wrong documents
- Adds LLM call latency to every query

### Multi-Query Expansion

Generate multiple query variations, retrieve for each, merge results:

```
Original: "best practices for chunking PDFs"

Expanded queries:
  1. "best practices for chunking PDFs"
  2. "how to split PDF documents for RAG"
  3. "PDF chunking strategies for vector search"
  4. "optimal chunk size for PDF text extraction"

Retrieve top-10 for each query -> merge with RRF -> deduplicate -> top-10
```

This increases recall by covering different phrasings that might match different
documents. The cost is 4x embedding calls and 4x retrieval calls.

### Step-Back Prompting

For specific questions, step back to a more general question first:

```
Specific: "What are the HNSW parameters for 10M vectors with 99% recall?"

Step-back: "How do you tune HNSW index parameters for different recall targets?"

Retrieve for both, combine context, answer the specific question.
```

This retrieves both specific guidance (if it exists) and general principles
(which help the LLM reason about the specific case).

### Comparison

| Technique | Latency Cost | Best For | Risk |
|---|---|---|---|
| Query rewriting | +1 LLM call | Conversational, ambiguous queries | Rewrite may lose intent |
| HyDE | +1 LLM call + 1 embed | Asymmetric query-doc language | Wrong hypothesis = wrong retrieval |
| Multi-query | +N embeds + N retrievals | Broad recall needs | Higher latency, more compute |
| Step-back | +1 LLM call + 1 retrieval | Specific technical questions | Over-generalization |

---

## 3. Multi-Hop Retrieval

Some questions require information from multiple documents that cannot be found
with a single retrieval pass.

### The Problem

```
Query: "Which team built the service that had the most incidents last quarter?"

Required information:
  1. Incident data from last quarter (which service had most?)
  2. Service ownership data (which team owns that service?)

No single chunk contains both pieces of information.
```

### Iterative Retrieval Pattern

```
Step 1: Retrieve for "most incidents last quarter"
  -> Finds: "The payment-service had 47 incidents in Q3 2025"

Step 2: LLM extracts entity: "payment-service"

Step 3: Retrieve for "team that owns payment-service"
  -> Finds: "payment-service is owned by the Platform Payments team"

Step 4: LLM combines: "The Platform Payments team built payment-service,
         which had the most incidents (47) last quarter."
```

### Implementation Pattern

```python
def multi_hop_retrieve(query: str, max_hops: int = 3) -> str:
    context = []
    current_query = query

    for hop in range(max_hops):
        # Retrieve for current query
        chunks = retrieve(current_query, top_k=5)
        context.extend(chunks)

        # Ask LLM: do we have enough to answer?
        assessment = llm(f"""
            Original question: {query}
            Retrieved context: {context}

            Can you fully answer the question? If yes, provide the answer.
            If no, what additional information do you need? Formulate a
            follow-up search query.
        """)

        if assessment.has_answer:
            return assessment.answer
        else:
            current_query = assessment.follow_up_query

    # Max hops reached, answer with what we have
    return llm(f"Answer based on available context: {context}\n\nQuestion: {query}")
```

### Challenges

- **Latency**: Each hop adds retrieval + LLM call latency
- **Error propagation**: Wrong entity extraction in hop 1 derails hop 2
- **When to stop**: Hard to know if more hops will help
- **Context accumulation**: Context grows with each hop, may exceed window

---

## 4. Agentic RAG

Instead of a fixed pipeline, an LLM agent decides when and what to retrieve.

### Self-RAG (Self-Reflective RAG)

The model learns to:
1. Decide if retrieval is needed
2. Retrieve if needed
3. Critique the retrieved results
4. Generate an answer
5. Evaluate its own answer for faithfulness

```
Query: "What's 2+2?"
  Agent: [NO_RETRIEVAL_NEEDED] -> "4"

Query: "What's the current stock price of AAPL?"
  Agent: [RETRIEVAL_NEEDED]
  -> Retrieve stock data
  -> [RELEVANT] The retrieved data is relevant
  -> Generate: "As of market close, AAPL is at $XXX"
  -> [SUPPORTED] The answer is supported by the context
```

### Adaptive Retrieval

```
Agent receives query
    |
    v
  Is retrieval needed?  ----NO----> Generate from parametric knowledge
    |
   YES
    |
    v
  What to retrieve?
    |
    +---> Vector search for semantic content
    +---> SQL query for structured data
    +---> API call for real-time data
    +---> Web search for current events
    |
    v
  Are results sufficient?  ---NO---> Reformulate query, try different source
    |
   YES
    |
    v
  Generate answer with retrieved context
    |
    v
  Does the answer look correct?  ---NO---> Retrieve more, regenerate
    |
   YES
    |
    v
  Return answer
```

### CRAG (Corrective RAG)

Evaluates retrieval quality and self-corrects:

```
1. Retrieve documents for query
2. Evaluate relevance of each document:
   - CORRECT: Document is relevant -> use it
   - AMBIGUOUS: Partially relevant -> refine query, re-retrieve
   - INCORRECT: Not relevant -> try web search as fallback
3. Generate answer from filtered/corrected context
```

### Tool-Augmented RAG

The agent has access to multiple retrieval tools:

```python
tools = [
    {"name": "search_docs", "description": "Search internal documentation"},
    {"name": "search_code", "description": "Search codebase"},
    {"name": "query_db", "description": "Run SQL against metrics database"},
    {"name": "search_web", "description": "Search the internet"},
    {"name": "search_tickets", "description": "Search Jira/Linear tickets"},
]
```

The LLM decides which tool(s) to call based on the query. This is essentially
an agent with retrieval tools -- covered in more depth in Module 04 (Agents).

---

## 5. RAG Evaluation Metrics

### The RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) defines four core metrics:

#### Faithfulness

Does the answer only contain information from the retrieved context?

```
Score = (number of claims in answer supported by context) /
        (total number of claims in answer)

Example:
  Context: "Python was created by Guido van Rossum in 1991."
  Question: "Who created Python and when?"
  Answer: "Python was created by Guido van Rossum in 1991.
           It has become the most popular language for data science."

  Claims in answer:
    1. "Created by Guido van Rossum" -> SUPPORTED by context
    2. "In 1991" -> SUPPORTED by context
    3. "Most popular for data science" -> NOT in context (hallucination)

  Faithfulness = 2/3 = 0.67
```

#### Answer Relevance

Does the answer address the question asked?

```
Method: Generate N questions from the answer, compute similarity between
        generated questions and the original question.

Score = mean cosine similarity of generated questions to original question

Example:
  Question: "What are HNSW parameters?"
  Answer: "HNSW uses three key parameters: M, ef_construction, and ef_search..."

  Generated questions from answer:
    "What parameters does HNSW use?" (sim: 0.95)
    "How do you configure HNSW?" (sim: 0.88)

  Answer relevance = mean(0.95, 0.88) = 0.915
```

#### Context Precision

Are the retrieved chunks relevant to answering the question?

```
Score = weighted precision of relevant chunks at each position

         sum(precision@k * relevance@k for k in 1..K)
       = ------------------------------------------------
         total number of relevant chunks

Rewards relevant chunks appearing at higher ranks.
```

#### Context Recall

Does the retrieved context contain all the information needed to answer?

```
Score = (number of ground-truth answer claims found in context) /
        (total number of ground-truth answer claims)

Requires ground-truth answers for evaluation.
```

### Retrieval-Specific Metrics

| Metric | Formula | Measures | Target |
|---|---|---|---|
| Recall@K | (relevant in top-K) / (total relevant) | Coverage | > 0.8 |
| Precision@K | (relevant in top-K) / K | Signal-to-noise | > 0.6 |
| MRR | 1 / (rank of first relevant result) | Top-result quality | > 0.7 |
| NDCG@K | DCG@K / ideal DCG@K | Ranking quality | > 0.7 |
| Hit Rate@K | 1 if any relevant in top-K, else 0 | Basic coverage | > 0.9 |

### End-to-End Evaluation

```
                     Retrieval Quality              Generation Quality
                     ------------------              ------------------
                     Recall@K                        Faithfulness
Query + Ground Truth Precision@K        Retrieved    Answer Relevance
  ------>            MRR            ---> Context ---> Answer Correctness
                     NDCG@K              + Query      Citation Accuracy
                     Context Precision
                     Context Recall
```

### Building an Evaluation Set

You need:
1. **Queries**: Representative user questions (50-200 minimum for statistical significance)
2. **Ground truth documents**: Which chunks should be retrieved for each query
3. **Ground truth answers**: Expected correct answers (for generation evaluation)
4. **Relevance judgments**: Binary (relevant/not) or graded (0-3 scale)

**Sourcing eval data**:
- Mine from query logs (real user queries)
- Have domain experts create query-answer pairs
- Use LLMs to generate synthetic queries from documents (then human-verify)
- Augment with adversarial queries (edge cases, ambiguous, multi-hop)

---

## 6. Failure Modes and Debugging

### Failure Mode: Empty or Irrelevant Retrieval

**Symptoms**: "I don't have information about that" or off-topic answers.

**Root causes**:
- Query-document vocabulary mismatch (query uses different terms than docs)
- Wrong embedding model for the domain
- Chunk size too large (relevant info diluted by noise)
- Missing documents (content gap in corpus)

**Fixes**:
```
1. Add hybrid search (BM25 catches exact term matches)
2. Try query expansion (multiple phrasings)
3. Reduce chunk size (more precise retrieval)
4. Audit corpus for coverage gaps
5. Try a domain-specific or larger embedding model
```

### Failure Mode: Lost in the Middle

**Symptoms**: LLM ignores relevant chunks that appear in the middle of the context.

Research shows LLMs attend more to information at the beginning and end of the
prompt, "losing" information in the middle. This is especially problematic when
passing many retrieved chunks.

**Fixes**:
```
1. Put most relevant chunks first and last (sandwich layout)
2. Reduce number of chunks (less middle to lose)
3. Use a reranker to ensure only high-quality chunks are included
4. Summarize chunks before injection
5. Use a model known for good long-context attention (Claude, GPT-4 Turbo)
```

### Failure Mode: Conflicting Sources

**Symptoms**: Answer flip-flops or contradicts itself.

**Root causes**: Multiple chunks contain contradictory information (e.g., different
versions of documentation, outdated vs current info).

**Fixes**:
```
1. Add date metadata, prefer recent chunks
2. Add version metadata, filter to latest version
3. Instruct the LLM to identify and flag conflicts
4. Deduplicate near-identical chunks at ingestion
5. Use authoritative source ranking in metadata
```

### Failure Mode: Hallucination Despite Context

**Symptoms**: Answer contains facts not present in retrieved chunks.

The LLM's parametric knowledge overrides the retrieved context, or the LLM
"fills in gaps" in the context with plausible-sounding but incorrect information.

**Fixes**:
```
1. Strengthen system prompt: "ONLY use the provided context"
2. Add: "If the context doesn't contain the answer, say so explicitly"
3. Require citations: "For each claim, cite the context section"
4. Post-generation fact-check: verify each claim against context
5. Lower temperature to reduce creative generation
6. Use faithfulness evaluation in CI/CD pipeline
```

### Failure Mode: Chunking Artifacts

**Symptoms**: Retrieved chunks are partial, missing key information.

**Root causes**:
- Chunks split mid-thought, mid-table, mid-code-block
- Important context in the adjacent chunk that wasn't retrieved
- Headers/metadata separated from their content

**Fixes**:
```
1. Use document-aware chunking (respect structural boundaries)
2. Add parent-child chunking (retrieve small, return parent)
3. Increase overlap between adjacent chunks
4. Include surrounding context: prepend section header to each chunk
5. Retrieve adjacent chunks (chunk N-1 and N+1) alongside matched chunk
```

### Debugging Checklist

```
When RAG quality is poor, debug in this order:

1. CHECK RETRIEVAL FIRST
   - Print the retrieved chunks for failing queries
   - Are relevant chunks in the top-K?
   - If not: retrieval problem (embedding model, chunking, search method)
   - If yes: generation problem (prompt, model, context assembly)

2. IF RETRIEVAL IS WRONG:
   [ ] Are the relevant documents even in the corpus?
   [ ] Try the query on the vector DB directly -- what comes back?
   [ ] Compare embedding similarity scores -- are they reasonable?
   [ ] Try BM25/keyword search -- does it find what vector misses?
   [ ] Review chunk boundaries -- are they sensible?
   [ ] Check metadata filters -- are they accidentally excluding results?

3. IF RETRIEVAL IS RIGHT BUT GENERATION IS WRONG:
   [ ] Print the full prompt sent to the LLM
   [ ] Is the context too long? (lost in the middle)
   [ ] Is the system prompt clear about using only provided context?
   [ ] Is the relevant chunk buried in noise? (reranking needed)
   [ ] Try with a stronger model -- does it work? (model capability)
   [ ] Lower temperature -- does it help? (reduce creativity)
```

---

## 7. Advanced Chunking

### Parent-Child Chunking

Embed small chunks for precise matching, but return the parent chunk for richer context.

```
Document: "API Reference Guide"
  |
  +-- Section: "Authentication" (parent -- returned to LLM)
  |     |
  |     +-- Paragraph 1 (child -- used for embedding + search)
  |     +-- Paragraph 2 (child -- used for embedding + search)
  |     +-- Code example (child -- used for embedding + search)
  |
  +-- Section: "Rate Limiting" (parent)
        |
        +-- Paragraph 1 (child)
        +-- Paragraph 2 (child)
```

**Implementation**:
- Store parent_chunk_id as metadata on each child chunk
- Retrieve child chunks via vector search
- Look up and return parent chunks (deduplicating if multiple children matched)
- Send parent chunks to the LLM

**Tradeoff**: Better context at the cost of sending larger chunks to the LLM,
potentially including irrelevant content.

### Sliding Window with Overlap

```
Document: [=======================]

Window 1: [=======]
Window 2:    [=======]
Window 3:       [=======]
Window 4:          [=======]

Overlap = 30-50% of window size
```

Ensures every passage appears in at least 2-3 chunks, so information at
boundaries is never lost. The cost is more chunks to store and search.

### Late Chunking

Embed the full document through the transformer first, then chunk the output
token embeddings (not the input text).

```
Traditional: chunk text -> embed each chunk independently
Late:        embed full document -> chunk the token embeddings -> pool per chunk

Advantage: Each chunk's embedding carries context from the full document
           (because self-attention saw the whole document before chunking)
```

This approach was introduced by Jina AI. It requires models that support
long input sequences and gives each chunk embedding awareness of the full
document context. Not widely adopted yet but promising.

### Proposition-Based Chunking

Decompose documents into atomic propositions (single facts) and embed each:

```
Original paragraph:
  "Python was created by Guido van Rossum and first released in 1991.
   It emphasizes code readability and supports multiple paradigms."

Propositions:
  1. "Python was created by Guido van Rossum."
  2. "Python was first released in 1991."
  3. "Python emphasizes code readability."
  4. "Python supports multiple programming paradigms."
```

Each proposition is a self-contained, atomic fact. This gives extremely precise
retrieval but produces many chunks and loses cross-proposition context.

**When to use**: Fact-heavy corpora (encyclopedias, knowledge bases) where
precision matters more than context richness.

### Contextual Chunk Headers

Prepend document/section context to each chunk before embedding:

```
Original chunk:
  "The function returns null if the user is not found."

With contextual header:
  "[Document: API Reference | Section: User Service | Subsection: Lookup Methods]
   The function returns null if the user is not found."
```

This makes each chunk self-contained. The embedding captures not just the chunk
content but also its location in the document hierarchy. Anthropic published
research showing this significantly improves retrieval accuracy.

---

## 8. Knowledge Graphs + RAG

### Why Combine?

Vector search finds semantically similar text, but it doesn't understand
**relationships** between entities. Knowledge graphs explicitly encode entities
and their relationships.

```
Vector search: "Who reports to the VP of Engineering?"
  -> Finds paragraphs mentioning VP of Engineering
  -> May or may not contain reporting structure

Knowledge graph: VP_of_Engineering --[manages]--> Team_Lead_A
                                   --[manages]--> Team_Lead_B
                 Team_Lead_A       --[manages]--> Engineer_1
  -> Traversal gives exact answer
```

### Graph-Augmented Retrieval (GraphRAG)

```
Query
  |
  +---> Extract entities from query ("VP of Engineering")
  |
  +---> Traverse knowledge graph for related entities
  |       VP_Engineering -> manages -> [Team_Lead_A, Team_Lead_B]
  |       Team_Lead_A -> manages -> [Engineer_1, Engineer_2]
  |
  +---> Use extracted entities to enhance retrieval
  |       Vector search for query + graph context
  |
  +---> Combine graph facts + retrieved text chunks
  |
  +---> LLM generates answer from combined context
```

### Building the Knowledge Graph

```
Source documents
    |
    v
Entity extraction (NER or LLM-based)
    |
    v
Relation extraction (LLM-based: "what relationships exist between entities?")
    |
    v
Knowledge graph construction (Neo4j, Amazon Neptune, or in-memory)
    |
    v
Graph stored alongside vector index
```

**Entity extraction prompt**:
```
Extract all entities (people, organizations, products, concepts) and their
relationships from the following text. Return as triples:
(entity1, relationship, entity2)
```

### Microsoft GraphRAG

Microsoft Research's approach uses LLMs to build a hierarchical community graph:

```
1. Extract entities and relationships from all documents
2. Build a graph of entity relationships
3. Detect communities (clusters) in the graph using Leiden algorithm
4. Generate summaries at multiple community levels
5. At query time, map query to relevant communities
6. Use community summaries + retrieved chunks for generation
```

This is especially powerful for **global queries** ("What are the main themes
across all documents?") where standard RAG fails because no single chunk
contains the answer.

### When to Use Knowledge Graphs

| Scenario | Standard RAG | Graph-Augmented RAG |
|---|---|---|
| Factoid questions about specific passages | Good | Overkill |
| Multi-hop reasoning across entities | Poor | Good |
| Relationship queries (who, reports-to) | Poor | Excellent |
| Global summarization across corpus | Poor | Good |
| Simple Q&A over documents | Good | Unnecessary complexity |

**Interview insight**: Knowledge graphs add significant complexity. Recommend them
only when the query patterns require entity relationships or multi-hop reasoning
that pure text retrieval cannot handle. For most RAG use cases, good chunking +
hybrid search + reranking is sufficient.

---

## 9. Production RAG Patterns

### Caching

```
Query Cache:
  query_embedding -> cache key
  If similar query was asked recently, return cached result

Semantic Cache:
  Embed query, check if any cached query embedding is within threshold
  Avoids redundant LLM calls for paraphrased questions

Chunk Cache:
  Cache frequently retrieved chunks in memory
  Reduces vector DB latency
```

### Streaming RAG

```
1. Retrieve chunks (wait for full retrieval)
2. Start streaming LLM response as chunks arrive
3. User sees first tokens while generation continues

Optimization: Start LLM generation as soon as top-1 chunk arrives,
              inject remaining chunks into a follow-up message.
              (Requires model that handles this gracefully.)
```

### Incremental Indexing

```
Document changes detected (webhook, file watcher, scheduled scan)
    |
    v
Identify changed documents (hash comparison)
    |
    v
Re-parse and re-chunk only changed documents
    |
    v
Delete old chunks for changed documents
    |
    v
Embed and insert new chunks
    |
    v
Optionally rebuild ANN index (or let DB handle online updates)
```

### Observability

Instrument every stage of the pipeline:

```
Metrics to track:
  - Retrieval latency (p50, p95, p99)
  - Number of chunks retrieved per query
  - Reranker latency
  - LLM generation latency
  - End-to-end latency
  - Retrieval relevance scores (distribution)
  - Faithfulness scores (sampled)
  - User feedback (thumbs up/down)
  - Cache hit rate
  - Embedding model throughput
```

### Guardrails

```
Pre-retrieval:
  - Input validation (length limits, content policy)
  - Query classification (is this in-scope?)

Post-retrieval:
  - Check if retrieval scores meet minimum threshold
  - If not, return "I don't have information about that" instead of hallucinating

Post-generation:
  - Fact-check claims against context
  - Check for PII leakage
  - Content safety filter
  - Citation verification
```

---

## Summary: What Separates Good RAG from Great RAG

```
Basic RAG:
  Chunk documents -> embed -> vector search -> prompt -> generate

Good RAG:
  + Document-aware chunking with metadata
  + Hybrid search (vector + BM25)
  + Cross-encoder reranking
  + Strong system prompt with grounding instructions
  + Evaluation pipeline (RAGAS metrics)

Great RAG:
  + Query transformation (rewriting, HyDE, expansion)
  + Parent-child chunking with contextual headers
  + Multi-hop retrieval for complex queries
  + Agentic retrieval (LLM decides when/what to retrieve)
  + Semantic caching
  + Incremental indexing with deduplication
  + Full observability and continuous evaluation
  + Feedback loop from user signals to retrieval tuning
```
