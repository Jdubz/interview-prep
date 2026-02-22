"""
Module 03: RAG & Retrieval -- Complete, Runnable Patterns

Demonstrates core RAG components with an in-memory vector store.
No external dependencies beyond numpy (for vector math).

Each example is self-contained. Read top-to-bottom or jump to specific patterns.

NOTE: In production, replace the in-memory store with a real vector database
and the mock embedding function with an actual embedding model API call.
"""

from __future__ import annotations

import math
import re
import hashlib
from dataclasses import dataclass, field
from typing import Protocol, Callable


# ---------------------------------------------------------------------------
# Shared types and utilities
# ---------------------------------------------------------------------------

Vector = list[float]


@dataclass
class Document:
    """A chunk of text with metadata, ready for storage in a vector DB."""
    id: str
    text: str
    vector: Vector = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result from vector search."""
    document: Document
    score: float


# Mock embedding function -- produces deterministic vectors from text.
# In production, replace with OpenAI/Cohere/local model API call.
def mock_embed(text: str, dims: int = 64) -> Vector:
    """Deterministic pseudo-embedding for demonstration.
    Uses character-level hashing to produce a fixed-length vector.
    NOT semantically meaningful -- just structurally correct."""
    h = hashlib.sha256(text.lower().encode()).hexdigest()
    raw = [int(h[i:i+2], 16) / 255.0 - 0.5 for i in range(0, min(len(h), dims * 2), 2)]
    # Pad or truncate to desired dimensions
    raw = (raw * (dims // len(raw) + 1))[:dims]
    # L2 normalize (like real embedding models do)
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors.
    For normalized vectors, this equals the dot product."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: Vector, b: Vector) -> float:
    """Dot product -- equivalent to cosine similarity for normalized vectors."""
    return sum(x * y for x, y in zip(a, b))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """L2 distance. Lower = more similar (opposite convention from similarity)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Pattern 1: In-Memory Vector Store
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """Minimal vector store for demonstrating RAG patterns.
    Brute-force search -- O(n) per query. Fine for < 100K documents."""

    def __init__(self, embed_fn: Callable[[str], Vector] = mock_embed):
        self.documents: list[Document] = []
        self.embed_fn = embed_fn

    def add(self, text: str, metadata: dict | None = None) -> Document:
        """Embed and store a document."""
        doc_id = f"doc-{len(self.documents)}"
        vector = self.embed_fn(text)
        doc = Document(id=doc_id, text=text, vector=vector, metadata=metadata or {})
        self.documents.append(doc)
        return doc

    def add_batch(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """Batch insert -- in production, batch the embedding API call too."""
        metadatas = metadatas or [{}] * len(texts)
        return [self.add(text, meta) for text, meta in zip(texts, metadatas)]

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Vector similarity search with optional metadata filtering.

        Demonstrates integrated filtering: filter and score in one pass."""
        query_vector = self.embed_fn(query)
        results = []

        for doc in self.documents:
            # Metadata filter -- skip documents that don't match
            if metadata_filter:
                if not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            score = cosine_similarity(query_vector, doc.vector)
            results.append(SearchResult(document=doc, score=score))

        # Sort by score descending, return top-k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Pattern 2: Chunking Strategies
# ---------------------------------------------------------------------------

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size chunking with overlap.
    Simplest strategy. Splits mid-sentence -- use as baseline only."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_recursive(
    text: str,
    max_size: int = 500,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursive character chunking (LangChain-style).
    Tries to split on paragraph breaks, then sentences, then words."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # Base case: text fits in one chunk
    if len(text) <= max_size:
        return [text.strip()] if text.strip() else []

    # Try each separator in order of preference
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    # If a single part exceeds max_size, recurse with finer separator
                    if len(part) > max_size:
                        remaining_seps = separators[separators.index(sep) + 1:]
                        chunks.extend(chunk_recursive(part, max_size, remaining_seps))
                        current = ""
                    else:
                        current = part

            if current.strip():
                chunks.append(current.strip())
            return chunks

    # Last resort: hard split at max_size
    return [text[i:i + max_size] for i in range(0, len(text), max_size)]


def chunk_by_headers(text: str, max_size: int = 1000) -> list[dict]:
    """Document-aware chunking for Markdown.
    Splits on headers, preserves section hierarchy as metadata.
    Returns dicts with 'text' and 'metadata' keys."""
    chunks = []
    current_headers: list[str] = []
    current_text = ""

    for line in text.split("\n"):
        # Detect markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)', line)

        if header_match:
            # Save previous chunk if non-empty
            if current_text.strip():
                chunks.append({
                    "text": current_text.strip(),
                    "metadata": {
                        "section_path": " > ".join(current_headers) if current_headers else "root",
                        "heading": current_headers[-1] if current_headers else "",
                    },
                })

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Maintain header hierarchy
            current_headers = current_headers[:level - 1]
            current_headers.append(title)
            current_text = line + "\n"
        else:
            current_text += line + "\n"

            # If current chunk exceeds max size, flush it
            if len(current_text) > max_size:
                chunks.append({
                    "text": current_text.strip(),
                    "metadata": {
                        "section_path": " > ".join(current_headers) if current_headers else "root",
                        "heading": current_headers[-1] if current_headers else "",
                    },
                })
                current_text = ""

    # Don't forget the last chunk
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "metadata": {
                "section_path": " > ".join(current_headers) if current_headers else "root",
                "heading": current_headers[-1] if current_headers else "",
            },
        })

    return chunks


def chunk_semantic_boundaries(
    text: str,
    embed_fn: Callable[[str], Vector] = mock_embed,
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 2,
) -> list[str]:
    """Semantic chunking: split where embedding similarity between
    consecutive sentences drops below threshold.

    In production, use a real embedding model for meaningful similarity."""
    # Split into sentences (simplified -- production would use spaCy or similar)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= min_chunk_sentences:
        return [text]

    # Embed each sentence
    embeddings = [embed_fn(s) for s in sentences]

    # Find split points where similarity drops
    chunks = []
    current_chunk_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])

        if sim < similarity_threshold and len(current_chunk_sentences) >= min_chunk_sentences:
            # Similarity dropped -- split here
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


# ---------------------------------------------------------------------------
# Pattern 3: Hybrid Search (Vector + Keyword)
# ---------------------------------------------------------------------------

def bm25_score(query_terms: list[str], doc_text: str, avg_doc_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 scoring for a single document.
    In production, use a proper BM25 implementation (rank_bm25, Elasticsearch, etc.)."""
    doc_terms = doc_text.lower().split()
    doc_len = len(doc_terms)
    score = 0.0

    for term in query_terms:
        # Term frequency in this document
        tf = doc_terms.count(term.lower())
        if tf == 0:
            continue
        # BM25 TF component (saturating TF)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        # Simplified IDF (would need corpus stats in production)
        idf = 1.0  # Placeholder -- real BM25 needs document frequency across corpus
        score += idf * tf_component

    return score


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merge multiple ranked result lists.

    Each ranked_list is a list of document IDs ordered by relevance.
    Returns merged list of (doc_id, rrf_score) sorted by score descending."""
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearcher:
    """Combines vector search and keyword (BM25) search with RRF fusion."""

    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5, vector_weight: int = 1, keyword_weight: int = 1) -> list[SearchResult]:
        """Hybrid search: vector + BM25, merged with RRF."""
        # Stage 1a: Vector search
        vector_results = self.vector_store.search(query, top_k=top_k * 4)
        vector_ranked = [r.document.id for r in vector_results]

        # Stage 1b: Keyword (BM25) search
        query_terms = query.lower().split()
        avg_doc_len = sum(len(d.text.split()) for d in self.vector_store.documents) / max(len(self.vector_store.documents), 1)

        keyword_scores = []
        for doc in self.vector_store.documents:
            score = bm25_score(query_terms, doc.text, avg_doc_len)
            keyword_scores.append((doc.id, score))

        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keyword_ranked = [doc_id for doc_id, _ in keyword_scores[:top_k * 4]]

        # Stage 2: RRF fusion
        # Weight by repeating ranked lists (simple weighting mechanism)
        ranked_lists = [vector_ranked] * vector_weight + [keyword_ranked] * keyword_weight
        fused = reciprocal_rank_fusion(ranked_lists)

        # Map back to SearchResult objects
        doc_map = {d.id: d for d in self.vector_store.documents}
        results = []
        for doc_id, rrf_score in fused[:top_k]:
            if doc_id in doc_map:
                results.append(SearchResult(document=doc_map[doc_id], score=rrf_score))

        return results


# ---------------------------------------------------------------------------
# Pattern 4: Cross-Encoder Reranker
# ---------------------------------------------------------------------------

def cross_encoder_rerank(
    query: str,
    results: list[SearchResult],
    top_n: int = 5,
    score_fn: Callable[[str, str], float] | None = None,
) -> list[SearchResult]:
    """Rerank results using a cross-encoder scoring function.

    In production, score_fn would call Cohere Rerank or a local
    cross-encoder model (bge-reranker-v2-m3, ms-marco-MiniLM).

    The mock scorer uses word overlap as a proxy for relevance."""

    def default_score_fn(q: str, doc_text: str) -> float:
        """Mock cross-encoder: word overlap ratio.
        Real cross-encoders jointly encode (query, doc) through a transformer."""
        q_words = set(q.lower().split())
        d_words = set(doc_text.lower().split())
        if not q_words:
            return 0.0
        overlap = len(q_words & d_words)
        return overlap / len(q_words)

    scorer = score_fn or default_score_fn

    # Score each (query, document) pair
    reranked = []
    for result in results:
        ce_score = scorer(query, result.document.text)
        reranked.append(SearchResult(document=result.document, score=ce_score))

    # Sort by cross-encoder score, return top-N
    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked[:top_n]


# ---------------------------------------------------------------------------
# Pattern 5: Query Transformation (HyDE)
# ---------------------------------------------------------------------------

def hyde_retrieve(
    query: str,
    vector_store: InMemoryVectorStore,
    generate_fn: Callable[[str], str] | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """HyDE: Hypothetical Document Embeddings.

    Instead of embedding the question, generate a hypothetical answer
    and embed that. The hypothesis is linguistically closer to the
    actual documents than the question is.

    generate_fn: LLM call that generates a hypothetical answer.
    In production, this is an actual LLM API call."""

    def default_generate(q: str) -> str:
        """Mock LLM -- in production, call OpenAI/Anthropic/etc."""
        return f"A comprehensive answer to the question '{q}' would explain that..."

    generator = generate_fn or default_generate

    # Step 1: Generate hypothetical answer
    hypothesis = generator(query)

    # Step 2: Embed the hypothesis (not the original query)
    hypothesis_vector = vector_store.embed_fn(hypothesis)

    # Step 3: Search using the hypothesis embedding
    results = []
    for doc in vector_store.documents:
        score = cosine_similarity(hypothesis_vector, doc.vector)
        results.append(SearchResult(document=doc, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Pattern 6: RAG Evaluation
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    answer: str,
    context_chunks: list[str],
) -> dict:
    """Evaluate whether the answer is grounded in the provided context.

    Simplified version: checks if key phrases from the answer appear
    in the context. Production version uses an LLM to extract claims
    from the answer and verify each against context (RAGAS approach)."""
    # Extract "claims" (simplified: sentences from the answer)
    claims = re.split(r'(?<=[.!?])\s+', answer)
    claims = [c.strip() for c in claims if len(c.strip()) > 10]

    if not claims:
        return {"faithfulness": 1.0, "supported": 0, "total": 0, "details": []}

    context_text = " ".join(context_chunks).lower()
    supported = 0
    details = []

    for claim in claims:
        # Check if key words from the claim appear in context
        claim_words = set(claim.lower().split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it", "this", "that"}
        claim_keywords = claim_words - stop_words

        if not claim_keywords:
            supported += 1
            details.append({"claim": claim, "supported": True, "reason": "no keywords"})
            continue

        # What fraction of claim keywords appear in context?
        found = sum(1 for w in claim_keywords if w in context_text)
        coverage = found / len(claim_keywords)

        is_supported = coverage > 0.5
        if is_supported:
            supported += 1
        details.append({
            "claim": claim,
            "supported": is_supported,
            "keyword_coverage": round(coverage, 2),
        })

    return {
        "faithfulness": round(supported / len(claims), 2),
        "supported": supported,
        "total": len(claims),
        "details": details,
    }


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> dict:
    """Compute retrieval quality metrics.

    retrieved_ids: ordered list of retrieved document IDs
    relevant_ids: set of ground-truth relevant document IDs
    k: evaluate at top-K (defaults to len(retrieved_ids))"""
    k = k or len(retrieved_ids)
    top_k = retrieved_ids[:k]

    # Recall@K: what fraction of relevant docs did we find?
    relevant_found = len(set(top_k) & relevant_ids)
    recall = relevant_found / len(relevant_ids) if relevant_ids else 0.0

    # Precision@K: what fraction of top-K are relevant?
    precision = relevant_found / k if k > 0 else 0.0

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            mrr = 1.0 / i
            break

    # Hit Rate: did we find ANY relevant document?
    hit_rate = 1.0 if relevant_found > 0 else 0.0

    return {
        "recall_at_k": round(recall, 3),
        "precision_at_k": round(precision, 3),
        "mrr": round(mrr, 3),
        "hit_rate": hit_rate,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Pattern 7: Complete RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline tying together all components.

    Demonstrates the full flow:
    query -> (optional transform) -> embed -> retrieve -> (optional rerank)
    -> assemble prompt -> (mock) generate"""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        hybrid: bool = False,
        rerank: bool = False,
        retrieval_k: int = 10,
        final_k: int = 5,
    ):
        self.vector_store = vector_store
        self.hybrid = hybrid
        self.rerank = rerank
        self.retrieval_k = retrieval_k
        self.final_k = final_k

        if hybrid:
            self.hybrid_searcher = HybridSearcher(vector_store)

    def ingest(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Ingest documents into the pipeline."""
        docs = self.vector_store.add_batch(texts, metadatas)
        return len(docs)

    def retrieve(self, query: str, metadata_filter: dict | None = None) -> list[SearchResult]:
        """Retrieve relevant chunks for a query."""
        if self.hybrid:
            results = self.hybrid_searcher.search(query, top_k=self.retrieval_k)
        else:
            results = self.vector_store.search(
                query, top_k=self.retrieval_k, metadata_filter=metadata_filter
            )

        if self.rerank:
            results = cross_encoder_rerank(query, results, top_n=self.final_k)
        else:
            results = results[:self.final_k]

        return results

    def build_prompt(self, query: str, results: list[SearchResult]) -> list[dict]:
        """Assemble the RAG prompt with system instructions and retrieved context."""
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.document.metadata.get("source", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{result.document.text}")

        context = "\n---\n".join(context_parts)

        return [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "If the context doesn't contain enough information, say so. "
                    "Cite the source number (e.g., [Source 1]) for each claim."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

    def query(self, query: str, metadata_filter: dict | None = None) -> dict:
        """Full RAG pipeline: retrieve -> prompt -> (mock) generate."""
        results = self.retrieve(query, metadata_filter)
        prompt = self.build_prompt(query, results)

        # In production, send `prompt` to LLM API.
        # Here we return the assembled prompt for inspection.
        return {
            "query": query,
            "retrieved_chunks": [
                {"text": r.document.text[:100] + "...", "score": round(r.score, 4)}
                for r in results
            ],
            "prompt": prompt,
            "num_chunks": len(results),
        }


# ---------------------------------------------------------------------------
# Demo: Putting It All Together
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the full RAG pipeline with sample data."""
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm. "
        "It builds a multi-layer graph for efficient nearest neighbor search. "
        "Key parameters are M (edges per node), ef_construction, and ef_search.",

        "Cosine similarity measures the angle between two vectors, ranging from -1 to 1. "
        "It is the default metric for text embeddings because it ignores magnitude "
        "and focuses on directional similarity.",

        "Reciprocal Rank Fusion (RRF) merges ranked lists from different search methods. "
        "The formula is: RRF_score = sum(1/(k + rank)) where k=60. "
        "Documents ranking well in multiple methods get boosted.",

        "Cross-encoder rerankers process (query, document) pairs jointly through a transformer. "
        "They are more accurate than bi-encoder similarity but too slow for first-stage retrieval. "
        "Use them to rerank 20-50 candidates from initial retrieval.",

        "Chunk size is one of the most impactful RAG decisions. "
        "Smaller chunks (100-200 tokens) give precise retrieval but less context. "
        "Larger chunks (500-1000 tokens) give more context but lower precision. "
        "Start with 500 tokens and tune based on evaluation.",

        "pgvector is a PostgreSQL extension for vector similarity search. "
        "It supports HNSW and IVFFlat indexes. "
        "Best for teams already using Postgres with moderate scale (under 10M vectors).",
    ]

    metadatas = [
        {"source": "ann-algorithms.md", "topic": "algorithms"},
        {"source": "similarity-metrics.md", "topic": "algorithms"},
        {"source": "hybrid-search.md", "topic": "search"},
        {"source": "reranking.md", "topic": "search"},
        {"source": "chunking-guide.md", "topic": "chunking"},
        {"source": "vector-databases.md", "topic": "databases"},
    ]

    # Initialize pipeline with hybrid search and reranking
    pipeline = RAGPipeline(
        vector_store=InMemoryVectorStore(),
        hybrid=True,
        rerank=True,
        retrieval_k=6,
        final_k=3,
    )

    # Ingest documents
    num_ingested = pipeline.ingest(documents, metadatas)
    print(f"\nIngested {num_ingested} documents")

    # Run a query
    print("\n--- Query: 'How does HNSW work?' ---")
    result = pipeline.query("How does HNSW work?")
    print(f"Retrieved {result['num_chunks']} chunks:")
    for chunk in result["retrieved_chunks"]:
        print(f"  Score: {chunk['score']:.4f} | {chunk['text'][:80]}...")

    # Demonstrate metadata filtering
    print("\n--- Query with metadata filter: topic=search ---")
    result = pipeline.query(
        "What are the best search techniques?",
        metadata_filter={"topic": "search"},
    )
    print(f"Retrieved {result['num_chunks']} chunks (filtered to 'search' topic):")
    for chunk in result["retrieved_chunks"]:
        print(f"  Score: {chunk['score']:.4f} | {chunk['text'][:80]}...")

    # Demonstrate chunking strategies
    print("\n\n" + "=" * 60)
    print("Chunking Strategies Demo")
    print("=" * 60)

    sample_text = """# Vector Databases

## Pinecone
Pinecone is a fully managed vector database. It provides automatic scaling and zero operational overhead. Best for teams that want minimal infrastructure management.

## pgvector
pgvector is a PostgreSQL extension for vector similarity search. It supports HNSW and IVFFlat indexing. Best for teams already using Postgres.

## Choosing a Database
Consider your scale, existing infrastructure, and operational capacity when selecting a vector database. For under 10M vectors with existing Postgres, pgvector is usually the right choice."""

    print("\n--- Fixed-size chunking (200 chars, 30 overlap) ---")
    fixed_chunks = chunk_fixed_size(sample_text, chunk_size=200, overlap=30)
    for i, chunk in enumerate(fixed_chunks):
        print(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:60]}...")

    print("\n--- Recursive chunking (300 chars max) ---")
    recursive_chunks = chunk_recursive(sample_text, max_size=300)
    for i, chunk in enumerate(recursive_chunks):
        print(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:60]}...")

    print("\n--- Header-based chunking ---")
    header_chunks = chunk_by_headers(sample_text)
    for i, chunk in enumerate(header_chunks):
        section = chunk["metadata"]["section_path"]
        print(f"  Chunk {i}: [section: {section}] {chunk['text'][:60]}...")

    # Demonstrate evaluation
    print("\n\n" + "=" * 60)
    print("Evaluation Demo")
    print("=" * 60)

    # Retrieval evaluation
    retrieved = ["doc-0", "doc-3", "doc-1", "doc-5", "doc-4"]
    relevant = {"doc-0", "doc-1", "doc-2"}  # ground truth

    metrics = evaluate_retrieval(retrieved, relevant, k=5)
    print(f"\nRetrieval metrics (K=5):")
    print(f"  Recall@5:    {metrics['recall_at_k']}")
    print(f"  Precision@5: {metrics['precision_at_k']}")
    print(f"  MRR:         {metrics['mrr']}")
    print(f"  Hit Rate:    {metrics['hit_rate']}")

    # Faithfulness evaluation
    context = ["Python was created by Guido van Rossum in 1991."]
    answer = "Python was created by Guido van Rossum in 1991. It is the most popular language for data science."

    faith_result = evaluate_faithfulness(answer, context)
    print(f"\nFaithfulness evaluation:")
    print(f"  Score: {faith_result['faithfulness']}")
    print(f"  Supported claims: {faith_result['supported']}/{faith_result['total']}")


if __name__ == "__main__":
    demo()
