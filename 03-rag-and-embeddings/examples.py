"""
RAG & Embeddings Examples

Python patterns for embedding generation, similarity search,
and a complete RAG pipeline. Uses generic interfaces — swap in any provider.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Generic Interfaces
# ---------------------------------------------------------------------------

# Vector embedding — a list of floats representing semantic meaning.
Embedding = list[float]


class EmbedFn(Protocol):
    """A function that converts text into an embedding vector."""
    async def __call__(self, text: str) -> Embedding: ...


class BatchEmbedFn(Protocol):
    """Batch embedding — more efficient for ingestion."""
    async def __call__(self, texts: list[str]) -> list[Embedding]: ...


@dataclass
class ChunkMetadata:
    source: str
    title: str | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredChunk:
    """A stored document chunk with its embedding and metadata."""
    id: str
    text: str
    embedding: Embedding
    metadata: ChunkMetadata


@dataclass
class RetrievalResult:
    """A retrieval result with similarity score."""
    chunk: StoredChunk
    score: float


class VectorStore(Protocol):
    """Minimal vector store interface."""
    async def upsert(self, chunks: list[StoredChunk]) -> None: ...
    async def query(
        self, embedding: Embedding, top_k: int, filter: dict[str, Any] | None = None
    ) -> list[RetrievalResult]: ...


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


class CompletionFn(Protocol):
    async def __call__(
        self,
        messages: list[Message],
        config: dict[str, Any],
    ) -> str: ...


# ---------------------------------------------------------------------------
# Similarity Functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """
    Cosine similarity between two vectors.
    Returns a value between -1 and 1 (1 = identical direction).
    """
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Chunking Strategies
# ---------------------------------------------------------------------------

@dataclass
class ChunkOptions:
    max_tokens: int
    overlap: int
    tokens_per_char: float = 0.25  # Approximate for English


def recursive_chunk(text: str, options: ChunkOptions) -> list[str]:
    """
    Recursive text splitter — respects natural boundaries.
    Tries to split on paragraphs, then sentences, then by size.
    """
    max_chars = int(options.max_tokens / options.tokens_per_char)
    overlap_chars = int(options.overlap / options.tokens_per_char)

    # If text fits in one chunk, return it
    if len(text) <= max_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Try splitting on natural boundaries, coarsest first
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part

            if len(candidate) > max_chars and current:
                chunks.append(current.strip())
                # Include overlap from the end of the previous chunk
                overlap_text = current[-overlap_chars:] if overlap_chars else ""
                current = f"{overlap_text}{sep}{part}"
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks

    # Fallback: hard split by character count with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars

    return chunks


# ---------------------------------------------------------------------------
# In-Memory Vector Store (for demonstration)
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """
    Simple in-memory vector store. In production, use Pinecone, Qdrant,
    pgvector, etc. This demonstrates the core interface.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}

    async def upsert(self, chunks: list[StoredChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    async def query(
        self,
        embedding: Embedding,
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        candidates = list(self._chunks.values())

        # Apply metadata filters
        if filter:
            candidates = [
                c for c in candidates
                if all(
                    getattr(c.metadata, k, c.metadata.extra.get(k)) == v
                    for k, v in filter.items()
                )
            ]

        # Score all candidates
        scored = [
            RetrievalResult(chunk=c, score=cosine_similarity(embedding, c.embedding))
            for c in candidates
        ]

        # Sort by score descending, return top-k
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

@dataclass
class IngestionOptions:
    chunk_options: ChunkOptions
    enrich_chunks: bool = False


async def ingest_document(
    document: dict[str, str],  # keys: text, source, title (optional)
    embed: BatchEmbedFn,
    store: VectorStore,
    options: IngestionOptions,
) -> dict[str, int]:
    """Ingest a document: chunk it, embed the chunks, store in vector DB."""
    # 1. Chunk the document
    text_chunks = recursive_chunk(document["text"], options.chunk_options)

    # 2. Optionally enrich chunks with document context
    texts_to_embed = []
    for text in text_chunks:
        if options.enrich_chunks and document.get("title"):
            texts_to_embed.append(f"{document['title']}\n\n{text}")
        else:
            texts_to_embed.append(text)

    # 3. Generate embeddings (batch for efficiency)
    embeddings = await embed(texts_to_embed)

    # 4. Create stored chunks with metadata
    stored_chunks = [
        StoredChunk(
            id=f"{document['source']}:chunk-{i}",
            text=text,
            embedding=embeddings[i],
            metadata=ChunkMetadata(
                source=document["source"],
                title=document.get("title"),
                chunk_index=i,
                total_chunks=len(text_chunks),
            ),
        )
        for i, text in enumerate(text_chunks)
    ]

    # 5. Store in vector DB
    await store.upsert(stored_chunks)

    return {"chunks_created": len(stored_chunks)}


# ---------------------------------------------------------------------------
# Retrieval Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RetrievalOptions:
    top_k: int
    score_threshold: float | None = None
    filter: dict[str, Any] | None = None


async def retrieve(
    query: str,
    embed: EmbedFn,
    store: VectorStore,
    options: RetrievalOptions,
) -> list[RetrievalResult]:
    """Retrieve relevant chunks for a query."""
    query_embedding = await embed(query)

    results = await store.query(
        query_embedding, options.top_k, options.filter
    )

    # Filter by score threshold
    if options.score_threshold is not None:
        results = [r for r in results if r.score >= options.score_threshold]

    return results


# ---------------------------------------------------------------------------
# Full RAG Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGOptions:
    retrieval: RetrievalOptions
    model: str
    system_prompt: str | None = None
    include_sources: bool = False


@dataclass
class RAGSource:
    source: str
    score: float
    excerpt: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[RAGSource]
    context_chunks_used: int


async def rag_query(
    query: str,
    embed: EmbedFn,
    store: VectorStore,
    complete: CompletionFn,
    options: RAGOptions,
) -> RAGResponse:
    """Complete RAG pipeline: retrieve relevant context, then generate an answer."""
    # 1. Retrieve relevant chunks
    results = await retrieve(query, embed, store, options.retrieval)

    # 2. Build context from retrieved chunks
    context_blocks = []
    for i, r in enumerate(results):
        source_label = r.chunk.metadata.title or r.chunk.metadata.source
        context_blocks.append(
            f'<source index="{i + 1}" name="{source_label}">\n{r.chunk.text}\n</source>'
        )

    context = "\n\n".join(context_blocks)

    # 3. Build the prompt
    sources_instruction = (
        "\n- End your response with a 'Sources:' section listing which sources you used"
        if options.include_sources
        else ""
    )

    system_prompt = options.system_prompt or (
        "You are a helpful assistant that answers questions based on the provided sources.\n\n"
        "Rules:\n"
        "- ONLY use information from the provided sources to answer\n"
        "- If the sources don't contain enough information, say so clearly\n"
        f"- Reference sources by their index when making claims{sources_instruction}"
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=f"<context>\n{context}\n</context>\n\nQuestion: {query}"),
    ]

    # 4. Generate answer
    answer = await complete(messages, {"model": options.model, "temperature": 0})

    # 5. Build response with source metadata
    return RAGResponse(
        answer=answer,
        sources=[
            RAGSource(
                source=r.chunk.metadata.source,
                score=r.score,
                excerpt=r.chunk.text[:200] + "...",
            )
            for r in results
        ],
        context_chunks_used=len(results),
    )


# ---------------------------------------------------------------------------
# Conversational RAG — Query Rewriting
# ---------------------------------------------------------------------------

async def rewrite_query_for_retrieval(
    current_query: str,
    conversation_history: list[Message],
    complete: CompletionFn,
) -> str:
    """
    In multi-turn conversations, follow-up questions often reference previous
    context. Rewrite the query to be standalone before retrieval.
    """
    if not conversation_history:
        return current_query

    recent_history = conversation_history[-6:]  # Last 3 turns
    history_text = "\n".join(f"{m.role}: {m.content}" for m in recent_history)

    result = await complete(
        [
            Message(
                role="system",
                content=(
                    "Rewrite the user's latest message as a standalone search query.\n"
                    "Incorporate any relevant context from the conversation history.\n"
                    "Return ONLY the rewritten query, nothing else.\n\n"
                    "If the message is already standalone, return it unchanged."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest message: {current_query}\n\n"
                    "Standalone query:"
                ),
            ),
        ],
        {"model": "gpt-4o-mini", "temperature": 0},
    )

    return result.strip()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (for hybrid search)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Merge results from multiple search methods (e.g., vector + keyword).
    Simple, effective, and parameter-free (k=60 is standard).
    """
    k = 60  # Standard constant
    scores: dict[str, dict[str, Any]] = {}

    for result_list in ranked_lists:
        for rank, result in enumerate(result_list):
            chunk_id = result.chunk.id
            rrf_score = 1 / (k + rank + 1)

            if chunk_id in scores:
                scores[chunk_id]["score"] += rrf_score
            else:
                scores[chunk_id] = {"score": rrf_score, "chunk": result.chunk}

    merged = [
        RetrievalResult(chunk=entry["chunk"], score=entry["score"])
        for entry in scores.values()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

async def demo(
    embed: EmbedFn, batch_embed: BatchEmbedFn, complete: CompletionFn
) -> None:
    store = InMemoryVectorStore()

    # Ingest documents
    await ingest_document(
        {"text": "Your long document text here...", "source": "docs/api-guide.md", "title": "API Guide"},
        batch_embed,
        store,
        IngestionOptions(
            chunk_options=ChunkOptions(max_tokens=500, overlap=50),
            enrich_chunks=True,
        ),
    )

    # Query with RAG
    response = await rag_query(
        "How do I authenticate with the API?",
        embed,
        store,
        complete,
        RAGOptions(
            retrieval=RetrievalOptions(top_k=5, score_threshold=0.7),
            model="gpt-4o",
            include_sources=True,
        ),
    )

    print("Answer:", response.answer)
    print("Sources:", response.sources)
