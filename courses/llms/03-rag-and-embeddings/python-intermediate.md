# Python Intermediate: RAG & Embeddings

> **Intermediate Python patterns for RAG systems.** This guide covers Python skills for working with embeddings, vector search, chunking, and retrieval pipelines.

---

## Working with Embeddings (Vectors)

### Understanding Embeddings as Lists

```python
# Embedding = list of floats representing semantic meaning
# Typical sizes: 384 (small), 768 (BERT), 1536 (OpenAI), 3072 (large)

embedding: list[float] = [0.123, -0.456, 0.789, ...]  # 1536 dimensions

# Properties
len(embedding)  # 1536
type(embedding)  # <class 'list'>
type(embedding[0])  # <class 'float'>

# Embeddings are just numbers!
# Similar meanings -> similar numbers
# Dissimilar meanings -> different numbers
```

### Vector Operations

```python
import math

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate similarity between two vectors (-1 to 1).
    1.0 = identical, 0.0 = orthogonal, -1.0 = opposite

    PYTHON CONCEPTS:
    - zip(): iterate two lists in parallel
    - sum() with generator expression
    - math.sqrt() for square root
    """
    # Dot product: multiply corresponding elements and sum
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitude (length) of each vector
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Cosine similarity formula
    return dot_product / (magnitude1 * magnitude2)

# Usage
query_emb = [0.1, 0.2, 0.3]
doc_emb = [0.15, 0.25, 0.28]
similarity = cosine_similarity(query_emb, doc_emb)  # ~0.999 (very similar)
```

### Batch Processing Embeddings

```python
def batch_embed_texts(
    texts: list[str],
    embed_fn: callable,
    batch_size: int = 100
) -> list[list[float]]:
    """
    Process texts in batches for efficiency.

    PYTHON CONCEPTS:
    - range() with step for batching
    - List slicing [start:end]
    - List concatenation with extend()
    """
    embeddings: list[list[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Slice: get next batch_size items
        batch_embeddings = embed_fn(batch)
        embeddings.extend(batch_embeddings)  # Add all at once

    return embeddings

# Example usage
texts = ["text1", "text2", ...] * 500  # 500 texts
embeddings = batch_embed_texts(texts, some_embed_function, batch_size=100)
# Processes in 5 batches of 100
```

---

## Generators for Memory Efficiency

### Why Generators?

```python
# BAD: Loads all embeddings into memory at once
def load_all_embeddings(file_path: str) -> list[list[float]]:
    embeddings = []
    with open(file_path) as f:
        for line in f:
            embedding = parse_embedding(line)
            embeddings.append(embedding)
    return embeddings  # Could be gigabytes!

# GOOD: Yields one embedding at a time
def stream_embeddings(file_path: str):
    """
    Generator function - uses 'yield' instead of 'return'.

    PYTHON: Generators produce values on-demand, not all at once.
    Memory usage stays constant regardless of file size.
    """
    with open(file_path) as f:
        for line in f:
            embedding = parse_embedding(line)
            yield embedding  # Pause here, resume when next value requested

# Usage
for embedding in stream_embeddings("embeddings.jsonl"):
    # Process one at a time - only one embedding in memory!
    similarity = cosine_similarity(query_emb, embedding)
```

### Generator Expressions

```python
# List comprehension: creates entire list in memory
squares = [x**2 for x in range(1_000_000)]  # 8+ MB memory

# Generator expression: computes on demand (note parentheses, not brackets)
squares_gen = (x**2 for x in range(1_000_000))  # ~128 bytes

# Use in aggregations
total = sum(x**2 for x in range(1_000_000))  # Memory efficient!

# Real RAG example: filter large result set
def find_relevant_chunks(
    query_embedding: list[float],
    chunk_embeddings: list[tuple[str, list[float]]],
    threshold: float = 0.7
):
    """
    PYTHON: Generator expression filters without creating intermediate list.
    """
    return (
        chunk_text
        for chunk_text, chunk_emb in chunk_embeddings
        if cosine_similarity(query_embedding, chunk_emb) > threshold
    )

# Only computes similarity when results are consumed
relevant = find_relevant_chunks(query_emb, all_chunks)
for chunk in relevant:  # Computed one at a time
    print(chunk)
```

---

## Async/Await for Concurrent Operations

### Why Async for RAG?

```python
# Typical RAG pipeline has many I/O operations:
# - Embedding API calls (network)
# - Vector database queries (network)
# - Document retrieval (disk/network)
# - LLM generation (network)

# These should run concurrently, not sequentially!
```

### Basic Async Patterns

```python
import asyncio
from typing import AsyncIterator

async def embed_text_async(text: str) -> list[float]:
    """
    Async embedding function (simulated).

    PYTHON: async def makes function return a coroutine
    Must be awaited or run with asyncio.run()
    """
    # Simulate API call
    await asyncio.sleep(0.1)  # Non-blocking wait
    return [0.1, 0.2, 0.3]  # Simplified embedding

# Wrong: This doesn't work!
# embedding = embed_text_async("hello")  # Returns coroutine object, not result

# Right: await the result
embedding = await embed_text_async("hello")  # Returns [0.1, 0.2, 0.3]
```

### Concurrent Embedding

```python
async def embed_documents_concurrent(docs: list[str]) -> list[list[float]]:
    """
    Embed multiple documents concurrently.

    PYTHON: asyncio.gather() runs multiple coroutines in parallel.
    Much faster than sequential for I/O-bound operations!
    """
    # Create all embedding tasks
    tasks = [embed_text_async(doc) for doc in docs]

    # Run all concurrently, wait for all to complete
    embeddings = await asyncio.gather(*tasks)

    return embeddings

# Usage
docs = ["Document 1", "Document 2", "Document 3"]
embeddings = await embed_documents_concurrent(docs)
# All 3 embedded in parallel - takes ~0.1s instead of ~0.3s!
```

### Async Generators for Streaming

```python
async def stream_search_results(
    query: str,
    vector_db: any,
    limit: int = 10
) -> AsyncIterator[dict]:
    """
    Stream search results as they're found.

    PYTHON: async def + yield = async generator
    Use 'async for' to consume results
    """
    query_emb = await embed_text_async(query)

    # Simulate progressive search
    for i in range(limit):
        await asyncio.sleep(0.05)  # Simulate search delay

        # Yield result immediately when found
        yield {
            "text": f"Result {i}",
            "score": 0.9 - (i * 0.05),
            "metadata": {"source": f"doc_{i}"}
        }

# Usage - process results as they arrive
async for result in stream_search_results("Python tutorials"):
    print(f"Found: {result['text']} (score: {result['score']})")
    # Can start processing immediately without waiting for all results!
```

---

## Text Chunking Strategies

### Character-based Chunking

```python
def chunk_by_characters(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[str]:
    """
    Split text into overlapping chunks.

    PYTHON CONCEPTS:
    - range() with step
    - String slicing
    - List building with append()
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        # Extract chunk
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap)
        start += (chunk_size - overlap)

    return chunks

# Usage
long_text = "..." * 10000
chunks = chunk_by_characters(long_text, chunk_size=1000, overlap=200)
```

### Sentence-based Chunking

```python
def chunk_by_sentences(
    text: str,
    max_sentences: int = 5
) -> list[str]:
    """
    Chunk by sentence boundaries for semantic coherence.

    PYTHON CONCEPTS:
    - String split with multiple delimiters (regex)
    - List slicing with step
    - String join
    """
    import re

    # Split on sentence boundaries (., !, ?)
    sentences = re.split(r'[.!?]+', text)

    # Clean up empty strings and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []

    # Group sentences into chunks
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        chunk = ". ".join(chunk_sentences) + "."
        chunks.append(chunk)

    return chunks
```

### Smart Chunking with Metadata

```python
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Chunk with metadata for better retrieval."""
    text: str
    chunk_index: int
    source_doc: str
    char_start: int
    char_end: int
    word_count: int

def chunk_with_metadata(
    text: str,
    source_id: str,
    chunk_size: int = 500
) -> list[DocumentChunk]:
    """
    Create chunks with metadata for tracking.

    PYTHON CONCEPTS:
    - Dataclass for structured data
    - Enumerate for index tracking
    - Multiple attributes per object
    """
    words = text.split()
    chunks: list[DocumentChunk] = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Calculate character positions
        char_start = len(" ".join(words[:i]))
        char_end = char_start + len(chunk_text)

        chunk = DocumentChunk(
            text=chunk_text,
            chunk_index=i // chunk_size,
            source_doc=source_id,
            char_start=char_start,
            char_end=char_end,
            word_count=len(chunk_words)
        )

        chunks.append(chunk)

    return chunks
```

---

## Sorting and Ranking Results

### Sorting by Similarity

```python
from dataclasses import dataclass

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict

def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    """
    Sort results by score (highest first).

    PYTHON CONCEPTS:
    - sorted() creates new sorted list
    - key parameter specifies sort criteria
    - lambda for inline function
    - reverse=True for descending order
    """
    return sorted(
        results,
        key=lambda r: r.score,  # Sort by score attribute
        reverse=True  # Highest scores first
    )

# Usage
results = [
    SearchResult("Doc A", 0.85, {}),
    SearchResult("Doc B", 0.92, {}),
    SearchResult("Doc C", 0.78, {})
]

ranked = rank_results(results)
# [SearchResult("Doc B", 0.92, ...), SearchResult("Doc A", 0.85, ...), ...]
```

### Top-K Selection

```python
import heapq

def get_top_k(
    items: list[tuple[str, float]],  # (text, score) pairs
    k: int = 10
) -> list[tuple[str, float]]:
    """
    Get top k items by score efficiently.

    PYTHON: heapq.nlargest() is more efficient than sorting when k << len(items)
    O(n log k) instead of O(n log n)
    """
    return heapq.nlargest(
        k,
        items,
        key=lambda x: x[1]  # Sort by score (second element)
    )

# Usage
all_results = [("doc1", 0.9), ("doc2", 0.7), ...] * 1000  # 1000 results
top_10 = get_top_k(all_results, k=10)  # Much faster than sorting all 1000
```

---

## Working with JSONL (JSON Lines)

### Reading JSONL Files

```python
import json

def read_jsonl(file_path: str) -> list[dict]:
    """
    Read JSONL file (one JSON object per line).

    PYTHON CONCEPTS:
    - Context manager (with statement)
    - List comprehension with file iteration
    - json.loads() for each line
    """
    with open(file_path) as f:
        return [json.loads(line) for line in f]

# Memory-efficient version with generator
def stream_jsonl(file_path: str):
    """Generator version - doesn't load entire file into memory."""
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)

# Usage
for doc in stream_jsonl("documents.jsonl"):
    # Process one document at a time
    embedding = embed_document(doc)
```

### Writing JSONL Files

```python
def write_jsonl(data: list[dict], file_path: str) -> None:
    """
    Write list of dicts to JSONL file.

    PYTHON CONCEPTS:
    - File write mode "w"
    - json.dumps() for each object
    - f-string with \n for newlines
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Usage
documents = [
    {"text": "Doc 1", "embedding": [0.1, 0.2, 0.3]},
    {"text": "Doc 2", "embedding": [0.4, 0.5, 0.6]}
]
write_jsonl(documents, "embeddings.jsonl")
```

---

## Caching for Performance

### Simple Dict Cache

```python
from typing import Callable

def cache_embeddings(embed_fn: Callable[[str], list[float]]):
    """
    Decorator to cache embedding function results.

    PYTHON CONCEPTS:
    - Closures (nested function accessing outer scope)
    - Decorators (function that wraps another function)
    - Dict as cache (hash table lookup)
    """
    cache: dict[str, list[float]] = {}

    def wrapped(text: str) -> list[float]:
        # Check cache first
        if text in cache:
            return cache[text]

        # Not cached - compute and store
        embedding = embed_fn(text)
        cache[text] = embedding
        return embedding

    return wrapped

# Usage
@cache_embeddings
def my_embed_function(text: str) -> list[float]:
    # Expensive operation
    return call_embedding_api(text)

# First call: hits API
emb1 = my_embed_function("hello")

# Second call: returns cached result (instant!)
emb2 = my_embed_function("hello")
```

### LRU Cache (Limited Size)

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used cache with size limit.

    PYTHON CONCEPTS:
    - Class with __init__ and methods
    - OrderedDict maintains insertion order
    - popitem() removes oldest item
    """
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, list[float]] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> list[float] | None:
        """Get from cache, return None if not found."""
        if key not in self.cache:
            return None

        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: list[float]) -> None:
        """Add to cache, evict oldest if full."""
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict oldest if over limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

# Usage
cache = LRUCache(max_size=100)

embedding = cache.get("query1")
if embedding is None:
    embedding = expensive_embed("query1")
    cache.put("query1", embedding)
```

---

## Next Steps

You now have intermediate Python skills for RAG systems! Continue to:

1. Read [concepts.md](concepts.md) and [architecture.md](architecture.md) for RAG theory
2. Study [examples.py](examples.py) to see these patterns in action
3. Move on to `04-agents-and-tool-use/` for advanced async patterns

**Key takeaways:**
- Embeddings are just lists of floats
- Use generators for memory efficiency
- Use async for concurrent I/O operations
- Chunk strategically with metadata
- Cache expensive operations

---

## Quick Reference

| Task | Python Pattern |
|---|---|
| Vector similarity | `sum(a*b for a,b in zip(v1,v2))` |
| Batch processing | `for i in range(0, len(items), batch):` |
| Generator | `def f(): yield x` (not return) |
| Generator expression | `(x*2 for x in items)` |
| Async function | `async def f(): await ...` |
| Concurrent calls | `await asyncio.gather(*tasks)` |
| Async generator | `async def f(): yield x` |
| Sort by score | `sorted(items, key=lambda x: x.score, reverse=True)` |
| Top K items | `heapq.nlargest(k, items, key=...)` |
| JSONL read | `[json.loads(line) for line in f]` |
| Simple cache | `if key in cache: return cache[key]` |

**Next:** [examples.py](examples.py) for complete RAG pipeline implementations
