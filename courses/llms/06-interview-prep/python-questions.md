# Python Interview Questions for LLM Engineers

> **Python questions specific to LLM/AI engineering roles.** Practice explaining these concepts clearly and concisely.

---

## Language Fundamentals

### Q: Explain the difference between `list` and `tuple` in Python. When would you use each?

**Answer:**
- **List**: Mutable, dynamic array. Use for collections that change (appending, removing items)
- **Tuple**: Immutable, fixed-size. Use for:
  - Fixed data structures (coordinates, RGB values)
  - Dictionary keys (must be immutable)
  - Function returns with multiple values
  - Performance (tuples are faster, use less memory)

**LLM Context:** Use tuples for (query, embedding) pairs in caching, lists for growing collections like message histories.

---

### Q: What's the difference between `@classmethod`, `@staticmethod`, and instance methods?

**Answer:**
- **Instance method**: Takes `self`, operates on instance data
- **`@classmethod`**: Takes `cls`, can access/modify class state, used for alternative constructors
- **`@staticmethod`**: Takes neither, utility function logically belongs to class but doesn't need class/instance data

```python
class LLMProvider:
    default_model = "claude"

    def __init__(self, api_key: str):
        self.api_key = api_key  # Instance

    @classmethod
    def from_env(cls):  # Alternative constructor
        return cls(os.getenv("API_KEY"))

    @staticmethod
    def estimate_tokens(text: str):  # Utility
        return len(text) // 4
```

---

### Q: Explain Python's GIL (Global Interpreter Lock) and its implications.

**Answer:**
The GIL prevents multiple threads from executing Python bytecode simultaneously.

**Implications:**
- **CPU-bound work**: Multi-threading doesn't help (use `multiprocessing` instead)
- **I/O-bound work**: Threads work well (GIL released during I/O)
- **Async/await**: Best for I/O-bound work like LLM API calls

**LLM Context:** Use `async/await` for concurrent LLM API calls, not threads. The GIL doesn't block async I/O.

---

## Async Programming

### Q: Explain `async`/`await` in Python. How does it differ from threading?

**Answer:**
- **Async/await**: Cooperative multitasking. Single thread, explicit yield points (`await`)
- **Threading**: Preemptive multitasking. OS switches threads, harder to reason about

**Async advantages:**
- No GIL issues for I/O
- Lower overhead than threads
- Easier to reason about (explicit await points)
- Perfect for I/O-bound work (API calls, database queries)

**When to use:** LLM API calls, embeddings, database queries—anything that waits for network/disk.

```python
# Sequential: 3 seconds total
r1 = await call_llm("prompt1")  # 1s
r2 = await call_llm("prompt2")  # 1s
r3 = await call_llm("prompt3")  # 1s

# Concurrent: 1 second total
results = await asyncio.gather(
    call_llm("prompt1"),
    call_llm("prompt2"),
    call_llm("prompt3")
)
```

---

### Q: What's the difference between `asyncio.gather()` and `asyncio.wait()`?

**Answer:**
- **`gather()`**:
  - Returns results in order
  - Raises first exception immediately
  - Simpler API, use for most cases

- **`wait()`**:
  - Returns (done, pending) sets
  - More control (FIRST_COMPLETED, ALL_COMPLETED)
  - Must extract results manually
  - Use for racing or partial results

```python
# gather - simple, ordered results
results = await asyncio.gather(call1(), call2(), call3())

# wait - first to complete
done, pending = await asyncio.wait(
    [call1(), call2()],
    return_when=asyncio.FIRST_COMPLETED
)
```

---

### Q: How do you handle errors in concurrent async operations?

**Answer:**
```python
# gather with return_exceptions=True
results = await asyncio.gather(
    call1(),
    call2(),
    call3(),
    return_exceptions=True  # Don't fail fast
)

# Check each result
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Call {i} failed: {result}")
    else:
        process(result)

# Or use try/except per task
async def safe_call(func):
    try:
        return await func()
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
```

---

## Type System

### Q: Explain Protocols vs ABCs (Abstract Base Classes). When to use each?

**Answer:**
- **Protocol** (structural typing): Duck typing with type checking. No inheritance needed.
- **ABC** (nominal typing): Explicit inheritance required.

**Use Protocol when:**
- You don't control the implementations
- You want flexibility
- LLM provider interfaces (works with any SDK)

**Use ABC when:**
- You need enforcement (can't forget to implement methods)
- Shared implementation logic
- Clear inheritance hierarchy

```python
# Protocol - flexible
class LLMProvider(Protocol):
    async def complete(self, prompt: str) -> str: ...

# Any class with this method works!
class CustomLLM:  # No inheritance needed
    async def complete(self, prompt: str) -> str:
        return "response"
```

---

### Q: What's `TypeVar` and when do you use it?

**Answer:**
`TypeVar` creates generic type variables for functions/classes that work with any type while preserving type safety.

```python
from typing import TypeVar

T = TypeVar('T')

def first_element(items: list[T]) -> T:
    """Returns same type as input list."""
    return items[0]

# Type checker knows:
x: int = first_element([1, 2, 3])      # x is int
y: str = first_element(["a", "b"])     # y is str

# Real example: retry wrapper
async def with_retry(func: Callable[[], T]) -> T:
    """Retry logic preserving return type."""
    for attempt in range(3):
        try:
            return await func()  # Returns T
        except Exception:
            if attempt == 2: raise
            await asyncio.sleep(2 ** attempt)
```

---

## Data Structures

### Q: Explain `collections.defaultdict`. When is it useful for LLM work?

**Answer:**
`defaultdict` automatically creates missing keys with a default value.

**Without defaultdict:**
```python
# Manual initialization
counts = {}
for token in tokens:
    if token not in counts:
        counts[token] = 0
    counts[token] += 1
```

**With defaultdict:**
```python
from collections import defaultdict

counts = defaultdict(int)  # Auto-creates 0 for missing keys
for token in tokens:
    counts[token] += 1  # Just increment!

# Group by category
by_category = defaultdict(list)
for doc in documents:
    by_category[doc.category].append(doc)
```

**LLM use cases:** Counting tokens, grouping documents, aggregating metrics.

---

### Q: Explain the difference between `deepcopy` and `copy`. When does it matter?

**Answer:**
- **Shallow copy (`copy.copy`)**: Copies object, but not nested objects (references)
- **Deep copy (`copy.deepcopy`)**: Recursively copies everything

```python
import copy

original = {"messages": [{"role": "user", "content": "hi"}]}

shallow = copy.copy(original)
deep = copy.deepcopy(original)

# Modify nested object
original["messages"][0]["content"] = "changed"

print(shallow["messages"][0]["content"])  # "changed" - shared reference!
print(deep["messages"][0]["content"])     # "hi" - independent copy
```

**When it matters:** Modifying conversation histories, cached prompts, configuration objects.

---

## Functional Programming

### Q: Explain list comprehensions vs `map`/`filter`. Which is more Pythonic?

**Answer:**
Both work, but **list comprehensions are more Pythonic** (preferred in Python).

```python
# List comprehension (preferred)
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# map/filter (functional style)
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

**Advantages of comprehensions:**
- More readable
- Faster
- Can combine map + filter
- Support nested loops

```python
# Extract user messages
user_msgs = [
    msg["content"]
    for msg in messages
    if msg["role"] == "user"
]
```

---

### Q: What's a generator? When would you use one instead of a list?

**Answer:**
Generator produces values on-demand (lazy evaluation) instead of creating entire list.

**Use generators when:**
- Large/infinite sequences
- You don't need all items at once
- Memory is limited

```python
# List - loads everything into memory
def load_documents_list(file_path):
    docs = []
    for line in open(file_path):
        docs.append(parse(line))
    return docs  # Could be gigabytes!

# Generator - one at a time
def load_documents_gen(file_path):
    for line in open(file_path):
        yield parse(line)  # Memory-efficient

# Usage
for doc in load_documents_gen("big_file.jsonl"):
    process(doc)  # Only one doc in memory at a time
```

---

## Error Handling

### Q: What's the difference between `raise`, `raise Exception`, and `raise Exception from e`?

**Answer:**
- **`raise`**: Re-raise current exception (in except block)
- **`raise Exception("msg")`**: Raise new exception, loses original context
- **`raise Exception("msg") from e`**: Chain exceptions, preserves context

```python
try:
    response = api.call(prompt)
except ConnectionError as e:
    # Bad - loses original error
    raise RuntimeError("API failed")

    # Good - preserves chain
    raise RuntimeError("API failed") from e

    # Good - re-raise original
    logger.error(f"API error: {e}")
    raise
```

**Why it matters:** Debugging production LLM issues requires full error context.

---

### Q: Explain `finally` vs `else` in try/except blocks.

**Answer:**
- **`finally`**: Always runs (cleanup code)
- **`else`**: Runs only if NO exception occurred

```python
try:
    result = await call_llm(prompt)
except TimeoutError:
    result = "timeout"
except Exception as e:
    logger.error(f"Failed: {e}")
    raise
else:
    # Only runs if no exception
    logger.info("Success!")
    cache.store(prompt, result)
finally:
    # Always runs (cleanup)
    metrics.increment("llm_calls")
```

---

## Memory & Performance

### Q: How do you profile Python code for performance issues?

**Answer:**
**Time profiling:**
```python
import cProfile

cProfile.run('my_function()')

# Or with decorator
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.4f}s")
        return result
    return wrapper
```

**Memory profiling:**
```python
import tracemalloc

tracemalloc.start()
result = expensive_operation()
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
```

**For LLM apps, profile:**
- API call latency (network)
- Embedding computation (CPU)
- Vector search (memory)
- Prompt building (string operations)

---

### Q: How can you reduce memory usage when processing large datasets?

**Answer:**
1. **Use generators** instead of lists
2. **Process in batches** (don't load all at once)
3. **Use itertools** for memory-efficient operations
4. **Stream from disk** instead of loading everything
5. **Use appropriate data structures** (sets for membership, dicts for lookups)

```python
# Bad - loads everything
def process_all(file_path):
    with open(file_path) as f:
        data = f.read()  # Entire file in memory
        return process(data)

# Good - streaming
def process_streaming(file_path):
    with open(file_path) as f:
        for line in f:  # One line at a time
            yield process(line)

# Batch processing
from itertools import islice

def process_batches(items, batch_size=100):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield process_batch(batch)
```

---

## Dataclasses & Type Hints

### Q: What are the advantages of `dataclasses` over regular classes?

**Answer:**
**Advantages:**
- Auto-generates `__init__`, `__repr__`, `__eq__`
- Less boilerplate
- Type hints built-in
- Immutable option (`frozen=True`)
- Conversion to dict with `asdict()`

```python
# Without dataclass
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"

    def __eq__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return self.role == other.role and self.content == other.content

# With dataclass
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

# All the same functionality, 3 lines instead of 13!
```

---

### Q: Explain `field(default_factory=...)` in dataclasses.

**Answer:**
**Problem:** Mutable defaults (lists, dicts) are dangerous in Python.

```python
# WRONG - shared between instances!
@dataclass
class Config:
    options: list = []  # BUG: all instances share same list!

c1 = Config()
c2 = Config()
c1.options.append("x")
print(c2.options)  # ["x"] - WRONG!

# RIGHT - each instance gets new list
from dataclasses import field

@dataclass
class Config:
    options: list = field(default_factory=list)

c1 = Config()
c2 = Config()
c1.options.append("x")
print(c2.options)  # [] - Correct!
```

**Use for:** Lists, dicts, sets, or any mutable default value.

---

## Testing

### Q: How do you test async functions in Python?

**Answer:**
Use `pytest` with `pytest-asyncio` plugin.

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_llm_call():
    """Test async LLM function."""
    # Mock the async API
    mock_client = AsyncMock()
    mock_client.complete.return_value = "test response"

    # Test the function
    result = await my_llm_function("prompt", mock_client)

    # Assertions
    assert result == "test response"
    mock_client.complete.assert_called_once_with("prompt")

# Fixture for reusable mocks
@pytest.fixture
async def mock_llm():
    llm = AsyncMock()
    llm.complete.return_value = "mocked"
    return llm

@pytest.mark.asyncio
async def test_with_fixture(mock_llm):
    result = await my_function(mock_llm)
    assert result == "mocked"
```

---

### Q: What's the purpose of `@pytest.mark.parametrize`?

**Answer:**
Run same test with multiple inputs without duplicating code.

```python
@pytest.mark.parametrize("input,expected", [
    ("positive text", "positive"),
    ("negative text", "negative"),
    ("neutral text", "neutral"),
])
async def test_sentiment(input, expected):
    """Runs 3 times with different inputs."""
    result = await classify_sentiment(input)
    assert result == expected

# Useful for: edge cases, multiple examples, regression tests
```

---

## Best Practices

### Q: How do you structure a Python LLM project for production?

**Answer:**
```
project/
├── src/
│   ├── __init__.py
│   ├── config.py          # Environment config
│   ├── models/            # LLM interfaces
│   │   ├── __init__.py
│   │   ├── base.py        # Protocol/ABC
│   │   └── providers.py   # OpenAI, Anthropic, etc.
│   ├── prompts/           # Prompt templates
│   │   └── templates.py
│   ├── tools/             # Agent tools
│   │   ├── __init__.py
│   │   └── calculator.py
│   └── utils/             # Helpers
│       ├── logging.py
│       ├── metrics.py
│       └── retry.py
├── tests/
│   ├── unit/              # Fast, isolated
│   ├── integration/       # With real APIs
│   └── fixtures/          # Test data
├── .env                   # Environment vars
├── pyproject.toml         # Dependencies
├── mypy.ini               # Type checking config
└── pytest.ini             # Test config
```

**Key principles:**
- Type hints everywhere
- Protocols for interfaces
- Environment-based config
- Comprehensive tests
- Structured logging

---

## Coding Exercise

### Q: Implement a caching decorator for async LLM calls with TTL (time-to-live).

**Answer:**
```python
import asyncio
import time
from functools import wraps
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def async_cache_with_ttl(ttl_seconds: float):
    """Decorator to cache async function results with expiration."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[str, tuple[Any, float]] = {}  # key -> (result, timestamp)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create cache key from arguments
            key = f"{func.__name__}:{args}:{kwargs}"

            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    print(f"Cache hit for {key}")
                    return result
                else:
                    print(f"Cache expired for {key}")

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())

            return result

        return wrapper
    return decorator

# Usage
@async_cache_with_ttl(ttl_seconds=60)
async def call_llm(prompt: str) -> str:
    print("Calling LLM API...")
    await asyncio.sleep(1)  # Simulate API call
    return f"Response to: {prompt}"

# First call - hits API
result1 = await call_llm("hello")  # Prints: Calling LLM API...

# Second call within 60s - cached
result2 = await call_llm("hello")  # Prints: Cache hit...

# After 60s - expired, hits API again
```

---

## Quick Tips for Interviews

1. **Start simple, then optimize**: Don't jump to complex solutions
2. **Think aloud**: Explain your reasoning as you code
3. **Consider edge cases**: Empty inputs, None values, errors
4. **Ask clarifying questions**: Requirements, scale, constraints
5. **Know time/space complexity**: For data structures and algorithms
6. **Use type hints**: Shows you write production-quality code
7. **Handle errors**: Always think about what can go wrong
8. **Test your code**: Walk through with example inputs

---

**Practice these topics:**
- Async patterns (very common in LLM interviews)
- Type system (Protocols, generics)
- Error handling and retries
- Testing strategies
- Performance optimization

**Next:** Practice coding these patterns and review [llm-questions.md](llm-questions.md) for LLM-specific questions.
