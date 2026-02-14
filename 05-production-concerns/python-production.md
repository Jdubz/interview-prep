# Python for Production LLM Systems

> **Production-ready Python patterns for reliable LLM applications.** This guide covers logging, testing, monitoring, type checking, and performance optimization.

---

## Logging

### Structured Logging

```python
import logging
import json
from typing import Any
from datetime import datetime

# Configure logging (do this once at app startup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Structured logging for better parsing
def log_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    success: bool,
    error: str | None = None
) -> None:
    """
    Log LLM API calls with structured data.

    PYTHON: logger.info() with extra dict for structured data
    JSON format is parsable by log aggregation tools
    """
    log_data = {
        "event": "llm_call",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": latency_ms,
        "success": success,
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }

    if success:
        logger.info(json.dumps(log_data))
    else:
        logger.error(json.dumps(log_data))

# Usage
log_llm_call(
    model="claude-sonnet-4",
    prompt_tokens=100,
    completion_tokens=50,
    latency_ms=234.5,
    success=True
)
```

### Context-aware Logging

```python
from contextvars import ContextVar
from uuid import uuid4

# Context variable for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

class RequestContext:
    """
    Attach request ID to all logs in async context.

    PYTHON CONCEPTS:
    - ContextVar for async-safe context
    - Context manager (__enter__/__exit__)
    - UUID for unique IDs
    """
    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or str(uuid4())
        self.token = None

    def __enter__(self):
        self.token = request_id_var.set(self.request_id)
        return self

    def __exit__(self, *args):
        request_id_var.reset(self.token)

# Custom logger that includes request ID
class ContextLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _add_context(self, msg: str) -> str:
        request_id = request_id_var.get('')
        if request_id:
            return f"[req:{request_id[:8]}] {msg}"
        return msg

    def info(self, msg: str, **kwargs):
        self.logger.info(self._add_context(msg), **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(self._add_context(msg), **kwargs)

# Usage
logger = ContextLogger(logging.getLogger(__name__))

async def handle_request(user_query: str):
    with RequestContext():  # Generates new request ID
        logger.info("Processing user query")
        # All logs in this context will have same request ID

        result = await call_llm(user_query)
        logger.info("LLM call completed")

        return result

# Logs will look like:
# [req:a1b2c3d4] Processing user query
# [req:a1b2c3d4] LLM call completed
```

---

## Testing

### Unit Tests for LLM Functions

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

# Function to test
async def classify_sentiment(
    text: str,
    llm_client: any
) -> str:
    """Classify text sentiment using LLM."""
    prompt = f"Classify sentiment as positive/negative/neutral: {text}"
    response = await llm_client.complete(prompt)
    return response.strip().lower()

# Test with mock
@pytest.mark.asyncio
async def test_classify_sentiment():
    """
    Test sentiment classification with mocked LLM.

    PYTHON TESTING:
    - pytest for test framework
    - AsyncMock for async functions
    - assert for verification
    """
    # Create mock LLM client
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "positive"

    # Test the function
    result = await classify_sentiment("I love this!", mock_llm)

    # Verify result
    assert result == "positive"

    # Verify LLM was called with correct prompt
    mock_llm.complete.assert_called_once()
    call_args = mock_llm.complete.call_args[0][0]
    assert "I love this!" in call_args
```

### Testing with Fixtures

```python
import pytest

@pytest.fixture
def sample_documents():
    """
    Fixture providing test data.

    PYTHON: @pytest.fixture decorator
    Function runs once, result is reused across tests
    """
    return [
        {"id": 1, "text": "Python is great"},
        {"id": 2, "text": "Testing is important"},
        {"id": 3, "text": "LLMs are powerful"}
    ]

@pytest.fixture
async def mock_embedding_model():
    """Async fixture for embedding model."""
    model = AsyncMock()
    model.embed.return_value = [0.1, 0.2, 0.3]  # Fixed embedding
    return model

# Use fixtures in tests
@pytest.mark.asyncio
async def test_document_embedding(sample_documents, mock_embedding_model):
    """Test document embedding with fixtures."""
    doc = sample_documents[0]

    embedding = await mock_embedding_model.embed(doc["text"])

    assert len(embedding) == 3
    assert embedding == [0.1, 0.2, 0.3]
```

### Parameterized Tests

```python
@pytest.mark.parametrize(
    "text,expected_sentiment",
    [
        ("I love this product!", "positive"),
        ("This is terrible", "negative"),
        ("It's okay", "neutral"),
        ("Amazing experience!", "positive"),
    ]
)
@pytest.mark.asyncio
async def test_sentiment_classification(text, expected_sentiment):
    """
    Run same test with different inputs.

    PYTHON: @pytest.mark.parametrize runs test multiple times
    Each tuple becomes a separate test case
    """
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = expected_sentiment

    result = await classify_sentiment(text, mock_llm)

    assert result == expected_sentiment
```

---

## Type Checking

### Running MyPy

```python
# mypy is a static type checker for Python
# Run with: mypy your_file.py

from typing import Any

# Good: Clear type hints
def process_llm_response(response: dict[str, Any]) -> str:
    """MyPy can verify this is used correctly."""
    return response["content"]

# MyPy will catch errors
result: int = process_llm_response({})  # Error: str not assignable to int

# Type narrowing with isinstance
def safe_process(value: str | int) -> str:
    """
    Type narrowing - MyPy understands isinstance checks.

    After isinstance check, MyPy knows exact type
    """
    if isinstance(value, str):
        # MyPy knows value is str here
        return value.upper()
    else:
        # MyPy knows value is int here
        return str(value)
```

### TypedDict for Structured Data

```python
from typing import TypedDict, NotRequired

class LLMResponse(TypedDict):
    """
    Typed dictionary - like dataclass but for dicts.

    PYTHON: TypedDict provides type safety for dict access
    MyPy can verify all required keys are present
    """
    content: str
    model: str
    tokens: int
    finish_reason: str
    metadata: NotRequired[dict[str, Any]]  # Optional key (3.11+)

# Type-safe usage
def parse_response(raw: dict) -> LLMResponse:
    """MyPy verifies this returns correct structure."""
    return {
        "content": raw["content"],
        "model": raw["model"],
        "tokens": raw["usage"]["total_tokens"],
        "finish_reason": raw["finish_reason"]
        # metadata is optional, can omit
    }

# MyPy catches missing keys
def broken_parse(raw: dict) -> LLMResponse:
    return {
        "content": raw["content"]
        # Error: Missing required keys!
    }
```

---

## Performance Monitoring

### Timing Decorator

```python
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar('T')

def async_timer(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time async function execution.

    PYTHON:
    - time.perf_counter() for high-resolution timing
    - @wraps preserves function metadata
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            logger.info(f"{func.__name__} took {elapsed:.2f}ms")

    return wrapper

# Usage
@async_timer
async def expensive_llm_call(prompt: str) -> str:
    response = await llm.complete(prompt)
    return response

# Automatically logs: expensive_llm_call took 234.56ms
```

### Memory Profiling

```python
import tracemalloc
from contextlib import contextmanager

@contextmanager
def memory_profiler(description: str):
    """
    Context manager to profile memory usage.

    PYTHON:
    - tracemalloc for memory tracking
    - Context manager for automatic cleanup
    """
    # Start tracking
    tracemalloc.start()

    try:
        yield
    finally:
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{description}:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

# Usage
with memory_profiler("Loading embeddings"):
    embeddings = load_large_embedding_file()

# Output:
# Loading embeddings:
#   Current: 512.34 MB
#   Peak: 756.12 MB
```

### Metrics Collection

```python
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

@dataclass
class MetricsCollector:
    """
    Collect application metrics for monitoring.

    PYTHON:
    - defaultdict for auto-initialization
    - field(default_factory) for mutable defaults
    """
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    metadata: dict[str, Any] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment counter."""
        self.counters[name] += amount

    def record_timing(self, name: str, ms: float) -> None:
        """Record timing in milliseconds."""
        self.timings[name].append(ms)

    def get_stats(self) -> dict[str, Any]:
        """Calculate statistics."""
        import statistics

        timing_stats = {}
        for name, values in self.timings.items():
            if values:
                timing_stats[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values)
                }

        return {
            "counters": dict(self.counters),
            "timings": timing_stats,
            "metadata": self.metadata
        }

# Usage
metrics = MetricsCollector()

# Track operations
metrics.increment("llm_calls")
metrics.increment("tokens_used", amount=150)
metrics.record_timing("llm_latency", 234.5)

# Get summary
stats = metrics.get_stats()
# {
#   "counters": {"llm_calls": 1, "tokens_used": 150},
#   "timings": {"llm_latency": {"count": 1, "mean": 234.5, ...}},
#   ...
# }
```

---

## Environment Configuration

### Environment Variables

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    """
    Application configuration from environment.

    PYTHON:
    - os.getenv() for environment variables
    - Default values for missing vars
    - Type conversion (str -> int, str -> bool)
    """
    # API keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Model config
    default_model: str = os.getenv("DEFAULT_MODEL", "claude-sonnet-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Feature flags
    enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "false").lower() == "true"

    # Limits
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    timeout_seconds: float = float(os.getenv("TIMEOUT_SECONDS", "30.0"))

    def validate(self) -> None:
        """Validate configuration."""
        if not self.anthropic_api_key and not self.openai_api_key:
            raise ValueError("No API keys configured!")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be between 0 and 1")

# Usage
config = Config()
config.validate()

# Access config
print(f"Using model: {config.default_model}")
```

---

## Error Handling in Production

### Custom Exceptions

```python
class LLMError(Exception):
    """Base exception for LLM operations."""
    pass

class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")

class InvalidResponseError(LLMError):
    """Raised when LLM response is invalid."""
    def __init__(self, response: str, expected: str):
        self.response = response
        self.expected = expected
        super().__init__(f"Expected {expected}, got: {response[:100]}")

class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    def __init__(self, tokens_used: int, limit: int):
        self.tokens_used = tokens_used
        self.limit = limit
        super().__init__(f"Token limit exceeded: {tokens_used}/{limit}")

# Usage with specific handling
async def robust_llm_call(prompt: str) -> str:
    """Handle all LLM errors appropriately."""
    try:
        return await call_llm(prompt)

    except RateLimitError as e:
        # Wait and retry
        logger.warning(f"Rate limited, waiting {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
        return await call_llm(prompt)

    except TokenLimitError as e:
        # Truncate prompt and retry
        logger.warning(f"Token limit exceeded, truncating")
        shorter_prompt = truncate_to_tokens(prompt, e.limit // 2)
        return await call_llm(shorter_prompt)

    except InvalidResponseError as e:
        # Log and raise
        logger.error(f"Invalid response: {e}")
        raise

    except LLMError as e:
        # Catch-all for other LLM errors
        logger.error(f"LLM error: {e}")
        raise
```

---

## Async Best Practices

### Proper Resource Cleanup

```python
import aiohttp
from contextlib import asynccontextmanager
from typing import AsyncIterator

class LLMClient:
    """
    Production LLM client with proper resource management.

    PYTHON:
    - __aenter__/__aexit__ for async context manager
    - aiohttp.ClientSession for connection pooling
    - Guaranteed cleanup even on exceptions
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Initialize session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Clean up session."""
        if self.session:
            await self.session.close()

    async def complete(self, prompt: str) -> str:
        """Call LLM API."""
        if not self.session:
            raise RuntimeError("Client not initialized")

        # Use persistent session (connection pooling)
        async with self.session.post(
            "https://api.anthropic.com/v1/complete",
            headers={"x-api-key": self.api_key},
            json={"prompt": prompt}
        ) as response:
            data = await response.json()
            return data["completion"]

# Usage - session is automatically managed
async with LLMClient(api_key="...") as client:
    result1 = await client.complete("Hello")
    result2 = await client.complete("World")
    # Session reused for both calls (faster!)
# Session automatically closed here, even if exception occurred
```

---

## Validation

### Input Validation with Pydantic

```python
from pydantic import BaseModel, Field, validator

class LLMRequest(BaseModel):
    """
    Validate LLM requests with Pydantic.

    PYTHON: Pydantic provides automatic validation
    - Type checking
    - Range validation
    - Custom validators
    """
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str = Field(..., pattern="^(gpt-4|claude-)")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0, le=100000)

    @validator('prompt')
    def prompt_not_empty_after_strip(cls, v):
        """Custom validator."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()

# Usage - automatic validation
try:
    request = LLMRequest(
        prompt="Hello",
        model="claude-sonnet-4",
        temperature=1.5  # Error: must be <= 1.0
    )
except ValueError as e:
    print(f"Validation error: {e}")

# Valid request
request = LLMRequest(
    prompt="What is Python?",
    model="claude-sonnet-4"
)
print(request.dict())  # Convert to dict
print(request.json())  # Convert to JSON
```

---

## Next Steps

You now have production-ready Python skills! Use these patterns for:

1. **Logging**: Track all LLM calls, errors, and performance
2. **Testing**: Write comprehensive tests with mocks and fixtures
3. **Monitoring**: Collect metrics and profile performance
4. **Configuration**: Manage environment variables safely
5. **Validation**: Ensure inputs are valid before processing

Continue to:
- Read [guide.md](guide.md) for production architecture patterns
- Study [examples.py](examples.py) for real implementations
- Review `06-interview-prep/` for interview questions on these topics

---

## Quick Reference

| Task | Python Pattern |
|---|---|
| Structured logging | `logger.info(json.dumps({...}))` |
| Request context | `ContextVar` for async-safe state |
| Unit tests | `@pytest.mark.asyncio + AsyncMock` |
| Fixtures | `@pytest.fixture` decorator |
| Parameterized tests | `@pytest.mark.parametrize` |
| Type checking | `mypy your_file.py` |
| TypedDict | `class MyDict(TypedDict): ...` |
| Timing | `time.perf_counter()` decorator |
| Memory profiling | `tracemalloc` context manager |
| Metrics | Custom MetricsCollector class |
| Environment vars | `os.getenv("VAR", "default")` |
| Custom exceptions | `class MyError(Exception):` |
| Async cleanup | `async with` context manager |
| Input validation | Pydantic BaseModel |

**Next:** [guide.md](guide.md) for full production architecture patterns
