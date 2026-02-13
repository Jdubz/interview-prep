"""
Production Concerns Examples

Python patterns for streaming, token counting, structured output
validation, caching, and retry logic. Provider-agnostic where possible.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Literal, Callable, Awaitable


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMConfig:
    model: str
    temperature: float = 1.0
    max_tokens: int | None = None
    response_format: Literal["json_object", "text"] | None = None


class CompletionFn(Protocol):
    async def __call__(self, messages: list[Message], config: LLMConfig) -> str: ...


# ---------------------------------------------------------------------------
# Example 1: Streaming Handler
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    text: str
    done: bool
    usage: dict[str, int] | None = None  # {"input_tokens": ..., "output_tokens": ...}


class StreamCompletionFn(Protocol):
    """A streaming completion function — yields chunks as they arrive."""
    def __call__(
        self, messages: list[Message], config: LLMConfig
    ) -> AsyncIterator[StreamChunk]: ...


async def handle_stream(
    stream: AsyncIterator[StreamChunk],
    *,
    on_token: Callable[[str], None] | None = None,
    on_complete: Callable[[str, dict[str, int] | None], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> str:
    """
    Process a streaming response with callbacks for real-time UI updates.
    Handles buffering, error recovery, and aggregation.
    """
    full_text = ""
    usage = None

    try:
        async for chunk in stream:
            full_text += chunk.text
            if on_token:
                on_token(chunk.text)
            if chunk.done:
                usage = chunk.usage

        if on_complete:
            on_complete(full_text, usage)
        return full_text

    except Exception as e:
        if on_error:
            on_error(e)
        if full_text:
            return full_text + "\n\n[Response interrupted]"
        raise


async def stream_to_sse(
    stream: AsyncIterator[StreamChunk],
    write: Callable[[str], None],
    close: Callable[[], None],
) -> None:
    """
    Stream to a Server-Sent Events response (e.g., in a FastAPI/Flask handler).
    This is the pattern for server → browser streaming.
    """
    try:
        async for chunk in stream:
            data = json.dumps({"text": chunk.text, "done": chunk.done})
            write(f"data: {data}\n\n")
            if chunk.done:
                write("data: [DONE]\n\n")
    finally:
        close()


# ---------------------------------------------------------------------------
# Example 2: Token Counting & Cost Estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Approximate token count for text.

    For precise counts, use a tokenizer (tiktoken for OpenAI, etc.).
    This approximation is useful for cost estimation and context window checks.
    """
    # Rule of thumb: ~4 characters per token for English
    return -(-len(text) // 4)  # Ceiling division


def estimate_message_tokens(messages: list[Message]) -> int:
    """Token counting for a full message array."""
    tokens = 0
    for msg in messages:
        tokens += 4  # ~4 tokens overhead per message (role, delimiters)
        tokens += estimate_tokens(msg.content)
    tokens += 3  # Priming tokens
    return tokens


@dataclass
class ModelPricing:
    """Cost in USD per 1M tokens."""
    input_per_million: float
    output_per_million: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.5, 10),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "claude-sonnet-4-5-20250929": ModelPricing(3, 15),
    "claude-haiku-4-5-20251001": ModelPricing(0.8, 4),
}


@dataclass
class CostEstimate:
    input_cost: float
    output_cost: float
    total_cost: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return CostEstimate(0, 0, 0)

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return CostEstimate(input_cost, output_cost, input_cost + output_cost)


@dataclass
class ContextFit:
    fits: bool
    estimated_tokens: int
    available: int


def check_context_fit(
    messages: list[Message],
    context_window: int,
    reserved_for_output: int = 4096,
) -> ContextFit:
    """Check if messages fit within a model's context window."""
    estimated = estimate_message_tokens(messages)
    available = context_window - reserved_for_output
    return ContextFit(
        fits=estimated <= available,
        estimated_tokens=estimated,
        available=available,
    )


# ---------------------------------------------------------------------------
# Example 3: Structured Output Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when LLM output fails validation."""
    pass


@dataclass
class SentimentResult:
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str


def validate_sentiment(data: Any) -> SentimentResult:
    """Validate a sentiment analysis response. Raises ValidationError on failure."""
    if not isinstance(data, dict):
        raise ValidationError("Expected a JSON object")

    sentiment = data.get("sentiment")
    if sentiment not in ("positive", "negative", "neutral"):
        raise ValidationError(f"Invalid sentiment: {sentiment}")

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        raise ValidationError(f"Confidence must be a number between 0 and 1, got: {confidence}")

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning:
        raise ValidationError("Reasoning must be a non-empty string")

    return SentimentResult(
        sentiment=sentiment,
        confidence=float(confidence),
        reasoning=reasoning,
    )


def extract_json(text: str) -> str:
    """Extract JSON from text that might contain markdown code fences."""
    # Try to find JSON in code fences
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a JSON object or array
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_match:
        return json_match.group(1).strip()

    return text.strip()


async def parse_with_retry(
    complete: CompletionFn,
    messages: list[Message],
    config: LLMConfig,
    validate: Callable[[Any], Any],
    max_retries: int = 2,
) -> Any:
    """
    Parse and validate LLM JSON output with retries.
    On validation failure, send the error back to the model for correction.
    """
    last_error = ""

    for i in range(max_retries + 1):
        if i == 0:
            current_messages = messages
        else:
            current_messages = [
                *messages,
                Message(role="assistant", content="[Previous attempt had invalid output]"),
                Message(
                    role="user",
                    content=(
                        "Your previous response was not valid JSON or didn't match "
                        f"the expected schema. Error: {last_error}\n\n"
                        "Please try again, returning ONLY valid JSON matching the required format."
                    ),
                ),
            ]

        raw = await complete(current_messages, config)
        json_str = extract_json(raw)

        try:
            parsed = json.loads(json_str)
            return validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    raise ValidationError(
        f"Failed to get valid structured output after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Example 4: Retry with Exponential Backoff
# ---------------------------------------------------------------------------

@dataclass
class RetryOptions:
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    retryable_statuses: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 529})


class RetryableError(Exception):
    def __init__(self, message: str, status_code: int, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    options: RetryOptions | None = None,
) -> T:
    """
    Retry wrapper with exponential backoff and jitter.
    Respects Retry-After headers from rate limit responses.
    """
    opts = options or RetryOptions()

    for attempt in range(opts.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            is_last = attempt == opts.max_retries
            if is_last:
                raise

            # Check if error is retryable
            if isinstance(e, RetryableError):
                if e.status_code not in opts.retryable_statuses:
                    raise
                if e.retry_after is not None:
                    await asyncio.sleep(e.retry_after)
                    continue

            # Exponential backoff with jitter
            import random
            delay = min(
                opts.base_delay * (2 ** attempt) + random.random(),
                opts.max_delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Example 5: Simple Response Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    token_count: int


class ResponseCache:
    """
    In-memory LRU cache for LLM responses.
    In production, use Redis or a dedicated caching layer.
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    def _key(self, messages: list[Message], config: LLMConfig) -> str:
        payload = json.dumps(
            {"messages": [(m.role, m.content) for m in messages], "model": config.model, "temperature": config.temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, messages: list[Message], config: LLMConfig) -> str | None:
        key = self._key(messages, config)
        entry = self._cache.get(key)

        if entry is None:
            return None

        if time.time() - entry.timestamp > self._ttl:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.value

    def set(self, messages: list[Message], config: LLMConfig, value: str, token_count: int = 0) -> None:
        key = self._key(messages, config)

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value=value, timestamp=time.time(), token_count=token_count)

    def wrap(self, complete: CompletionFn) -> CompletionFn:
        """Wrap a completion function with caching."""
        cache = self

        async def cached_complete(messages: list[Message], config: LLMConfig) -> str:
            # Only cache deterministic requests
            if (config.temperature or 1.0) > 0:
                return await complete(messages, config)

            cached = cache.get(messages, config)
            if cached is not None:
                return cached

            result = await complete(messages, config)
            cache.set(messages, config, result)
            return result

        return cached_complete

    @property
    def stats(self) -> dict[str, int]:
        return {"size": len(self._cache), "max_entries": self._max_entries}


# ---------------------------------------------------------------------------
# Example 6: Request Tracking / Observability
# ---------------------------------------------------------------------------

@dataclass
class RequestLog:
    id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: CostEstimate
    success: bool
    error: str | None = None


def with_observability(
    complete: CompletionFn,
    on_log: Callable[[RequestLog], None],
) -> CompletionFn:
    """Wraps a completion function with logging and metrics."""
    counter = 0

    async def observed(messages: list[Message], config: LLMConfig) -> str:
        nonlocal counter
        counter += 1
        request_id = f"req-{counter}-{int(time.time()):x}"
        start = time.time()
        input_tokens = estimate_message_tokens(messages)

        try:
            result = await complete(messages, config)
            output_tokens = estimate_tokens(result)
            latency_ms = (time.time() - start) * 1000

            on_log(RequestLog(
                id=request_id,
                timestamp=start,
                model=config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=estimate_cost(config.model, input_tokens, output_tokens),
                success=True,
            ))
            return result

        except Exception as e:
            on_log(RequestLog(
                id=request_id,
                timestamp=start,
                model=config.model,
                input_tokens=input_tokens,
                output_tokens=0,
                latency_ms=(time.time() - start) * 1000,
                cost=estimate_cost(config.model, input_tokens, 0),
                success=False,
                error=str(e),
            ))
            raise

    return observed


# ---------------------------------------------------------------------------
# Example 7: Context Window Management
# ---------------------------------------------------------------------------

async def manage_context(
    messages: list[Message],
    context_window: int,
    complete: CompletionFn,
    strategy: Literal["truncate", "summarize"] = "truncate",
) -> list[Message]:
    """
    Manage conversation history to stay within context limits.
    Strategies: truncation or summarization.
    """
    reserved = 4096

    if check_context_fit(messages, context_window, reserved).fits:
        return messages

    system = [m for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]

    if strategy == "truncate":
        # Keep system + most recent messages that fit
        for i in range(len(conversation)):
            candidate = system + conversation[i:]
            if check_context_fit(candidate, context_window, reserved).fits:
                return candidate
        # Last resort
        return system + [conversation[-1]]

    # Summarization strategy
    old_messages = conversation[:-4]
    recent_messages = conversation[-4:]

    summary_text = "\n".join(f"{m.role}: {m.content}" for m in old_messages)

    summary = await complete(
        [
            Message(
                role="system",
                content="Summarize this conversation concisely, preserving key facts and decisions. Max 200 words.",
            ),
            Message(role="user", content=summary_text),
        ],
        LLMConfig(model="gpt-4o-mini", temperature=0, max_tokens=300),
    )

    return [
        *system,
        Message(role="assistant", content=f"[Previous conversation summary: {summary}]"),
        *recent_messages,
    ]


# ---------------------------------------------------------------------------
# Composing It All Together
# ---------------------------------------------------------------------------

def create_production_client(
    base_complete: CompletionFn,
    *,
    enable_cache: bool = False,
    enable_retry: bool = False,
    enable_logging: bool = False,
    on_log: Callable[[RequestLog], None] | None = None,
) -> CompletionFn:
    """Create a production-ready completion function by composing middleware."""
    complete = base_complete

    # Layer 1: Retry (innermost — retries the actual API call)
    if enable_retry:
        inner = complete

        async def retrying_complete(messages: list[Message], config: LLMConfig) -> str:
            return await with_retry(lambda: inner(messages, config))

        complete = retrying_complete

    # Layer 2: Caching (caches successful results)
    if enable_cache:
        cache = ResponseCache()
        complete = cache.wrap(complete)

    # Layer 3: Observability (outermost — logs everything including cache hits)
    if enable_logging and on_log:
        complete = with_observability(complete, on_log)

    return complete


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

async def demo(base_complete: CompletionFn, stream_complete: StreamCompletionFn) -> None:
    # Create a production client with all middleware
    complete = create_production_client(
        base_complete,
        enable_cache=True,
        enable_retry=True,
        enable_logging=True,
        on_log=lambda log: print(
            f"[{log.id}] {log.model} | {log.latency_ms:.0f}ms | "
            f"${log.cost.total_cost:.4f} | {'OK' if log.success else 'ERROR'}"
        ),
    )

    # Structured output with validation and retry
    sentiment = await parse_with_retry(
        complete,
        [
            Message(
                role="system",
                content=(
                    'Analyze sentiment. Return JSON: '
                    '{ "sentiment": "positive"|"negative"|"neutral", '
                    '"confidence": 0-1, "reasoning": "..." }'
                ),
            ),
            Message(role="user", content="This product exceeded my expectations!"),
        ],
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
        validate_sentiment,
    )
    print("Sentiment:", sentiment)

    # Streaming
    stream = stream_complete(
        [Message(role="user", content="Explain RAG in 3 sentences.")],
        LLMConfig(model="gpt-4o", temperature=0.3),
    )

    full_response = await handle_stream(
        stream,
        on_token=lambda token: print(token, end="", flush=True),
        on_complete=lambda text, usage: print(f"\n\nDone. Tokens: {usage}"),
    )

    # Cost estimation
    cost = estimate_cost("gpt-4o", 1000, 500)
    print(f"Estimated cost: ${cost.total_cost:.4f}")
