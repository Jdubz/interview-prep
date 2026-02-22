"""
Module 07 — Production Systems: Complete Runnable Patterns

Each class and function is a self-contained production pattern for LLM systems.
These are reference implementations -- adapt to your stack.

Dependencies: httpx, tiktoken (optional), pydantic (optional)
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Streaming Response Handler (SSE Parser)
# ---------------------------------------------------------------------------

async def stream_sse_response(
    url: str,
    headers: dict,
    payload: dict,
) -> AsyncIterator[str]:
    """
    Parse an SSE stream from an OpenAI-compatible API.
    Yields content tokens as they arrive.

    Production notes:
    - Uses httpx for async HTTP with proper timeout handling.
    - Handles partial lines, empty events, and the [DONE] sentinel.
    - In production, wrap this with retry logic and circuit breaker.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0),
        ) as response:
            response.raise_for_status()

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                # SSE events are separated by double newlines
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.strip().split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            return
                        try:
                            parsed = json.loads(data)
                            delta = parsed["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            logger.warning("Failed to parse SSE event: %s", data)


async def collect_streamed_tool_calls(
    url: str,
    headers: dict,
    payload: dict,
) -> list[dict]:
    """
    Collect tool calls from an SSE stream.

    Tool call arguments arrive in fragments across multiple events.
    We accumulate them by index and parse the complete JSON at the end.
    """
    import httpx

    tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_buffer}

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", url, headers=headers, json=payload,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0),
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.strip().split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        parsed = json.loads(data)
                        delta = parsed["choices"][0].get("delta", {})
                        for tc in delta.get("tool_calls", []):
                            idx = tc["index"]
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "",
                                }
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls[idx]["name"] = tc["function"]["name"]
                            tool_calls[idx]["arguments"] += tc.get("function", {}).get("arguments", "")

    # Parse accumulated argument strings into dicts
    results = []
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        tc["arguments"] = json.loads(tc["arguments"])
        results.append(tc)
    return results


# ---------------------------------------------------------------------------
# 2. Response Cache with TTL
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Hash-based response cache with TTL.

    Cache key is a SHA-256 hash of the full request signature:
    model + messages + temperature + tools + response_format.

    Production notes:
    - Replace dict with Redis for multi-process/multi-node deployments.
    - Add max-size eviction (LRU) to bound memory usage.
    - Consider separate TTLs per feature or query type.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self._cache: dict[str, dict] = {}
        self._default_ttl = default_ttl_seconds

    def _make_key(self, model: str, messages: list[dict], **kwargs) -> str:
        """Deterministic hash of request parameters."""
        key_data = json.dumps(
            {"model": model, "messages": messages, **kwargs},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], **kwargs) -> Optional[dict]:
        """Look up a cached response. Returns None on miss or expiry."""
        key = self._make_key(model, messages, **kwargs)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None
        logger.info("Cache hit for key %s", key[:12])
        return entry["response"]

    def put(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        ttl_seconds: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Store a response in the cache."""
        key = self._make_key(model, messages, **kwargs)
        self._cache[key] = {
            "response": response,
            "expires_at": time.time() + (ttl_seconds or self._default_ttl),
            "created_at": time.time(),
        }

    def invalidate(self, model: str, messages: list[dict], **kwargs) -> None:
        """Explicitly remove a cache entry."""
        key = self._make_key(model, messages, **kwargs)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# 3. Cost Tracker
# ---------------------------------------------------------------------------

# Pricing per million tokens (input, output) -- update as prices change
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
}


@dataclass
class UsageRecord:
    timestamp: float
    model: str
    feature: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class CostTracker:
    """
    Track LLM costs per request, per feature, per user, per model.

    Production notes:
    - In production, write records to a database or analytics pipeline.
    - Use this for real-time dashboards and alerting.
    - The alert thresholds should be configurable per feature.
    """

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._by_feature: dict[str, float] = defaultdict(float)
        self._by_user: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)
        self._daily_cost: float = 0.0

    def record(
        self,
        model: str,
        feature: str,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> UsageRecord:
        """Record a single LLM API call and compute its cost."""
        pricing = MODEL_PRICING.get(model, (5.0, 15.0))  # fallback to mid-range
        cost = (
            input_tokens * pricing[0] / 1_000_000
            + output_tokens * pricing[1] / 1_000_000
        )

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            feature=feature,
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        self._records.append(record)
        self._by_feature[feature] += cost
        self._by_user[user_id] += cost
        self._by_model[model] += cost
        self._daily_cost += cost

        return record

    def get_cost_by_feature(self) -> dict[str, float]:
        return dict(self._by_feature)

    def get_cost_by_user(self) -> dict[str, float]:
        return dict(self._by_user)

    def get_cost_by_model(self) -> dict[str, float]:
        return dict(self._by_model)

    def get_daily_cost(self) -> float:
        return self._daily_cost

    def check_alerts(
        self,
        daily_budget: float = 100.0,
        per_user_limit: float = 5.0,
    ) -> list[str]:
        """Check for cost anomalies and return alert messages."""
        alerts = []
        if self._daily_cost > daily_budget:
            alerts.append(
                f"ALERT: Daily cost ${self._daily_cost:.2f} exceeds budget ${daily_budget:.2f}"
            )
        for user_id, cost in self._by_user.items():
            if cost > per_user_limit:
                alerts.append(
                    f"ALERT: User {user_id} cost ${cost:.2f} exceeds limit ${per_user_limit:.2f}"
                )
        return alerts


# ---------------------------------------------------------------------------
# 4. Model Router
# ---------------------------------------------------------------------------

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Model tiers mapped to complexity levels
MODEL_TIERS = {
    TaskComplexity.SIMPLE: "gpt-4o-mini",
    TaskComplexity.MODERATE: "claude-sonnet-4-20250514",
    TaskComplexity.COMPLEX: "claude-opus-4-20250514",
}


class ModelRouter:
    """
    Route requests to the cheapest model that can handle the task.

    Uses keyword heuristics for zero-latency routing. In production,
    consider a small classifier model for more accurate routing.

    Production notes:
    - Log routing decisions for analysis. Track quality by routed tier.
    - Periodically review: are simple-routed tasks actually being handled well?
    - A/B test routing thresholds.
    """

    # Keywords that indicate the task needs a more capable model
    COMPLEX_SIGNALS = [
        "analyze", "compare", "contrast", "evaluate", "synthesize",
        "step by step", "reasoning", "explain why", "trade-offs",
        "design", "architect", "complex", "nuanced", "multi-step",
    ]

    MODERATE_SIGNALS = [
        "summarize", "extract", "convert", "translate", "rewrite",
        "format", "parse", "classify", "categorize",
    ]

    def classify(self, query: str) -> TaskComplexity:
        """Classify task complexity based on the user query."""
        query_lower = query.lower()

        # Check for complex signals first (higher priority)
        complex_score = sum(1 for s in self.COMPLEX_SIGNALS if s in query_lower)
        if complex_score >= 2 or len(query) > 2000:
            return TaskComplexity.COMPLEX

        moderate_score = sum(1 for s in self.MODERATE_SIGNALS if s in query_lower)
        if moderate_score >= 1 or len(query) > 500:
            return TaskComplexity.MODERATE

        return TaskComplexity.SIMPLE

    def select_model(self, query: str) -> str:
        """Select the appropriate model for the given query."""
        complexity = self.classify(query)
        model = MODEL_TIERS[complexity]
        logger.info("Routed query to %s (complexity: %s)", model, complexity.value)
        return model


# ---------------------------------------------------------------------------
# 5. Rate Limiter (Token Bucket)
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for LLM API calls.

    Supports both RPM (requests per minute) and TPM (tokens per minute).

    Production notes:
    - For multi-process deployments, use Redis-backed rate limiting.
    - This implementation is single-process and async-safe.
    - Separate limiters per provider/model if limits differ.
    """

    def __init__(self, max_rpm: int = 500, max_tpm: int = 100_000):
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        self._request_tokens = float(max_rpm)
        self._token_tokens = float(max_tpm)
        self._last_refill = time.monotonic()
        self._rpm_rate = max_rpm / 60.0  # tokens per second
        self._tpm_rate = max_tpm / 60.0

    def _refill(self) -> None:
        """Refill buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._request_tokens = min(
            self._max_rpm,
            self._request_tokens + elapsed * self._rpm_rate,
        )
        self._token_tokens = min(
            self._max_tpm,
            self._token_tokens + elapsed * self._tpm_rate,
        )
        self._last_refill = now

    def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """
        Try to acquire capacity for a request.

        Args:
            estimated_tokens: Estimated total tokens (input + output) for the request.

        Returns:
            True if the request can proceed, False if rate limited.
        """
        self._refill()
        if self._request_tokens < 1 or self._token_tokens < estimated_tokens:
            return False
        self._request_tokens -= 1
        self._token_tokens -= estimated_tokens
        return True

    async def wait_and_acquire(self, estimated_tokens: int = 1000) -> None:
        """Block until capacity is available, then acquire."""
        while not self.try_acquire(estimated_tokens):
            await asyncio.sleep(0.1)  # poll every 100ms

    @property
    def available_requests(self) -> int:
        self._refill()
        return int(self._request_tokens)

    @property
    def available_tokens(self) -> int:
        self._refill()
        return int(self._token_tokens)


# ---------------------------------------------------------------------------
# 6. Retry Handler with Exponential Backoff and Fallback
# ---------------------------------------------------------------------------

class RetryableError(Exception):
    """Raised for errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Raised for errors that should not be retried."""
    pass


async def call_with_retry_and_fallback(
    primary_fn,
    fallback_fns: list = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> dict:
    """
    Call an LLM API with exponential backoff and model fallback.

    1. Try the primary function up to max_retries times with backoff.
    2. If all retries fail, try each fallback function in order.
    3. If all fallbacks fail, raise the last error.

    Production notes:
    - The primary_fn and fallback_fns are async callables returning a response dict.
    - In practice, these wrap your LLM client calls for different models.
    - Log every retry and fallback for observability.
    """
    fallback_fns = fallback_fns or []
    last_error = None

    # Try primary with retries
    for attempt in range(max_retries):
        try:
            return await primary_fn()
        except NonRetryableError:
            raise  # don't retry 400, 401, 404
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay)
                logger.warning(
                    "Primary call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, max_retries, e, jitter,
                )
                await asyncio.sleep(jitter)

    # Try fallbacks
    for i, fallback_fn in enumerate(fallback_fns):
        try:
            logger.warning("Trying fallback %d after primary exhausted", i + 1)
            return await fallback_fn()
        except Exception as e:
            last_error = e
            logger.warning("Fallback %d failed: %s", i + 1, e)

    raise last_error  # all options exhausted


# ---------------------------------------------------------------------------
# 7. Request/Response Logger
# ---------------------------------------------------------------------------

@dataclass
class LLMLogEntry:
    """Structured log entry for a single LLM API call."""
    request_id: str
    timestamp: float
    model: str
    provider: str
    feature: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    ttft_ms: Optional[float]
    status: str  # "success", "error", "timeout"
    error_message: Optional[str]
    cache_hit: bool
    tools_called: list[str]
    prompt_hash: str
    temperature: float
    max_tokens: int


class LLMLogger:
    """
    Structured logger for LLM API calls.

    Captures all fields needed for cost tracking, debugging, and monitoring.

    Production notes:
    - In production, emit these as structured JSON to your logging pipeline.
    - Separate prompt/response content from metadata (content is large).
    - Implement log levels: metadata always, content on debug or by flag.
    """

    def __init__(self):
        self._entries: list[LLMLogEntry] = []

    def log_call(
        self,
        request_id: str,
        model: str,
        provider: str,
        feature: str,
        user_id: str,
        messages: list[dict],
        response: Optional[dict],
        latency_ms: float,
        ttft_ms: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        cache_hit: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMLogEntry:
        """Log a single LLM API call with all relevant metadata."""
        # Compute cost from response usage
        input_tokens = 0
        output_tokens = 0
        tools_called = []

        if response and "usage" in response:
            input_tokens = response["usage"].get("prompt_tokens", 0)
            output_tokens = response["usage"].get("completion_tokens", 0)

        # Extract tool call names from response
        if response:
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                for tc in message.get("tool_calls", []):
                    tools_called.append(tc.get("function", {}).get("name", "unknown"))

        pricing = MODEL_PRICING.get(model, (5.0, 15.0))
        cost = (
            input_tokens * pricing[0] / 1_000_000
            + output_tokens * pricing[1] / 1_000_000
        )

        prompt_hash = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:16]

        entry = LLMLogEntry(
            request_id=request_id,
            timestamp=time.time(),
            model=model,
            provider=provider,
            feature=feature,
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status=status,
            error_message=error_message,
            cache_hit=cache_hit,
            tools_called=tools_called,
            prompt_hash=prompt_hash,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._entries.append(entry)

        # Emit structured JSON log
        logger.info(json.dumps({
            "type": "llm_call",
            "request_id": entry.request_id,
            "model": entry.model,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "cost_usd": round(entry.cost_usd, 6),
            "latency_ms": round(entry.latency_ms, 1),
            "ttft_ms": round(entry.ttft_ms, 1) if entry.ttft_ms else None,
            "status": entry.status,
            "cache_hit": entry.cache_hit,
        }))

        return entry

    def get_entries(
        self,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> list[LLMLogEntry]:
        """Query log entries with optional filters."""
        results = self._entries
        if feature:
            results = [e for e in results if e.feature == feature]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results


# ---------------------------------------------------------------------------
# 8. Circuit Breaker for LLM API Calls
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for LLM provider calls.

    Prevents cascading failures by fast-failing when a provider is down,
    instead of waiting for timeouts on every request.

    States:
      CLOSED:    Requests flow normally. Track consecutive failures.
      OPEN:      Requests fail immediately. Wait for reset_timeout.
      HALF_OPEN: Allow one test request. Success closes circuit, failure reopens.

    Production notes:
    - Use one circuit breaker per provider (not per model).
    - Integrate with your fallback chain: when circuit opens, route to fallback.
    - Emit metrics on state transitions for alerting.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        name: str = "default",
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.monotonic()

    @property
    def state(self) -> CircuitState:
        """Get current state, automatically transitioning OPEN -> HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._reset_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()
        logger.info(
            "Circuit breaker [%s]: %s -> %s",
            self.name, old_state.value, new_state.value,
        )

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state  # triggers OPEN -> HALF_OPEN check
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # allow the test request
        return False  # OPEN -- fail fast

    def record_success(self) -> None:
        """Record a successful API call."""
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed API call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self._failure_threshold:
            self._transition_to(CircuitState.OPEN)

    async def call(self, fn, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Raises CircuitBreakerOpen if the circuit is open.
        Records success/failure and transitions state accordingly.
        """
        if not self.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker [{self.name}] is OPEN. "
                f"Retry after {self._reset_timeout}s."
            )
        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting requests."""
    pass


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

async def demo():
    """Demonstrate the production patterns working together."""

    # -- Cache --
    cache = ResponseCache(default_ttl_seconds=300)
    messages = [{"role": "user", "content": "What is Python?"}]

    cached = cache.get("gpt-4o-mini", messages)
    if cached:
        print("Cache hit:", cached)
    else:
        print("Cache miss -- would call API here")
        fake_response = {"choices": [{"message": {"content": "Python is..."}}]}
        cache.put("gpt-4o-mini", messages, fake_response)

    # -- Model Router --
    router = ModelRouter()
    print("Simple query model:", router.select_model("Say hello"))
    print("Complex query model:", router.select_model(
        "Analyze the trade-offs between microservices and monolithic architecture, "
        "step by step, comparing scalability and maintainability."
    ))

    # -- Rate Limiter --
    limiter = TokenBucketRateLimiter(max_rpm=10, max_tpm=50_000)
    print(f"Available requests: {limiter.available_requests}")
    print(f"Can acquire: {limiter.try_acquire(estimated_tokens=2000)}")

    # -- Cost Tracker --
    tracker = CostTracker()
    record = tracker.record(
        model="gpt-4o-mini",
        feature="support_chat",
        user_id="user-123",
        input_tokens=1500,
        output_tokens=400,
        latency_ms=1200,
    )
    print(f"Request cost: ${record.cost_usd:.4f}")
    print(f"Daily cost: ${tracker.get_daily_cost():.4f}")

    # -- Circuit Breaker --
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=5, name="openai")
    print(f"Circuit state: {cb.state.value}")

    # Simulate failures
    for i in range(3):
        cb.record_failure()
    print(f"After 3 failures: {cb.state.value}")  # should be OPEN


if __name__ == "__main__":
    asyncio.run(demo())
