"""
Module 07 — Production Systems: Exercises

Skeleton implementations with TODOs. Each exercise builds a production-grade
component for LLM systems. Fill in the implementations.

Difficulty ratings:
  [1] Straightforward -- apply concepts from the README directly.
  [2] Moderate -- requires combining multiple concepts.
  [3] Challenging -- requires design decisions and tradeoff analysis.
"""

import asyncio
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


# ===========================================================================
# Exercise 1: SSE Streaming Parser with Tool Call Support [2]
#
# Build a streaming parser that handles both content tokens and tool calls.
# The parser should:
# - Yield content tokens as they arrive
# - Accumulate tool call arguments across multiple events
# - Return complete tool calls when the stream finishes
# - Handle errors mid-stream gracefully
# ===========================================================================

@dataclass
class StreamEvent:
    """Represents a parsed event from an SSE stream."""
    event_type: str  # "content", "tool_call_start", "tool_call_delta", "done", "error"
    content: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    tool_call_name: Optional[str] = None
    tool_call_arguments_delta: Optional[str] = None


class SSEStreamParser:
    """
    Parse an SSE stream from an OpenAI-compatible API.

    Handles:
    - Content tokens (yielded immediately)
    - Tool call fragments (accumulated and returned on completion)
    - Mid-stream errors
    - The [DONE] sentinel

    Usage:
        parser = SSEStreamParser()
        async for event in parser.parse(raw_sse_lines):
            if event.event_type == "content":
                print(event.content, end="")
            elif event.event_type == "done":
                tool_calls = parser.get_completed_tool_calls()
    """

    def __init__(self):
        self._tool_call_buffers: dict[int, dict] = {}
        # Each buffer: {"id": str, "name": str, "arguments": str}

    async def parse(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamEvent]:
        """
        Parse raw SSE lines into structured StreamEvents.

        Args:
            lines: Async iterator of raw SSE lines (e.g., from an HTTP stream).

        Yields:
            StreamEvent for each meaningful event in the stream.
        """
        # TODO: Implement the SSE parser.
        #
        # For each line:
        # 1. Skip empty lines and lines that don't start with "data: "
        # 2. Handle the "[DONE]" sentinel -- yield a "done" event and return
        # 3. Parse the JSON data
        # 4. Extract the delta from choices[0].delta
        # 5. If delta has "content", yield a "content" event
        # 6. If delta has "tool_calls", process each tool call:
        #    a. If this is a new tool call index, yield "tool_call_start"
        #       and initialize the buffer
        #    b. If arguments are present, yield "tool_call_delta"
        #       and append to the buffer
        # 7. Handle JSON parse errors by yielding an "error" event
        #
        # Hint: Tool calls have an "index" field. Multiple tool calls can
        # be in progress simultaneously (parallel tool use).

        raise NotImplementedError("Implement SSE parser")

    def get_completed_tool_calls(self) -> list[dict]:
        """
        Return all accumulated tool calls with their parsed arguments.

        Returns:
            List of dicts with keys: id, name, arguments (parsed JSON).
        """
        # TODO: Iterate over self._tool_call_buffers, parse the
        # accumulated argument strings into JSON, and return the results.
        # Handle JSON parse errors gracefully.

        raise NotImplementedError("Implement tool call aggregation")


# ===========================================================================
# Exercise 2: Response Cache with Semantic Similarity [3]
#
# Build a cache that supports both exact-match and semantic similarity
# lookups. When an exact match misses, fall back to finding semantically
# similar cached queries.
# ===========================================================================

@dataclass
class CacheEntry:
    query_text: str
    query_embedding: list[float]
    response: dict
    created_at: float
    expires_at: float
    hit_count: int = 0


class SemanticCache:
    """
    Response cache with exact-match and semantic similarity fallback.

    Lookup order:
    1. Exact match (hash-based, O(1))
    2. Semantic match (embedding similarity, O(n) or indexed)

    The embed_fn should accept a string and return a list of floats.
    """

    def __init__(
        self,
        embed_fn,  # Callable[[str], list[float]]
        similarity_threshold: float = 0.95,
        default_ttl_seconds: int = 3600,
        max_entries: int = 10_000,
    ):
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._exact_cache: dict[str, CacheEntry] = {}   # hash -> entry
        self._semantic_entries: list[CacheEntry] = []     # for similarity search

    def _hash_query(self, query: str, model: str, **kwargs) -> str:
        """Create a deterministic hash for exact-match lookup."""
        # TODO: Hash the query, model, and any other relevant parameters.
        # Use SHA-256. Sort keys for determinism.
        raise NotImplementedError

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        # TODO: Implement cosine similarity.
        # cos_sim = dot(a, b) / (norm(a) * norm(b))
        # Handle zero vectors gracefully.
        raise NotImplementedError

    def get(self, query: str, model: str, **kwargs) -> Optional[dict]:
        """
        Look up a cached response.

        1. Try exact match first (fast).
        2. If no exact match, compute query embedding and search
           for semantically similar cached queries.
        3. Return None on miss.

        On hit, update the hit_count for cache analytics.
        """
        # TODO: Implement the two-tier lookup.
        #
        # Exact match:
        #   - Hash the query
        #   - Look up in self._exact_cache
        #   - Check TTL (expired entries should be removed)
        #
        # Semantic match (if exact miss):
        #   - Embed the query using self._embed_fn
        #   - Iterate over self._semantic_entries
        #   - Compute cosine similarity with each entry's embedding
        #   - If similarity > self._similarity_threshold, return the response
        #   - Return the highest-similarity match above threshold
        #   - Remove expired entries as you encounter them

        raise NotImplementedError

    def put(self, query: str, model: str, response: dict, **kwargs) -> None:
        """
        Store a response in both exact and semantic caches.

        If the cache is full (max_entries), evict the least-recently-used entry.
        """
        # TODO: Implement cache insertion.
        #
        # 1. Compute the query hash for exact cache
        # 2. Compute the query embedding for semantic cache
        # 3. Create a CacheEntry
        # 4. Store in both self._exact_cache and self._semantic_entries
        # 5. If over max_entries, evict the oldest entry

        raise NotImplementedError

    def stats(self) -> dict:
        """Return cache statistics."""
        # TODO: Return a dict with:
        #   - exact_entries: number of entries in exact cache
        #   - semantic_entries: number of entries in semantic cache
        #   - total_hits: sum of hit_count across all entries

        raise NotImplementedError


# ===========================================================================
# Exercise 3: Cost Monitoring System with Anomaly Detection [2]
#
# Track LLM costs by feature and detect anomalies using a rolling average.
# Alert when a feature's cost deviates significantly from its baseline.
# ===========================================================================

@dataclass
class CostRecord:
    timestamp: float
    feature: str
    model: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostMonitor:
    """
    Track LLM costs by feature and detect anomalies.

    Anomaly detection: compare the current hour's cost to the trailing
    7-day hourly average for that feature. Alert if current > N * average.
    """

    # Pricing per million tokens: (input, output)
    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
    }

    def __init__(self, anomaly_multiplier: float = 3.0):
        self._records: list[CostRecord] = []
        self._anomaly_multiplier = anomaly_multiplier

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute the USD cost for a single API call."""
        # TODO: Look up the model in PRICING. If not found, use a reasonable
        # default. Compute: (input_tokens * input_price + output_tokens * output_price) / 1M
        raise NotImplementedError

    def record(
        self,
        feature: str,
        model: str,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Record a single API call and compute its cost."""
        # TODO: Compute cost, create a CostRecord, append to self._records, return it.
        raise NotImplementedError

    def get_hourly_cost_by_feature(self, hours_back: int = 1) -> dict[str, float]:
        """
        Get total cost per feature for the last N hours.

        Returns:
            Dict mapping feature name to total cost in USD.
        """
        # TODO: Filter self._records to the last N hours.
        # Group by feature, sum costs.
        raise NotImplementedError

    def get_trailing_average(self, feature: str, days: int = 7) -> float:
        """
        Compute the average hourly cost for a feature over the trailing N days.

        Returns:
            Average cost per hour in USD. Returns 0.0 if no data.
        """
        # TODO:
        # 1. Filter records for this feature in the last N days
        # 2. Compute total cost
        # 3. Divide by (N * 24) to get average hourly cost
        # Handle the case where there's no historical data.
        raise NotImplementedError

    def check_anomalies(self) -> list[str]:
        """
        Check all features for cost anomalies in the current hour.

        An anomaly is when the current hour's cost exceeds
        anomaly_multiplier * trailing_average.

        Returns:
            List of alert messages for anomalous features.
        """
        # TODO:
        # 1. Get current hourly costs by feature
        # 2. For each feature, compute trailing average
        # 3. If current > multiplier * average, generate an alert
        # 4. Also alert if there's no historical baseline (new feature spike)
        raise NotImplementedError

    def get_top_users(self, feature: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Get the top N users by cost for a given feature.

        Returns:
            List of (user_id, total_cost) tuples, sorted descending by cost.
        """
        # TODO: Filter by feature, group by user_id, sum costs, sort, return top N.
        raise NotImplementedError


# ===========================================================================
# Exercise 4: Model Router with Confidence-Based Escalation [3]
#
# Implement a cascade router that tries a cheap model first and escalates
# to a more expensive model if the response confidence is low.
# ===========================================================================

@dataclass
class RoutingResult:
    model_used: str
    response: str
    confidence: float
    escalated: bool
    total_cost: float
    attempts: list[dict]  # [{model, response, confidence, cost}]


class CascadeRouter:
    """
    Try the cheapest model first. If confidence is below threshold,
    escalate to the next tier.

    Model tiers (cheapest to most expensive):
      1. gpt-4o-mini
      2. claude-sonnet-4-20250514
      3. claude-opus-4-20250514

    Confidence estimation uses a configurable strategy.
    """

    TIERS = [
        {"model": "gpt-4o-mini", "input_price": 0.15, "output_price": 0.60},
        {"model": "claude-sonnet-4-20250514", "input_price": 3.00, "output_price": 15.00},
        {"model": "claude-opus-4-20250514", "input_price": 15.00, "output_price": 75.00},
    ]

    def __init__(
        self,
        call_fn,  # async Callable[[str, str], dict] -- (model, prompt) -> response
        confidence_fn,  # Callable[[dict], float] -- response -> confidence 0-1
        confidence_threshold: float = 0.8,
        max_tier: int = 3,  # max number of tiers to try
    ):
        self._call_fn = call_fn
        self._confidence_fn = confidence_fn
        self._confidence_threshold = confidence_threshold
        self._max_tier = min(max_tier, len(self.TIERS))

    async def route(self, prompt: str) -> RoutingResult:
        """
        Route a prompt through the cascade.

        1. Start with the cheapest tier.
        2. Call the model and estimate confidence.
        3. If confidence >= threshold, return the response.
        4. If confidence < threshold and more tiers available, escalate.
        5. If all tiers exhausted, return the best response seen.

        Returns:
            RoutingResult with the final response, model used, and cost breakdown.
        """
        # TODO: Implement the cascade routing logic.
        #
        # Track all attempts for observability:
        #   attempts = []
        #   for each tier up to self._max_tier:
        #     response = await self._call_fn(tier["model"], prompt)
        #     confidence = self._confidence_fn(response)
        #     cost = compute cost from token counts
        #     attempts.append({model, response_text, confidence, cost})
        #     if confidence >= threshold: return result
        #
        # If no tier met the threshold, return the attempt with highest confidence.
        #
        # Hint: extract response text and token counts from the response dict.
        # The response dict follows OpenAI's format:
        #   response["choices"][0]["message"]["content"] -> text
        #   response["usage"]["prompt_tokens"] -> input tokens
        #   response["usage"]["completion_tokens"] -> output tokens

        raise NotImplementedError

    def estimate_savings(
        self,
        routing_results: list[RoutingResult],
    ) -> dict:
        """
        Estimate cost savings compared to always using the most expensive tier.

        Args:
            routing_results: List of completed routing results.

        Returns:
            Dict with keys: actual_cost, max_tier_cost, savings_pct, escalation_rate
        """
        # TODO:
        # 1. Sum actual costs from all routing results
        # 2. Estimate what it would have cost using only the most expensive tier
        # 3. Compute savings percentage
        # 4. Compute escalation rate (% of requests that went beyond tier 1)

        raise NotImplementedError


# ===========================================================================
# Exercise 5: Dual Rate Limiter (RPM + TPM) [2]
#
# Build a rate limiter that enforces both requests-per-minute and
# tokens-per-minute limits, matching how providers actually rate limit.
# ===========================================================================

class DualRateLimiter:
    """
    Enforce both RPM and TPM limits using the token bucket algorithm.

    A request must pass BOTH limits to proceed. This matches how providers
    enforce rate limits: you can hit either the RPM or TPM ceiling.

    Features:
    - Separate buckets for requests and tokens
    - Priority support (high priority requests can exceed soft limits)
    - Wait-or-reject semantics
    """

    def __init__(
        self,
        max_rpm: int = 500,
        max_tpm: int = 100_000,
        priority_boost_pct: float = 0.2,  # high-priority gets 20% more capacity
    ):
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        self._priority_boost_pct = priority_boost_pct

        # TODO: Initialize two token buckets (one for RPM, one for TPM).
        # Each bucket needs:
        #   - current_tokens: float (starts at max)
        #   - last_refill_time: float
        #   - refill_rate: float (tokens per second)
        #   - max_capacity: float

        raise NotImplementedError

    def _refill_buckets(self) -> None:
        """Refill both buckets based on elapsed time since last refill."""
        # TODO: For each bucket:
        # 1. Calculate elapsed time since last refill
        # 2. Add elapsed * refill_rate tokens (capped at max_capacity)
        # 3. Update last_refill_time
        raise NotImplementedError

    def try_acquire(
        self,
        estimated_tokens: int,
        priority: str = "normal",
    ) -> bool:
        """
        Try to acquire capacity for a request.

        Args:
            estimated_tokens: Estimated total tokens (input + output).
            priority: "high" or "normal". High priority can exceed soft limits.

        Returns:
            True if the request can proceed, False if rate limited.
        """
        # TODO:
        # 1. Refill buckets
        # 2. Determine effective limits (apply priority boost if high priority)
        # 3. Check if BOTH buckets have sufficient capacity
        # 4. If yes, consume from both buckets and return True
        # 5. If no, return False (do NOT consume)

        raise NotImplementedError

    async def wait_and_acquire(
        self,
        estimated_tokens: int,
        priority: str = "normal",
        timeout_seconds: float = 30.0,
    ) -> bool:
        """
        Wait until capacity is available, then acquire. Times out after deadline.

        Returns:
            True if acquired within timeout, False if timed out.
        """
        # TODO:
        # 1. Try to acquire immediately
        # 2. If failed, poll every 100ms until acquired or timeout
        # 3. Return True on success, False on timeout
        raise NotImplementedError

    def get_status(self) -> dict:
        """Return current rate limiter status for monitoring."""
        # TODO: Return a dict with:
        #   - rpm_available: int
        #   - rpm_max: int
        #   - tpm_available: int
        #   - tpm_max: int
        #   - rpm_utilization_pct: float
        #   - tpm_utilization_pct: float
        raise NotImplementedError


# ===========================================================================
# Exercise 6: Observability Pipeline [3]
#
# Design a complete observability pipeline that captures, aggregates, and
# reports on all relevant LLM metrics. This is the glue that ties together
# cost tracking, latency monitoring, error tracking, and quality scoring.
# ===========================================================================

@dataclass
class LLMCallMetrics:
    """All metrics captured for a single LLM API call."""
    request_id: str
    timestamp: float
    model: str
    provider: str
    feature: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ttft_ms: Optional[float]  # time to first token
    total_latency_ms: float
    status: str  # "success", "error", "timeout"
    error_type: Optional[str]
    cache_hit: bool
    tools_called: list[str]
    output_parse_success: Optional[bool]  # for structured output
    quality_score: Optional[float]  # from eval pipeline, 0-1


class ObservabilityPipeline:
    """
    Central observability pipeline for LLM applications.

    Captures per-call metrics, computes aggregates, and generates
    reports and alerts.

    In production, this would emit to your metrics backend
    (Datadog, Prometheus, CloudWatch) and logging pipeline (ELK, Loki).
    This in-memory implementation demonstrates the logic.
    """

    def __init__(self):
        self._metrics: list[LLMCallMetrics] = []
        self._alert_rules: list[dict] = []

    def record(self, metrics: LLMCallMetrics) -> None:
        """
        Record metrics from a single LLM API call.

        In production, this would:
        1. Emit the metrics to your time-series database
        2. Update running aggregates
        3. Check alert rules
        4. Log to your structured logging pipeline
        """
        # TODO: Store the metrics and check alert rules.
        raise NotImplementedError

    def add_alert_rule(
        self,
        name: str,
        metric: str,         # e.g., "error_rate", "p95_latency", "hourly_cost"
        threshold: float,
        window_minutes: int,  # look-back window
        comparison: str,      # "gt" (greater than) or "lt" (less than)
    ) -> None:
        """
        Register an alert rule.

        Example:
            pipeline.add_alert_rule("high_error_rate", "error_rate", 0.05, 5, "gt")
            # Alert if error rate > 5% in the last 5 minutes
        """
        # TODO: Store the alert rule for checking on each record().
        raise NotImplementedError

    def _check_alerts(self) -> list[str]:
        """
        Evaluate all alert rules against current metrics.

        Returns:
            List of triggered alert messages.
        """
        # TODO: For each alert rule:
        # 1. Filter metrics to the rule's time window
        # 2. Compute the metric value:
        #    - "error_rate": count(status != "success") / total_count
        #    - "p95_latency": 95th percentile of total_latency_ms
        #    - "hourly_cost": sum(cost_usd) extrapolated to hourly rate
        #    - "cache_hit_rate": count(cache_hit) / total_count
        # 3. Compare against threshold using the comparison operator
        # 4. If triggered, generate an alert message

        raise NotImplementedError

    def get_dashboard_metrics(self, window_minutes: int = 60) -> dict:
        """
        Compute dashboard metrics for the given time window.

        Returns a dict with:
        - total_requests: int
        - error_rate: float (0-1)
        - p50_latency_ms: float
        - p95_latency_ms: float
        - p99_latency_ms: float
        - p50_ttft_ms: float
        - p95_ttft_ms: float
        - total_cost_usd: float
        - avg_cost_per_request: float
        - cache_hit_rate: float (0-1)
        - output_parse_success_rate: float (0-1)
        - top_models: list of (model, count) tuples
        - top_features: list of (feature, count) tuples
        - top_error_types: list of (error_type, count) tuples
        """
        # TODO: Filter to the time window, compute all metrics.
        #
        # For percentiles, sort the values and index:
        #   p95 = sorted_values[int(len(sorted_values) * 0.95)]
        #
        # Handle empty windows gracefully (return zeros).

        raise NotImplementedError

    def get_cost_breakdown(self, window_minutes: int = 60) -> dict:
        """
        Break down costs by model, feature, and user.

        Returns:
            {
                "by_model": {model: cost},
                "by_feature": {feature: cost},
                "by_user": {user_id: cost},  (top 10 only)
                "total": float
            }
        """
        # TODO: Filter to window, group and sum costs.
        raise NotImplementedError

    def get_latency_breakdown(
        self, feature: str, window_minutes: int = 60
    ) -> dict:
        """
        Detailed latency breakdown for a specific feature.

        Returns:
            {
                "count": int,
                "p50_ms": float,
                "p95_ms": float,
                "p99_ms": float,
                "ttft_p50_ms": float,
                "ttft_p95_ms": float,
                "by_model": {model: {"p50": float, "p95": float}},
                "cache_hit_latency_p50": float,
                "cache_miss_latency_p50": float,
            }
        """
        # TODO: Filter by feature and window. Compute percentiles overall
        # and sliced by model and cache hit/miss.
        raise NotImplementedError


# ===========================================================================
# Test Helpers
# ===========================================================================

def _test_exercise_1():
    """Verify SSEStreamParser handles content and tool calls."""
    parser = SSEStreamParser()

    # Simulate raw SSE lines
    raw_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"test\\"}"}}]}}]}',
        'data: [DONE]',
    ]

    async def run():
        async def line_iter():
            for line in raw_lines:
                yield line

        events = []
        async for event in parser.parse(line_iter()):
            events.append(event)

        content_events = [e for e in events if e.event_type == "content"]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"

        tool_calls = parser.get_completed_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["arguments"] == {"q": "test"}

        print("Exercise 1: PASSED")

    asyncio.run(run())


def _test_exercise_5():
    """Verify DualRateLimiter enforces both RPM and TPM."""
    limiter = DualRateLimiter(max_rpm=5, max_tpm=10_000)

    # Should allow first request
    assert limiter.try_acquire(2000) is True, "First request should succeed"

    # Should track remaining capacity
    status = limiter.get_status()
    assert status["rpm_available"] == 4, f"Expected 4 RPM remaining, got {status['rpm_available']}"

    # Exhaust RPM
    for _ in range(4):
        limiter.try_acquire(100)

    # Should be rate limited (RPM exhausted)
    assert limiter.try_acquire(100) is False, "Should be RPM limited"

    print("Exercise 5: PASSED")


if __name__ == "__main__":
    print("Run individual test functions to verify your implementations.")
    print("Example: _test_exercise_1()")
    print()
    print("Exercises:")
    print("  1. SSEStreamParser -- parse streaming responses with tool calls")
    print("  2. SemanticCache -- response cache with embedding similarity")
    print("  3. CostMonitor -- cost tracking with anomaly detection")
    print("  4. CascadeRouter -- confidence-based model escalation")
    print("  5. DualRateLimiter -- enforce RPM + TPM limits")
    print("  6. ObservabilityPipeline -- capture and aggregate LLM metrics")
