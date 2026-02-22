# Module 07 — Production Systems

Core knowledge for building, operating, and scaling LLM-powered systems in production. This module covers the engineering concerns that separate a working prototype from a reliable, cost-effective, observable production service.

---

## Streaming Architecture

### Why Streaming Matters

LLM inference is slow. A typical response of 500 tokens takes 5-15 seconds to generate in full. Without streaming, the user stares at a spinner for the entire duration. With streaming, the first token appears in 100-500ms and subsequent tokens flow in at 30-100 tokens/second.

| Metric | Non-Streaming | Streaming |
|---|---|---|
| Time to first visible token | 2-15s (entire generation) | 100-500ms |
| User perception | "Is it broken?" | "It's actively working" |
| Error recovery | All-or-nothing | Can abort early, show partial |
| Memory footprint | Full response buffered server-side | Incremental delivery |

### Server-Sent Events (SSE)

All major providers (OpenAI, Anthropic, Google) use SSE for streaming. SSE is a unidirectional HTTP protocol where the server pushes text events over a long-lived connection.

```
POST /v1/chat/completions
Content-Type: application/json
Accept: text/event-stream

---

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" world"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{}}],"finish_reason":"stop","usage":{"prompt_tokens":12,"completion_tokens":3}}

data: [DONE]
```

Key properties of SSE:
- Each event is prefixed with `data: ` and terminated by a double newline.
- The `[DONE]` sentinel signals stream completion.
- Unlike WebSockets, SSE is unidirectional (server to client) and built on standard HTTP.
- Automatic reconnection is built into the browser EventSource API, but most LLM use cases manage the connection manually.

### Streaming Implementation Patterns

**Token-by-token forwarding.** The simplest pattern: proxy each SSE event from the provider to your client. Your server acts as a pass-through. Use this when your client handles rendering.

**Buffer and transform.** Collect tokens into a buffer, apply transformations (markdown rendering, citation insertion, content filtering), and forward to the client on a cadence (e.g., every 100ms or every sentence boundary).

**Full-buffer with streaming UX.** Buffer the entire response but send periodic progress updates to the client. Useful when you need to validate or post-process the full response before displaying it.

### Partial JSON in Streams

When requesting structured output with `response_format: { type: "json_object" }`, the JSON arrives incrementally. You cannot parse it until the stream completes.

```
Token 1: {"
Token 2: name
Token 3: ":
Token 4:  "
Token 5: Alice
Token 6: ",
Token 7:  "age
Token 8: ":
Token 9:  30
Token 10: }
```

Strategies:
- **Buffer entirely, parse at end.** Simplest. Wait for `finish_reason: "stop"`, concatenate all deltas, parse JSON.
- **Incremental JSON parser.** Libraries like `ijson` (Python) can emit events as JSON structure emerges. Useful for very large JSON responses where you want to start processing early.
- **Streaming display of partial fields.** If you know the schema, you can detect when a field value starts and stream that field's content to the UI while buffering the overall structure.

### Tool Calls in Streams

Tool calls arrive incrementally across multiple SSE events. The structure differs by provider:

```
# OpenAI tool call streaming
delta: {"tool_calls": [{"index": 0, "id": "call_abc", "function": {"name": "get_weather"}}]}
delta: {"tool_calls": [{"index": 0, "function": {"arguments": "{\"ci"}}]}
delta: {"tool_calls": [{"index": 0, "function": {"arguments": "ty\":"}}]}
delta: {"tool_calls": [{"index": 0, "function": {"arguments": " \"NYC"}}]}
delta: {"tool_calls": [{"index": 0, "function": {"arguments": "\"}"}}]}
```

You must:
1. Accumulate argument fragments by tool call index.
2. Parse the complete JSON arguments only after the tool call is fully received.
3. Handle multiple concurrent tool calls (parallel tool use) — each has a distinct `index`.

### Backpressure

If your client or downstream consumer is slower than the token generation rate, tokens queue up in memory. In high-concurrency scenarios this leads to OOM.

Mitigation approaches:
- **Bounded buffers.** Set a maximum buffer size per stream. If exceeded, drop the connection and return an error.
- **Flow control.** If proxying through your own server, use async generators with bounded queues. If the client is slow, the queue fills, and you can pause reading from the provider.
- **Client-side throttling.** Batch UI updates to avoid rendering every individual token (e.g., update the DOM every 50ms, not every token).

---

## Caching Strategies

Caching is the single highest-ROI optimization for most LLM applications. A well-designed cache reduces cost, latency, and provider dependency simultaneously.

### Response Caching (Exact Match)

Hash the full request and store the response.

```
cache_key = hash(model + system_prompt + messages + temperature + tools + response_format)
```

**When it works:**
- Temperature = 0 (deterministic output).
- Repeated identical queries (search, FAQ, classification).
- Batch processing with duplicate inputs.

**When it fails:**
- Creative tasks where variety matters.
- Conversations with unique context.
- Queries over volatile data (unless TTL is short).

**TTL strategies:**
- Static data (product catalog): TTL = 24 hours.
- Semi-dynamic (support docs): TTL = 1-4 hours.
- Real-time data (stock prices in context): TTL = 0 (no caching) or < 1 minute.

Cache hit rates in production:
- FAQ / support bots: 30-60% hit rate typical.
- Search-style queries: 10-30% hit rate.
- Unique conversations: < 5% hit rate.

### Semantic Caching

For queries that are semantically similar but not identical, use embedding similarity.

```
1. Embed the incoming query.
2. Search the cache for entries with cosine similarity > threshold (e.g., 0.95).
3. If a match is found, return the cached response.
4. If not, call the LLM, cache the query embedding + response.
```

**Threshold tuning is critical:**
- 0.98+: Nearly identical queries only. Safe but low hit rate.
- 0.95: Minor rephrasings. Good balance for most use cases.
- 0.90: Meaningfully different queries may match. Risk of returning wrong answers.
- < 0.90: Do not use. Too many false positives.

**Implementation considerations:**
- Embedding computation adds 10-50ms latency per query.
- Vector similarity search adds 1-10ms (with an index).
- Net benefit is only positive if cache hit rate is high enough to offset the overhead on misses.
- Use a separate, fast vector store (not your main RAG store) for the cache index.

### Prompt Caching (Provider-Side)

Providers cache the KV-cache of your prompt prefix server-side. This reduces both cost and latency for repeated prefixes.

**Anthropic prompt caching:**
- Automatically caches prompt prefixes >= 1024 tokens (Sonnet) or >= 2048 tokens (Haiku).
- Cached input tokens cost 90% less than uncached.
- Cache write costs 25% more than standard input.
- Cache TTL: 5 minutes (refreshed on each use).
- Breakpoints: use `cache_control` to mark cacheable boundaries.

**OpenAI automatic caching:**
- Caches prompt prefixes >= 1024 tokens automatically.
- Cached tokens cost 50% less.
- No explicit opt-in required.

**Design patterns for cache-friendly prompts:**
- Place static content (system prompt, tool definitions, few-shot examples) at the beginning.
- Place dynamic content (user query, retrieved context) at the end.
- Keep the static prefix identical across requests — any change invalidates the cache.

```
+------------------------------------------+
|  System prompt (static, cacheable)       |  <-- cached across requests
|  Tool definitions (static, cacheable)    |
|  Few-shot examples (static, cacheable)   |
+------------------------------------------+
|  Retrieved context (varies per query)    |  <-- not cached
|  User message (varies per query)         |
+------------------------------------------+
```

### Cache Invalidation

The two hard problems in computer science apply here:

- **TTL-based.** Simple and predictable. Set TTL based on data freshness requirements.
- **Event-driven.** Invalidate when source data changes (e.g., when a document is updated in your knowledge base, invalidate all cached responses that used that document).
- **Version-based.** Include a version identifier in the cache key. Bump the version when your prompt template changes, when you update the model, or when you change retrieval parameters.
- **Manual purge.** Provide an admin interface to flush specific cache entries or the entire cache.

---

## Cost Optimization

### Model Routing

Route requests to the cheapest model that can handle the task. This is the highest-impact cost optimization available.

```
+----------------+     +-------------------+     +-----------------+
| Incoming       | --> | Complexity        | --> | Model Selection |
| Request        |     | Classifier        |     |                 |
+----------------+     +-------------------+     +-----------------+
                              |                         |
                        +-----+-----+             +-----+-----+
                        |           |             |           |
                     Simple     Complex        Haiku/     Opus/
                                             GPT-4o-mini  GPT-4o
```

**Routing strategies:**

| Strategy | How It Works | Tradeoff |
|---|---|---|
| Task-based | Route by feature (classify=small, reason=large) | Simple, coarse |
| Keyword heuristic | Check for complexity signals in the query | Fast, brittle |
| Classifier model | Small model classifies complexity | Adds latency, more accurate |
| Cascade | Try small model first, escalate if confidence is low | Higher latency on hard tasks |

**Cost comparison (approximate, mid-2025):**

| Model | Input $/1M tokens | Output $/1M tokens | Relative cost |
|---|---|---|---|
| GPT-4o-mini | $0.15 | $0.60 | 1x |
| Claude 3.5 Haiku | $0.80 | $4.00 | 5x |
| GPT-4o | $2.50 | $10.00 | 17x |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 20x |
| Claude Opus 4 | $15.00 | $75.00 | 100x |
| o1 | $15.00 | $60.00 | 100x |

Routing 80% of traffic to a small model cuts costs by 70-90% if the small model handles those tasks adequately.

### Token Reduction

Every token costs money. At scale, small reductions compound.

- **Concise system prompts.** Audit your system prompts regularly. A 2000-token system prompt at 100K requests/day costs real money.
- **Summarize conversation history.** Instead of passing the full 20-turn conversation, summarize older turns. Keep the last 2-3 turns verbatim for context.
- **Selective RAG.** Retrieve 3-5 highly relevant chunks instead of 10-15 mediocre ones. Better retrieval > more retrieval.
- **Output length control.** Set `max_tokens` to a reasonable limit. Use instructions like "respond in 2-3 sentences" for simple queries.
- **Compression.** For long documents, pre-summarize and cache the summaries.

### Batch Processing

Group similar requests and process them together.

- **Provider batch APIs.** OpenAI offers a Batch API at 50% discount with 24-hour turnaround. Ideal for non-real-time workloads: nightly content generation, bulk classification, embedding large datasets.
- **Client-side batching.** Group related queries into a single prompt when possible (e.g., "Classify all of these 10 items:" instead of 10 separate calls).

### Cost Monitoring and Alerting

Track costs at multiple granularities:

```
Per-Request:  model, input_tokens, output_tokens, cost_usd
Per-Feature:  SUM(cost) GROUP BY feature_name
Per-User:     SUM(cost) GROUP BY user_id
Per-Model:    SUM(cost) GROUP BY model
Per-Day:      SUM(cost) GROUP BY date
```

Set alerts for:
- Daily cost exceeding 2x the trailing 7-day average.
- Single-user cost exceeding threshold (abuse detection).
- Per-feature cost spikes (regression or misuse).
- Token-per-request anomalies (prompt injection stuffing the context).

---

## Rate Limiting and Quota Management

### Provider Rate Limits

Every provider imposes limits. Exceeding them returns HTTP 429 with a `Retry-After` header.

| Provider | Tier | RPM | TPM |
|---|---|---|---|
| OpenAI | Tier 1 | 500 | 30,000 |
| OpenAI | Tier 3 | 5,000 | 600,000 |
| OpenAI | Tier 5 | 10,000 | 10,000,000 |
| Anthropic | Build (Tier 1) | 50 | 40,000 |
| Anthropic | Build (Tier 4) | 4,000 | 400,000 |
| Google | Free | 15 | N/A |
| Google | Pay-as-you-go | 1,000 | 4,000,000 |

These limits apply per-model, per-organization. Check your provider dashboard for current limits.

### Client-Side Rate Limiting

Implement your own rate limiting to stay under provider limits and protect your system.

**Token bucket algorithm:**
- A bucket holds up to `max_tokens` tokens.
- Tokens refill at `rate` tokens per second.
- Each request attempts to consume a token. If the bucket is empty, the request waits or is rejected.
- Simple, memory-efficient, handles bursts gracefully.

**Sliding window:**
- Track request timestamps in a rolling window.
- Count requests in the last 60 seconds.
- More accurate than token bucket but requires more memory.

### Priority Queues

Not all requests are equal. Implement priority levels:

```
Priority 1 (Critical):  Real-time user requests, interactive chat
Priority 2 (High):      Triggered workflows, webhook responses
Priority 3 (Normal):    Background processing, scheduled tasks
Priority 4 (Low):       Batch jobs, analytics, experimentation
```

When approaching rate limits, process higher-priority requests first and queue or drop lower-priority ones.

### Exponential Backoff with Jitter

When you receive a 429 or 5xx error:

```
attempt = 0
while attempt < max_retries:
    try:
        response = call_api()
        return response
    except RateLimitError:
        wait = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0, wait)
        sleep(jitter)
        attempt += 1
```

**Why jitter matters:** Without jitter, all clients that hit a rate limit simultaneously will retry simultaneously, causing a thundering herd. Jitter spreads retries across time.

### API Key Rotation

For high-volume production systems:
- Maintain a pool of API keys across multiple accounts or organizations.
- Route requests round-robin or by load.
- Monitor per-key usage to avoid hitting per-key limits.
- Rotate keys on a schedule for security.
- Caution: check provider ToS — some prohibit circumventing rate limits via multiple accounts.

---

## Observability

### What to Log

Every LLM call in production should emit a structured log entry:

```json
{
  "request_id": "req-abc-123",
  "timestamp": "2025-03-15T10:30:00Z",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "feature": "customer_support_chat",
  "user_id": "user-456",
  "input_tokens": 1250,
  "output_tokens": 340,
  "total_tokens": 1590,
  "cost_usd": 0.0089,
  "latency_ms": 2340,
  "ttft_ms": 180,
  "status": "success",
  "temperature": 0.7,
  "max_tokens": 1024,
  "tools_called": ["search_kb"],
  "cache_hit": false,
  "prompt_hash": "sha256:abc123...",
  "response_hash": "sha256:def456..."
}
```

Store the full prompt and response separately (they can be large) linked by `request_id`. Use a tiered storage strategy: hot (7 days), warm (30 days), cold (1 year) for compliance.

### Key Metrics

**Latency metrics:**
- **TTFT (Time to First Token).** Measures how quickly the user sees initial output. Target: < 500ms for interactive use.
- **Total latency.** End-to-end time including all tokens. Depends on output length.
- **Tokens per second.** Generation throughput. Useful for comparing providers and models.
- **P50/P95/P99 latencies.** Tail latencies matter. A P99 of 30s means 1 in 100 users waits 30 seconds.

**Reliability metrics:**
- **Error rate.** Percentage of requests that fail (4xx, 5xx, timeouts). Target: < 0.1%.
- **Timeout rate.** Requests that exceed your deadline. Separate from provider errors.
- **Retry rate.** How often requests need retries. High retry rates indicate capacity issues.

**Cost metrics:**
- **Cost per request.** Average and P95.
- **Cost per feature.** Which product features are expensive?
- **Cost per user.** Detect abuse and plan pricing.
- **Daily/weekly burn rate.** For budgeting and anomaly detection.

**Quality metrics:**
- **Structured output parse success rate.** What percentage of responses parse correctly?
- **User feedback scores.** Thumbs up/down, ratings.
- **Eval scores.** Automated quality scores from your eval pipeline.

### Distributed Tracing for Multi-Step Pipelines

Complex LLM applications involve multiple steps. Tracing ties them together:

```
Trace: user-query-abc
│
├── Span: intent_classification (12ms, 50 tokens)
│   └── model: gpt-4o-mini, cost: $0.00004
│
├── Span: rag_retrieval (85ms)
│   ├── Span: embed_query (15ms, 8 tokens)
│   ├── Span: vector_search (40ms, 5 results)
│   └── Span: rerank (30ms, 5 → 3 results)
│
├── Span: llm_generation (2100ms, 1200 in + 350 out tokens)
│   └── model: claude-sonnet-4-20250514, cost: $0.0089
│
├── Span: safety_check (45ms)
│   └── flagged: false
│
└── Total: 2242ms, cost: $0.0089
```

Use OpenTelemetry spans for vendor-neutral tracing. Annotate spans with:
- Model and provider
- Token counts and cost
- Cache hit/miss
- Quality scores (if available)

### Observability Tools

| Tool | Type | Strengths |
|---|---|---|
| LangSmith | LLM-native | Deep prompt debugging, evals, datasets |
| Helicone | LLM proxy | Zero-code integration, cost tracking |
| Braintrust | Eval + logging | Strong eval framework, prompt playground |
| OpenTelemetry + Grafana | General observability | Vendor-neutral, integrates with existing infra |
| Datadog LLM Observability | APM extension | Integrates with existing Datadog setup |
| Custom (OTel + DB) | DIY | Full control, no vendor lock-in |

For most teams: start with a lightweight LLM-specific tool (Helicone or Braintrust) and export metrics to your existing observability stack.

---

## Structured Output Reliability

### The Problem

You ask for JSON. The model returns:

```
Sure! Here's the JSON you requested:

```json
{"name": "Alice", "age": 30}
```

Hope that helps!
```

Your JSON parser throws. In production, this is unacceptable.

### Provider-Enforced JSON Mode

The most reliable approach. The provider constrains token generation to valid JSON.

**OpenAI:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
)
```

**Anthropic (via tool use):**
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[...],
    tools=[{
        "name": "extract_user_info",
        "description": "Extract user information",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_user_info"}
)
```

### Retry with Feedback

When parsing fails, send the error back to the model:

```
System: You must respond with valid JSON matching this schema: {...}
User: Extract info from: "Alice is 30 years old"
Assistant: {"name": "Alice", "age": "thirty"}  <-- wrong type
System: JSON validation error: "age" must be integer, got string "thirty". Fix it.
Assistant: {"name": "Alice", "age": 30}  <-- fixed
```

This works surprisingly well. Most models self-correct on the first retry. Limit retries to 2-3 attempts.

### Lenient Parsing

Before retrying (which costs tokens), try to salvage the response:

1. Strip markdown code fences (` ```json ... ``` `).
2. Extract JSON from surrounding text (regex for `{...}` or `[...]`).
3. Fix common errors: trailing commas, single quotes, unquoted keys.
4. Try `json5` or `pyjson5` for relaxed JSON parsing.

### Defense in Depth

Layer these strategies:

```
1. Prompt: "Respond with valid JSON matching this schema: ..."
2. Provider: Use response_format or tool_choice to enforce JSON
3. Parse: json.loads() with lenient pre-processing
4. Validate: Pydantic model validation
5. Retry: On failure, send error back to model (max 2 retries)
6. Fallback: Return structured error to caller
```

In practice, with provider-enforced JSON mode, parse failures drop below 0.1%. Without it, expect 2-10% failure rates depending on the model and prompt.

---

## Latency Optimization

### Optimization Hierarchy

Ordered by effort-to-impact ratio:

| Technique | Perceived Latency | Actual Latency | Effort |
|---|---|---|---|
| Enable streaming | Massive reduction | No change | Low |
| Response caching | Eliminates (on hit) | Eliminates (on hit) | Low |
| Prompt caching (provider) | Moderate reduction | Moderate reduction | Low |
| Smaller model for simple tasks | Moderate reduction | Significant reduction | Medium |
| Reduce prompt length | Minor reduction | Minor reduction | Medium |
| Parallel tool calls | No change per call | Significant for agents | Medium |
| Speculative execution | No change | Moderate reduction | High |
| Self-hosted inference | Variable | Variable | High |
| Edge deployment | Eliminates network | Depends on hardware | Very High |

### Streaming for Perceived Latency

Even though total generation time is unchanged, streaming transforms the user experience:

- Without streaming: 8 seconds of nothing, then the full response.
- With streaming: first token in 200ms, response "types" in real-time.

Streaming is the single most impactful latency optimization for user-facing applications.

### Parallel Execution

For agentic workflows that require multiple independent tool calls or LLM calls:

```
Sequential:
  classify(query)  →  search(query)  →  generate(results)
  200ms               300ms              2000ms
  Total: 2500ms

Parallel where possible:
  classify(query)  ─┐
  200ms             ├──→  generate(results)
  search(query)   ─┘     2000ms
  300ms
  Total: 2300ms
```

Many providers support parallel tool calls natively. For custom orchestration, use `asyncio.gather()` or equivalent.

### Speculative Execution

Start downstream work before the LLM response is confirmed:

- Begin RAG retrieval while the user is still typing.
- Pre-fetch tool results for likely tool calls.
- Start rendering a response skeleton before the LLM finishes.

Risk: wasted computation if the speculation is wrong. Use for high-value, predictable flows.

---

## Error Handling

### Retry Strategy

```
+----------+     +--------+     +---------+     +----------+
| API Call | --> | Failed | --> | Retry?  | --> | Backoff  |
+----------+     +--------+     +---------+     +----------+
                                    |                |
                              No retries left    Wait, retry
                                    |
                              +-----------+
                              | Fallback  |
                              +-----------+
```

**Retryable errors:**
- 429 (Rate Limited) -- always retry with backoff.
- 500, 502, 503 (Server Error) -- retry up to 3 times.
- Timeout -- retry once, consider increasing timeout.
- Connection reset -- retry immediately once.

**Non-retryable errors:**
- 400 (Bad Request) -- your request is malformed. Fix it.
- 401 (Unauthorized) -- bad API key. Do not retry.
- 404 (Not Found) -- wrong model name or endpoint. Do not retry.

### Fallback Models

When your primary model is unavailable or overloaded, fall through to alternatives:

```
Primary:   Claude Sonnet → best quality for your use case
Fallback 1: GPT-4o      → comparable quality, different provider
Fallback 2: GPT-4o-mini → reduced quality, high availability
Fallback 3: Cached response → stale but better than nothing
Fallback 4: Graceful error  → "Service temporarily unavailable"
```

Implement fallbacks per feature. A support chatbot might accept lower quality from a fallback model. A medical diagnosis tool might not -- it should return an error rather than use a weaker model.

### Circuit Breaker Pattern

Prevent cascading failures when a provider is down:

```
States:
  CLOSED:     Normal operation. Requests go through.
  OPEN:       Provider is down. Requests fail fast (no API call).
  HALF-OPEN:  After cooldown, allow one test request through.

Transitions:
  CLOSED → OPEN:      After N consecutive failures (e.g., 5).
  OPEN → HALF-OPEN:   After cooldown period (e.g., 30 seconds).
  HALF-OPEN → CLOSED: If test request succeeds.
  HALF-OPEN → OPEN:   If test request fails.
```

Benefits:
- Prevents wasting time and money on calls that will fail.
- Reduces load on a struggling provider, giving it time to recover.
- Enables faster failover to backup providers.

### Graceful Degradation

When LLM calls fail and no fallback is available:

- **Cached responses.** Return a stale cached response with a "this may be outdated" warning.
- **Rule-based fallback.** For classification or routing, fall back to keyword matching or business rules.
- **Reduced functionality.** Disable the LLM-powered feature and show a static alternative.
- **Queue for later.** Accept the request, queue it, and process when the provider recovers. Notify the user asynchronously.
- **Honest error messages.** "Our AI assistant is temporarily unavailable. Here's our FAQ page." is better than a cryptic error.

---

## Interview Angle

### Common Interview Questions

1. "How would you handle an LLM provider outage in production?"
   - Circuit breaker, fallback models, cached responses, graceful degradation.

2. "Walk me through how you'd optimize costs for an LLM application doing 1M requests/day."
   - Model routing, caching, prompt optimization, batch processing. Show the math.

3. "How do you monitor an LLM application in production?"
   - Structured logging, key metrics (TTFT, error rate, cost), distributed tracing, alerting.

4. "Design a streaming architecture for a chatbot."
   - SSE, token buffering, partial JSON handling, backpressure, error recovery.

5. "How would you handle rate limits from your LLM provider?"
   - Token bucket, priority queues, backoff with jitter, key rotation, capacity planning.

### What Interviewers Are Looking For

- **Systems thinking.** Can you reason about the full stack: client, server, provider, and the interactions between them?
- **Production experience.** Have you actually operated LLM systems? Real numbers, real failure modes, real tradeoffs.
- **Cost awareness.** Can you estimate costs and optimize them without sacrificing quality?
- **Resilience patterns.** Do you know how to handle failures gracefully, not just the happy path?
- **Observability mindset.** Can you debug a production issue from metrics and logs?

The bar for senior engineers is not "can you call an API" -- it's "can you build a reliable, cost-effective, observable system that handles millions of requests?"
