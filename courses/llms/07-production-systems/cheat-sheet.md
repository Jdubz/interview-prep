# Module 07 — Cheat Sheet: Production Systems Quick Reference

---

## Cost Estimation Formulas

### Per-Request Cost

```
cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)
```

### Daily Cost

```
daily_cost = requests_per_day * avg_cost_per_request
```

### Worked Example

```
Feature: Customer support chatbot
Requests/day:     10,000
Avg input tokens:  2,000 (system prompt + history + user message)
Avg output tokens: 400

Using Claude 3.5 Sonnet ($3/1M input, $15/1M output):
  Input cost:  10,000 * 2,000 * $3 / 1,000,000   = $60/day
  Output cost: 10,000 * 400 * $15 / 1,000,000     = $60/day
  Total:                                           = $120/day = $3,600/month

Using GPT-4o-mini ($0.15/1M input, $0.60/1M output):
  Input cost:  10,000 * 2,000 * $0.15 / 1,000,000 = $3/day
  Output cost: 10,000 * 400 * $0.60 / 1,000,000   = $2.40/day
  Total:                                           = $5.40/day = $162/month

Savings from routing 80% to mini, 20% to Sonnet:
  0.80 * $5.40 + 0.20 * $120 = $4.32 + $24.00 = $28.32/day = $850/month
  (vs $3,600/month all-Sonnet → 76% savings)
```

### Token-to-Word Approximation

```
1 token  ≈ 0.75 words (English)
1 word   ≈ 1.33 tokens
1 page   ≈ 300 words ≈ 400 tokens
1M tokens ≈ 750K words ≈ 2,500 pages
```

---

## Caching Decision Tree

```
Is the query deterministic (temperature=0)?
├── Yes → Is the exact same query likely to repeat?
│         ├── Yes → EXACT MATCH CACHE (hash-based, highest hit rate)
│         └── No  → Are similar queries likely?
│                   ├── Yes → SEMANTIC CACHE (embedding similarity)
│                   └── No  → Are prompts sharing a long prefix?
│                             ├── Yes → PROMPT CACHE (provider-side)
│                             └── No  → No caching benefit
└── No  → Is the query classification/extraction (structured output)?
          ├── Yes → EXACT MATCH CACHE (output is deterministic in practice)
          └── No  → Is freshness acceptable with TTL?
                    ├── Yes → EXACT MATCH CACHE with short TTL
                    └── No  → No response caching (consider prompt caching only)
```

### Cache Type Comparison

| Cache Type | Hit Rate | Latency Added (miss) | Cost Savings | Implementation |
|---|---|---|---|---|
| Exact match | Low-Medium | ~1ms (hash lookup) | High per hit | Redis/in-memory |
| Semantic | Medium | 10-50ms (embedding + search) | High per hit | Vector store |
| Prompt (provider) | High | 0ms (automatic) | 50-90% input cost | Configuration |

---

## Latency Optimization Checklist

### Quick Wins (hours to implement)

- [ ] Enable streaming for all user-facing responses
- [ ] Enable provider prompt caching (structure prompts with static prefix)
- [ ] Add exact-match response caching for repeated queries
- [ ] Set appropriate `max_tokens` to prevent unnecessarily long responses
- [ ] Use smaller models for simple tasks (classification, extraction, routing)

### Medium Effort (days to implement)

- [ ] Reduce prompt length (audit system prompts, trim few-shot examples)
- [ ] Implement parallel tool execution for agents
- [ ] Add semantic caching for similar queries
- [ ] Implement model routing (small/medium/large by task complexity)
- [ ] Pre-compute and cache embeddings for static content
- [ ] Use async/concurrent API calls for independent operations

### High Effort (weeks to implement)

- [ ] Self-host models for latency-sensitive, high-volume features
- [ ] Implement speculative execution (pre-fetch likely tool results)
- [ ] Edge deployment for offline/ultra-low-latency use cases
- [ ] Build a cascade architecture (try small model first, escalate)
- [ ] Implement custom batching for throughput-critical pipelines

---

## Provider Rate Limits Reference (Mid-2025)

### OpenAI

| Tier | Requirement | RPM | TPM (most models) |
|---|---|---|---|
| Free | Default | 3 | 40,000 |
| Tier 1 | $5 paid | 500 | 30,000 |
| Tier 2 | $50 paid, 7+ days | 5,000 | 450,000 |
| Tier 3 | $100 paid, 7+ days | 5,000 | 600,000 |
| Tier 4 | $250 paid, 14+ days | 10,000 | 800,000 |
| Tier 5 | $1,000 paid, 30+ days | 10,000 | 10,000,000 |

### Anthropic

| Tier | Requirement | RPM | Input TPM | Output TPM |
|---|---|---|---|---|
| Build Tier 1 | $0 credit | 50 | 40,000 | 8,000 |
| Build Tier 2 | $40 credit | 1,000 | 80,000 | 16,000 |
| Build Tier 3 | $200 credit | 2,000 | 160,000 | 32,000 |
| Build Tier 4 | $400 credit | 4,000 | 400,000 | 80,000 |
| Scale | Custom | Custom | Custom | Custom |

### Rate Limit Response Handling

```
HTTP 429 Response Headers:
  retry-after: 2              ← seconds to wait
  x-ratelimit-limit-requests: 500
  x-ratelimit-remaining-requests: 0
  x-ratelimit-reset-requests: 2025-03-15T10:30:05Z
```

---

## Observability Must-Haves Checklist

### Per-Request Logging

- [ ] Request ID (for correlation)
- [ ] Timestamp
- [ ] Model name and version
- [ ] Provider
- [ ] Feature/endpoint name
- [ ] User ID
- [ ] Input token count
- [ ] Output token count
- [ ] Cost (USD)
- [ ] TTFT (time to first token)
- [ ] Total latency
- [ ] Status (success/error)
- [ ] Error type and message (if applicable)
- [ ] Cache hit/miss
- [ ] Tool calls (names and durations)
- [ ] Prompt hash (for grouping identical prompts)

### Dashboard Metrics

| Metric | Aggregation | Alert Threshold |
|---|---|---|
| Error rate | 5-min rolling | > 5% |
| P95 latency | 5-min rolling | > 10s |
| TTFT P95 | 5-min rolling | > 2s |
| Cost per hour | Hourly sum | > 2x trailing avg |
| Requests per minute | 1-min count | > 80% of rate limit |
| Cache hit rate | Hourly ratio | < 20% (if caching enabled) |
| Token per request (avg) | Hourly avg | > 2x baseline |

### Distributed Tracing Structure

```
Trace
├── Span: request_received
├── Span: input_validation
├── Span: cache_lookup (hit/miss)
├── Span: context_assembly
│   ├── Span: embedding
│   └── Span: vector_search
├── Span: llm_call
│   ├── Attribute: model
│   ├── Attribute: input_tokens
│   ├── Attribute: output_tokens
│   ├── Attribute: ttft_ms
│   └── Attribute: cost_usd
├── Span: output_validation
└── Span: response_sent
```

---

## Error Handling Patterns

| Error | HTTP Code | Retry? | Strategy |
|---|---|---|---|
| Rate limited | 429 | Yes | Exponential backoff with jitter, respect Retry-After |
| Server error | 500 | Yes | Retry up to 3x with backoff |
| Bad gateway | 502 | Yes | Retry up to 3x, consider provider failover |
| Service unavailable | 503 | Yes | Retry with longer backoff, activate circuit breaker |
| Timeout | N/A | Yes | Retry once with increased timeout |
| Bad request | 400 | No | Fix the request (prompt too long, invalid params) |
| Auth failure | 401 | No | Check API key, do not retry |
| Model not found | 404 | No | Check model name, do not retry |
| Content filtered | 400 | Maybe | Rephrase input if appropriate |
| Context overflow | 400 | No | Truncate input, reduce context |
| Invalid JSON output | N/A | Yes | Retry with error feedback, lenient parsing |

### Circuit Breaker Configuration

```
failure_threshold:   5 consecutive failures
reset_timeout:       30 seconds
half_open_max:       1 test request
monitor_window:      60 seconds
```

---

## Model Hosting Comparison

| | API (OpenAI/Anthropic) | Self-Hosted (vLLM) | Edge (llama.cpp) |
|---|---|---|---|
| **Setup time** | Minutes | Days | Hours |
| **Models available** | Provider's only | Any open-weight | Small open-weight |
| **Max model size** | N/A (provider handles) | GPU-limited | Device RAM-limited |
| **Cost model** | Pay per token | Fixed GPU cost | Free (user hardware) |
| **Cost at 1M tok/day** | $3-150 | $50-150 (GPU) | $0 |
| **Cost at 100M tok/day** | $300-15,000 | $50-150 (GPU) | $0 |
| **Latency (TTFT)** | 100-500ms | 50-200ms | 10-100ms |
| **Throughput** | Rate limited | GPU-limited | Single user |
| **Data privacy** | Data leaves network | Data stays local | Data on device |
| **Reliability** | Provider SLA | Your SLA | Device dependent |
| **Scaling** | Automatic | Manual/auto-scale | N/A |
| **Best for** | Most use cases | High volume, privacy | Offline, privacy, mobile |

---

## Infrastructure Sizing Guidelines

### Self-Hosted GPU Requirements

| Model Size | Min VRAM (INT4) | Min VRAM (FP16) | Recommended GPU |
|---|---|---|---|
| 3B | 3 GB | 6 GB | T4, L4 |
| 7-8B | 6 GB | 16 GB | T4, L4, A10G |
| 13B | 10 GB | 26 GB | A10G, A100 40GB |
| 34B | 20 GB | 68 GB | A100 80GB |
| 70B | 40 GB | 140 GB | 2x A100 80GB, 2x H100 |
| 405B | 220 GB | 810 GB | 8x H100 |

### Context Window Memory Impact

KV cache memory grows linearly with sequence length and batch size:

```
KV cache per token ≈ 2 * num_layers * hidden_dim * 2 bytes (FP16)

Example (Llama 3.1 70B):
  80 layers * 8192 hidden_dim * 2 (K+V) * 2 bytes = 2.6 MB per token
  8K context = ~20 GB KV cache per request
  32 concurrent requests at 8K = ~640 GB KV cache (PagedAttention helps significantly)
```

### Provider Tier Recommendation by Scale

| Daily Requests | Daily Tokens (approx) | Recommended Tier |
|---|---|---|
| < 100 | < 500K | Free / Tier 1 |
| 100 - 1,000 | 500K - 5M | Tier 2 |
| 1,000 - 10,000 | 5M - 50M | Tier 3 |
| 10,000 - 100,000 | 50M - 500M | Tier 4-5 / Scale |
| 100,000+ | 500M+ | Enterprise / Scale / Self-host |

---

## Key Numbers to Remember

| Metric | Value |
|---|---|
| Token generation speed (API) | 30-100 tokens/sec |
| TTFT target (interactive) | < 500ms |
| TTFT target (batch) | N/A |
| Cache hit rate (FAQ bots) | 30-60% |
| Cache hit rate (unique convos) | < 5% |
| Provider uptime SLA | 99.9% (typical) |
| Circuit breaker threshold | 5 consecutive failures |
| Exponential backoff base | 1 second, max 60 seconds |
| Prompt cache min length (Anthropic) | 1024 tokens (Sonnet) |
| Prompt cache savings (Anthropic) | 90% input cost reduction |
| Prompt cache savings (OpenAI) | 50% input cost reduction |
| Model routing cost savings | 50-80% with 70-80% small model routing |
| Batch API discount (OpenAI) | 50% |
