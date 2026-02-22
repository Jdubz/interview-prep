# Production Concerns

## Streaming

### Why Stream?

LLM generation is slow (10-100+ tokens/second). Streaming returns tokens as they're generated, dramatically improving perceived latency.

| Metric | Without Streaming | With Streaming |
|---|---|---|
| Time to first visible token | 2-10s (full generation) | 100-500ms |
| User perception | "Is it broken?" | "It's thinking..." |

### How It Works

Providers use **Server-Sent Events (SSE)** — a one-way HTTP stream where the server pushes chunks:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: {"choices":[{"delta":{"content":"!"}}]}
data: [DONE]
```

### Implementation Considerations

- **Partial JSON:** If you're expecting structured output, you can't parse until the stream completes. Buffer the full response, then parse.
- **Tool calls in streams:** Tool call arguments arrive incrementally — you need to buffer and reconstruct the full call.
- **Error handling:** Streams can fail mid-response. Handle partial results gracefully.
- **Backpressure:** If your consumer (UI, API client) is slow, you may need to buffer.

---

## Caching

### Response Caching

Cache complete responses for identical prompts:

```
Hash(system_prompt + user_messages + model + temperature) → cached response
```

**When it works:** Deterministic queries (temperature=0), repeated questions, batch processing.

**Cache invalidation:** Time-based TTL, or invalidate when underlying data changes (for RAG).

### Semantic Caching

For similar (not identical) queries, use embedding similarity:

```
1. Embed the new query
2. Search cache for queries with cosine similarity > threshold
3. If found, return the cached response
4. If not, call the LLM and cache the result
```

**Tradeoff:** Cache hit rate vs. response accuracy. A threshold too low returns stale/wrong results.

### Prompt Caching (Provider Feature)

Some providers cache the prompt prefix server-side:

- **Anthropic:** Automatic prompt caching — repeated prefixes (system prompts, large contexts) are cached and charged at reduced rates
- **OpenAI:** Cached prompt prefixes on longer prompts

This is especially impactful for RAG (large context) and agents (repeated system prompt + tool definitions).

---

## Cost Optimization

### Model Selection Per Task

Don't use your most expensive model for everything:

| Task | Recommended Tier | Why |
|---|---|---|
| Intent classification | Small (GPT-4o-mini, Haiku) | Simple task, high volume |
| Data extraction | Medium (GPT-4o, Sonnet) | Needs accuracy but not creativity |
| Complex reasoning | Large (o1, Opus) | Requires deep reasoning |
| Embeddings | Embedding model | 100x cheaper than LLMs |

### Token Reduction

- **Concise system prompts:** Every token costs money at scale
- **Summarize conversation history:** Don't pass 50 turns when a summary works
- **Selective RAG:** Retrieve fewer, more relevant chunks instead of stuffing the context
- **Output limits:** Set `max_tokens` appropriately — don't let the model ramble

### Cost Monitoring

Track at multiple levels:
- Per-request (for debugging)
- Per-feature (which features are expensive?)
- Per-user (detect abuse)
- Per-model (compare alternatives)

### Rough Cost Math

```
1M tokens ≈ 750K words ≈ 3,000 pages of text

At $3/1M input tokens, $15/1M output tokens (GPT-4o approximate):
- 1,000 customer support interactions/day
- Average 2K input + 500 output tokens each
- Daily cost: (2M × $3 + 500K × $15) / 1M = $6 + $7.50 = $13.50/day
```

---

## Rate Limiting

### Provider Rate Limits

All providers impose limits:
- **Requests per minute (RPM)**
- **Tokens per minute (TPM)**
- **Tokens per day (TPD)**

### Client-Side Rate Limiting

Protect your own system:

```python
# Token bucket pattern
# - Tokens accumulate at a fixed rate
# - Each request consumes tokens
# - If bucket is empty, queue or reject
```

### Best Practices

- **Exponential backoff:** On 429 errors, wait 2^n seconds
- **Request queuing:** Buffer requests and process at a sustainable rate
- **Priority queues:** Real-time user requests > background batch jobs
- **Multiple API keys:** Distribute load across keys/accounts for high volume

---

## Evaluation (Evals)

### Why Evals Matter

Prompt changes, model updates, and provider changes can silently degrade quality. Evals catch regressions before users do.

### Building an Eval Pipeline

```
Test Cases → Run Prompts → Score Outputs → Compare to Baseline → Report
```

1. **Curate test cases:** (input, expected_output) pairs from real usage
2. **Automated scoring:**
   - **Exact match:** For classification
   - **Contains/regex:** For structured output
   - **LLM-as-judge:** Use a powerful model to grade another model's output
   - **Embedding similarity:** Is the output semantically close to expected?
3. **Baseline comparison:** Always compare against your current production version
4. **Track over time:** Dashboard of eval scores per prompt version / model

### Eval-Driven Development

```
1. Define success criteria and test cases FIRST
2. Write initial prompt
3. Run evals → establish baseline
4. Iterate on prompt/model
5. Run evals → compare to baseline
6. Only deploy if evals improve (or hold steady)
```

---

## Safety

### Input Safety

- **Content filtering:** Reject or flag harmful/inappropriate inputs
- **Prompt injection defense:** Delimiters, input sanitization, instruction hierarchy
- **Input length limits:** Prevent context stuffing attacks
- **PII detection:** Redact sensitive data before sending to the LLM

### Output Safety

- **Content filtering:** Check outputs for harmful, biased, or inappropriate content
- **PII in outputs:** Ensure the model doesn't leak PII from training data or context
- **Schema validation:** Verify structured outputs match expected schemas
- **Hallucination checks:** Cross-reference factual claims for high-stakes applications

### Architectural Safety

- **Principle of least privilege:** Only give the model access to tools it needs
- **Sandboxing:** Run model-generated code in sandboxes
- **Audit logging:** Log all inputs, outputs, and tool calls
- **Human oversight:** Human-in-the-loop for high-stakes actions
- **Kill switches:** Ability to disable LLM features quickly

---

## Observability

### What to Log

| What | Why |
|---|---|
| Full prompt (input) | Debug quality issues |
| Full response (output) | Audit and compliance |
| Model + parameters | Reproduce results |
| Latency (TTFT, total) | Performance monitoring |
| Token counts (in/out) | Cost tracking |
| Tool calls + results | Debug agent behavior |
| Error details | Reliability monitoring |

### Key Metrics

- **Latency:** Time-to-first-token (TTFT), total generation time
- **Error rate:** API failures, timeouts, malformed responses
- **Cost:** Tokens consumed, dollars spent
- **Quality:** Eval scores, user feedback, thumbs up/down
- **Usage:** Requests per feature, per user, per model

### Tracing

For multi-step pipelines (RAG, agents), trace the full execution:

```
Request ID: req-123
├── Embed query: 50ms, 12 tokens
├── Vector search: 80ms, 5 results
├── Rerank: 120ms, 5 → 3 results
├── LLM generation: 1200ms, 450 input + 200 output tokens
└── Total: 1450ms, cost: $0.002
```

Tools: LangSmith, Helicone, Braintrust, custom (OpenTelemetry spans).

---

## Structured Output Reliability

### The Problem

LLMs don't always produce valid JSON, even when asked. In production, you need guarantees.

### Solutions (Ranked by Reliability)

1. **Provider-enforced schemas:** OpenAI `response_format: { type: "json_schema" }`, Anthropic tool use for structured output — most reliable
2. **Constrained decoding:** Open-source tools like Outlines or LMQL constrain token generation to match a grammar
3. **Retry with validation:** Parse the output, if invalid, send back the error and ask the model to fix it
4. **Lenient parsing:** Strip markdown code fences, fix common JSON errors, extract JSON from text

### Defense in Depth

```python
# 1. Ask for JSON in the prompt
# 2. Use provider-enforced JSON mode if available
# 3. Parse and validate with a schema (e.g., Pydantic)
# 4. If validation fails, retry once with the error message
# 5. If retry fails, return a structured error to the caller
```

---

## Latency Optimization

| Technique | Impact | Complexity |
|---|---|---|
| Streaming | Perceived latency ↓↓↓ | Low |
| Smaller model for simple tasks | Real latency ↓↓ | Low |
| Prompt length reduction | Real latency ↓ | Medium |
| Parallel tool calls | Agent latency ↓↓ | Medium |
| Caching | Latency ↓↓↓ (cache hits) | Medium |
| Speculative execution | Latency ↓ | High |
| Edge deployment (open source) | Network latency ↓ | High |
