# Module 07 — Deep Dive: Production Infrastructure

Advanced topics for engineers building and operating LLM inference infrastructure at scale.

---

## Inference Infrastructure: Self-Hosting

### When Self-Hosting Beats API

| Factor | API | Self-Hosted |
|---|---|---|
| Setup time | Minutes | Days to weeks |
| Cost at low volume | Lower (pay per token) | Higher (fixed GPU cost) |
| Cost at high volume | Higher | Lower (amortized GPU) |
| Latency | Network + provider queue | Local, predictable |
| Data privacy | Data leaves your network | Data stays on your infra |
| Model selection | Provider's models only | Any open-weight model |
| Customization | None (fine-tuning limited) | Full control |
| Reliability | Provider SLA | Your SLA |
| Scaling | Automatic | You manage it |

**Break-even analysis.** A single A100 80GB GPU costs roughly $2/hour on cloud ($1,440/month). Running Llama 3.1 70B (quantized) at ~30 tokens/second, that's ~77M output tokens/month. At GPT-4o output pricing ($10/1M tokens), that's $770/month equivalent. Self-hosting starts winning when:
- You have consistent, high-volume traffic (> 50M tokens/month).
- You need data privacy guarantees.
- You need customization (fine-tuned models, custom decoding).
- Latency predictability matters more than raw speed.

### Inference Engines

**vLLM**
The de facto standard for high-throughput self-hosted inference.
- PagedAttention for efficient GPU memory management.
- Continuous batching for maximum throughput.
- Tensor parallelism for multi-GPU serving.
- OpenAI-compatible API server built in.
- Supports most popular model architectures.

```bash
# Serve a model with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

**Text Generation Inference (TGI)**
Hugging Face's inference server. Production-tested at scale.
- Flash Attention 2 integration.
- Token streaming out of the box.
- Quantization support (GPTQ, AWQ, EETQ).
- Good Docker support for deployment.

```bash
# Serve with TGI
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-70B-Instruct \
    --num-shard 4 \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

**Ollama**
Optimized for local development and edge deployment.
- Simple CLI: `ollama run llama3.1`.
- Automatic quantization and model management.
- REST API compatible with common tooling.
- Not designed for high-throughput production serving.

### GPU Selection

| GPU | VRAM | FP16 TFLOPS | Typical Use | Cloud $/hr |
|---|---|---|---|---|
| A10G | 24 GB | 31.2 | 7B-13B models | ~$1.00 |
| L4 | 24 GB | 30.3 | 7B-13B models | ~$0.80 |
| A100 40GB | 40 GB | 77.9 | 13B-34B models | ~$1.50 |
| A100 80GB | 80 GB | 77.9 | 34B-70B (quantized) | ~$2.00 |
| H100 80GB | 80 GB | 267.6 | 70B+ models | ~$3.50 |
| H200 141GB | 141 GB | 267.6 | 70B (full precision) | ~$4.50 |

**Rule of thumb for VRAM requirements:**
- FP16: ~2 bytes per parameter. A 70B model needs ~140 GB VRAM.
- INT8 quantization: ~1 byte per parameter. 70B needs ~70 GB.
- INT4 quantization: ~0.5 bytes per parameter. 70B needs ~35 GB.
- Add 10-20% overhead for KV cache and activations.

### Throughput Estimation

Throughput depends on model size, GPU, quantization, batch size, and sequence length.

Rough benchmarks (tokens/second output, high batch):

| Model | GPU | Quantization | Throughput (tokens/s) |
|---|---|---|---|
| Llama 3.1 8B | A10G | FP16 | 80-120 |
| Llama 3.1 8B | A10G | INT4 (AWQ) | 150-200 |
| Llama 3.1 70B | 4x A100 80GB | FP16 | 40-60 |
| Llama 3.1 70B | 2x H100 | INT4 (AWQ) | 80-120 |
| Llama 3.1 70B | 4x H100 | FP16 | 100-150 |
| Mixtral 8x7B | 2x A100 80GB | FP16 | 60-90 |

These are aggregate throughput with continuous batching at high utilization. Per-request latency will be higher at high batch sizes.

---

## Batching and Throughput

### Why Batching Matters

GPU utilization is the key to cost-effective inference. A single request uses a fraction of the GPU's compute capacity. Batching amortizes the fixed costs (weight loading, memory transfers) across multiple requests.

```
Single request:  GPU utilization ~10-20%
Batch of 8:      GPU utilization ~60-80%
Batch of 32:     GPU utilization ~90%+ (diminishing returns)
```

### Static Batching

Collect N requests, process them together, return all results.

```
Requests:  [r1, r2, r3, r4]  →  GPU processes batch  →  [resp1, resp2, resp3, resp4]
```

**Problem:** All requests in the batch must complete before any result is returned. A request generating 10 tokens waits for a request generating 500 tokens. This wastes GPU cycles (the short request's slots are idle) and increases latency for short requests.

### Continuous Batching

The modern approach used by vLLM, TGI, and other production engines.

```
Iteration 1: [r1(token 5), r2(token 3), r3(token 1), -----]
Iteration 2: [r1(token 6), r2(done!),    r3(token 2), r4(token 1)]  ← r2 finishes, r4 joins
Iteration 3: [r1(token 7), r5(token 1),  r3(token 3), r4(token 2)]  ← r5 fills the slot
```

Key properties:
- Requests enter and leave the batch independently.
- No request waits for another to finish.
- GPU utilization stays high because slots are immediately reused.
- Dramatically better throughput and latency compared to static batching.

### PagedAttention (vLLM)

The KV cache is the memory bottleneck in LLM inference. Each token generates a key-value pair that must be stored for attention computation. For long sequences with large batches, KV cache can consume more memory than the model weights.

**Traditional approach:** Pre-allocate a contiguous block of memory for each request's maximum possible sequence length. Wastes memory on short sequences.

**PagedAttention:** Borrows ideas from virtual memory in operating systems.
- KV cache is divided into fixed-size "pages" (blocks).
- Pages are allocated on demand as the sequence grows.
- Pages can be non-contiguous in physical memory.
- Completed sequences free their pages immediately.

```
Traditional allocation:
  Request 1: [████████████________________]  (12 tokens used, 16 wasted)
  Request 2: [████████████████████________]  (20 tokens used, 8 wasted)

PagedAttention:
  Request 1: [page1][page2][page3]           (12 tokens, 0 wasted)
  Request 2: [page4][page5][page6][page7][page8]  (20 tokens, ~0 wasted)
  Free pool: [page9][page10]...              (available for new requests)
```

Result: 2-4x higher throughput compared to naive memory management, because more requests fit in GPU memory simultaneously.

### Throughput vs. Latency Tradeoffs

```
          Throughput
              ▲
              │        ╭────────────
              │      ╱
              │    ╱
              │  ╱
              │╱
              └────────────────────► Batch Size

          Per-Request Latency
              ▲
              │                  ╱
              │                ╱
              │              ╱
              │          ╱
              │  ───────╱
              └────────────────────► Batch Size
```

- Increasing batch size improves throughput (up to GPU saturation).
- Increasing batch size also increases per-request latency (more compute per iteration).
- The sweet spot depends on your SLA: interactive chat needs low latency (small batches), batch processing needs high throughput (large batches).

---

## Speculative Decoding

### The Core Idea

Autoregressive generation is bottlenecked by sequential token generation. Each token requires a full forward pass through the model. Speculative decoding uses a small, fast "draft" model to propose multiple tokens, which the large "target" model verifies in parallel.

```
Draft model (small, fast):
  Proposes: ["The", "capital", "of", "France", "is", "Paris"]

Target model (large, accurate):
  Verifies in one forward pass:
  - "The" ✓
  - "capital" ✓
  - "of" ✓
  - "France" ✓
  - "is" ✓
  - "Paris" ✓  → Accept all 6 tokens

  6 tokens generated in ~1 forward pass of the target model
  instead of 6 sequential forward passes.
```

### When It Helps

Speculative decoding helps most when:
- **The draft model is good at predicting the target model's output.** For factual, predictable text (code, structured data), acceptance rates are high (70-90%). For creative text, acceptance rates are lower (40-60%).
- **The draft model is significantly faster.** A 7B draft model with a 70B target is a good pairing.
- **Latency matters more than throughput.** Speculative decoding improves per-request latency but may reduce total throughput (the target model verifies speculative tokens in addition to its normal work).

### Speedup Expectations

| Scenario | Draft Acceptance Rate | Effective Speedup |
|---|---|---|
| Code generation | 80-90% | 2-3x |
| Factual Q&A | 70-85% | 1.5-2.5x |
| Creative writing | 40-60% | 1.2-1.5x |
| Very creative/random | 20-40% | 1.0-1.2x (may not help) |

### Provider Support

- **Anthropic:** Not exposed as a user-facing feature (may be used internally).
- **OpenAI:** Not exposed directly. The `predicted_output` parameter in some APIs is related.
- **Google:** Speculative decoding available for Gemini on Vertex AI.
- **Self-hosted:** vLLM and TGI both support speculative decoding with configurable draft models.

---

## Edge Deployment

### Running Models On-Device

For latency-critical, privacy-sensitive, or offline use cases, run models directly on user devices.

| Framework | Platform | Best For |
|---|---|---|
| llama.cpp / GGUF | CPU, Mac, Linux, Windows | Broadest compatibility, CPU-optimized |
| Apple MLX | Apple Silicon (M1-M4) | Native Metal acceleration on Mac/iOS |
| ONNX Runtime | Cross-platform | Edge devices, Windows, mobile |
| MediaPipe (Google) | Android, iOS, Web | Mobile-first applications |
| MLC LLM | Cross-platform | Compiled models for specific hardware |

### Performance Expectations (On-Device)

| Device | Model | Quantization | Tokens/sec |
|---|---|---|---|
| MacBook Pro M3 Max (48GB) | Llama 3.1 8B | Q4_K_M | 40-60 |
| MacBook Pro M3 Max (48GB) | Llama 3.1 70B | Q4_K_M | 8-12 |
| MacBook Air M2 (16GB) | Llama 3.1 8B | Q4_K_M | 20-30 |
| iPhone 15 Pro (8GB) | Llama 3.2 3B | Q4_K_M | 10-15 |
| Pixel 8 (12GB) | Gemma 2B | INT4 | 8-12 |

### When to Deploy on Edge

**Good fit:**
- Privacy-critical applications (medical, legal, personal data).
- Offline functionality (field work, aircraft, submarines).
- Latency-critical (real-time autocomplete, local code assistance).
- Cost optimization at massive user scale (inference cost moves to user hardware).

**Poor fit:**
- Tasks requiring large models (> 13B parameters on most devices).
- High throughput requirements (edge devices serve one user at a time).
- Applications needing the latest frontier model capabilities.
- Users on low-end hardware.

---

## Multi-Model Architectures

### Router Pattern

A lightweight classifier routes requests to the appropriate specialist model.

```
                         ┌─────────────┐
                         │   Router    │
                         │  (small LM  │
                         │  or rules)  │
                         └──────┬──────┘
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Simple  │ │  Medium  │ │  Complex │
              │  Model   │ │  Model   │ │  Model   │
              │ (Haiku)  │ │ (Sonnet) │ │  (Opus)  │
              └──────────┘ └──────────┘ └──────────┘
```

**Router implementation options:**
- **Keyword heuristics.** Fast, no extra API call. Check for complexity indicators: "analyze", "compare", "step by step" route to the large model. Simple queries route to the small model.
- **Embedding classifier.** Embed the query, classify into complexity buckets using a trained classifier. More accurate, adds 10-20ms.
- **Small LLM as router.** Use a fast model (GPT-4o-mini, Haiku) to classify the required complexity. Most accurate, adds 200-500ms and cost.

### Cascade Pattern

Try the cheapest model first. Escalate if the response quality is insufficient.

```
                    ┌──────────────┐
                    │  Small Model │
                    │   (Haiku)    │
                    └──────┬───────┘
                           │
                    confidence > 0.8?
                     ╱           ╲
                   Yes            No
                    │              │
              Return response     │
                           ┌──────┴───────┐
                           │ Medium Model  │
                           │  (Sonnet)     │
                           └──────┬────────┘
                                  │
                           confidence > 0.8?
                            ╱           ╲
                          Yes            No
                           │              │
                     Return response     │
                                  ┌──────┴───────┐
                                  │ Large Model   │
                                  │   (Opus)      │
                                  └───────────────┘
```

**Confidence estimation methods:**
- **Log probabilities.** If available, use the model's token-level log probs. Low entropy = high confidence.
- **Self-reported confidence.** Ask the model to rate its own confidence (unreliable, but cheap).
- **Output validation.** Run the output through a validator. If it passes, accept; if not, escalate.
- **Heuristic checks.** Length, presence of hedging language ("I'm not sure"), refusals.

**Cascade economics:**
- If 70% of requests are handled by the small model, 25% by medium, 5% by large:
  - Average cost = 0.70 * $small + 0.25 * $medium + 0.05 * $large
  - Plus escalation overhead (duplicate processing for escalated requests)
- Net savings: typically 50-70% compared to always using the large model.

### Ensemble Approaches

Multiple models generate responses, a selector picks the best one.

```
            ┌──────────┐
    ┌──────►│ Model A  │──────┐
    │       └──────────┘      │
    │       ┌──────────┐      │     ┌──────────┐
Query ────►│ Model B  │──────┼────►│ Selector │──► Best Response
    │       └──────────┘      │     └──────────┘
    │       ┌──────────┐      │
    └──────►│ Model C  │──────┘
            └──────────┘
```

**When to use ensembles:**
- High-stakes decisions where quality justifies 3x cost.
- Tasks where different models have complementary strengths.
- A/B testing and model comparison in production.

**Selection strategies:**
- LLM-as-judge: a separate model evaluates the responses.
- Voting: for classification tasks, take the majority vote.
- Confidence-weighted: pick the response from the most confident model.

---

## Gateway and Proxy Patterns

### Why Use a Gateway

A gateway sits between your application and LLM providers, providing:

```
Your Application
       │
       ▼
┌──────────────────┐
│   LLM Gateway    │
│                  │
│  - Unified API   │
│  - Load balance  │
│  - Failover      │
│  - Rate limiting │
│  - Logging       │
│  - Caching       │
│  - Cost tracking │
└──────┬───────────┘
       │
  ┌────┼────┐
  ▼    ▼    ▼
 OAI  Anth  Goog
```

### Gateway Options

| Tool | Type | Key Features |
|---|---|---|
| LiteLLM | OSS proxy | OpenAI-compatible API for 100+ models, load balancing, fallbacks |
| Portkey | Managed service | Gateway + observability, caching, prompt management |
| Helicone | Managed proxy | Zero-code logging, caching, rate limiting |
| Custom (FastAPI) | DIY | Full control, no dependencies, build exactly what you need |

### LiteLLM Example

```python
# litellm provides a unified interface across providers
import litellm

# Same API, different providers
response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    fallbacks=["openai/gpt-4o", "openai/gpt-4o-mini"],
    # If Anthropic fails, try OpenAI, then mini
)
```

### Custom Gateway Considerations

For most teams, a lightweight custom gateway built on FastAPI or similar is the right choice:

**Must-have features:**
- Unified request/response format across providers.
- Structured logging of every request (see Observability section).
- Retry logic with exponential backoff.
- Per-provider health checks and circuit breakers.
- API key management and rotation.

**Nice-to-have features:**
- Response caching (exact and semantic).
- Rate limiting per client/feature.
- Cost tracking and budget enforcement.
- A/B testing (route percentage of traffic to different models).
- Prompt versioning and management.

---

## Token Budget Management

### The Problem

Every LLM has a context window limit. Your application must ensure the total tokens (input + output) fit within this limit. Exceeding it causes a hard error.

```
Context Window: 128,000 tokens
├── System prompt:        2,000 tokens (fixed)
├── Tool definitions:     1,500 tokens (fixed)
├── Few-shot examples:    1,000 tokens (fixed)
├── Conversation history: variable
├── RAG context:          variable
├── User message:         variable
└── Reserved for output:  4,096 tokens (max_tokens)
    ────────────────────
    Available for dynamic content: 119,404 tokens
```

### Budget Allocation Strategy

```python
# Token budget calculator
CONTEXT_WINDOW = 128_000
SYSTEM_PROMPT_TOKENS = 2_000
TOOL_DEFINITIONS_TOKENS = 1_500
FEW_SHOT_TOKENS = 1_000
OUTPUT_RESERVED = 4_096
SAFETY_MARGIN = 500  # buffer for tokenizer discrepancies

FIXED_TOKENS = (SYSTEM_PROMPT_TOKENS + TOOL_DEFINITIONS_TOKENS
                + FEW_SHOT_TOKENS + OUTPUT_RESERVED + SAFETY_MARGIN)

AVAILABLE_FOR_DYNAMIC = CONTEXT_WINDOW - FIXED_TOKENS  # 119,404

# Allocate dynamic budget
def allocate_budget(user_message_tokens: int) -> dict:
    remaining = AVAILABLE_FOR_DYNAMIC - user_message_tokens
    return {
        "conversation_history": int(remaining * 0.4),
        "rag_context": int(remaining * 0.6),
    }
```

### Dynamic Context Window Selection

Some providers offer models with multiple context window sizes at different price points. Route to the smallest sufficient window.

```
Estimated input tokens:
  < 4,000   → Use 8K context model (cheapest)
  < 16,000  → Use 32K context model
  < 64,000  → Use 128K context model
  < 200,000 → Use 200K context model (most expensive)
```

### Conversation History Management

For long conversations, you must trim or summarize history to fit the budget.

**Strategies (in order of sophistication):**
1. **Truncate old messages.** Drop messages from the beginning. Simple but loses context.
2. **Sliding window.** Keep the last N turns. Simple and usually sufficient.
3. **Summarize and truncate.** Periodically summarize older messages, keep the summary + recent turns.
4. **Importance-weighted.** Score each message by relevance to the current query, keep the most relevant.

---

## Capacity Planning

### Estimating TPM Requirements

```
Daily active users:           10,000
Avg requests per user per day: 5
Avg input tokens per request:  2,000
Avg output tokens per request: 500
Peak multiplier:               3x average

Daily tokens:
  Input:  10,000 × 5 × 2,000 = 100M tokens/day
  Output: 10,000 × 5 × 500   = 25M tokens/day

Peak TPM (assuming 8-hour active window):
  Input:  100M / (8 × 60) × 3 = 625,000 TPM
  Output: 25M / (8 × 60) × 3  = 156,250 TPM

Required provider tier: OpenAI Tier 4+ or Anthropic Build Tier 4
```

### Provider Tier Selection

Plan for peak, not average. If you hit rate limits at peak, users experience errors during your busiest period. Size for 2-3x your expected peak to handle traffic spikes.

### Handling Traffic Spikes

**Reactive strategies:**
- Auto-scaling self-hosted inference (add GPU nodes).
- Overflow to a secondary provider.
- Aggressive caching during spikes.
- Request queuing with priority.
- Graceful degradation (smaller models, shorter responses).

**Proactive strategies:**
- Pre-warm capacity before known events (product launches, marketing campaigns).
- Load test regularly to identify bottlenecks.
- Maintain headroom: operate at < 70% of your rate limits normally.
- Use batch APIs for deferrable work, reserving real-time capacity for interactive requests.

### Auto-Scaling for Self-Hosted

```
Metric:        GPU utilization / request queue depth
Scale-up:      When utilization > 80% for 2 minutes, add 1 node
Scale-down:    When utilization < 30% for 10 minutes, remove 1 node
Min instances:  2 (for redundancy)
Max instances:  determined by budget
Cool-down:      5 minutes between scaling actions
```

GPU nodes take 2-5 minutes to start and load model weights. Plan for this startup latency.

---

## Disaster Recovery

### Provider Outages

Every provider has outages. Plan for them.

```
Outage Detection:
  - Health check endpoint: ping provider every 30 seconds
  - Error rate monitoring: if error rate > 20% in 1 minute, declare outage
  - Circuit breaker: after 5 consecutive failures, open circuit

Failover:
  Primary:   Anthropic (Claude Sonnet)
  Secondary: OpenAI (GPT-4o)
  Tertiary:  Google (Gemini Pro)
  Emergency: Self-hosted (Llama 70B) or cached responses

Recovery:
  - Half-open circuit breaker tests primary every 30 seconds
  - When primary recovers, gradually shift traffic back (10% → 25% → 50% → 100%)
  - Don't snap back to 100% immediately (the provider may be fragile post-outage)
```

### Model Deprecation

Providers deprecate models with varying notice periods (weeks to months). Your system must handle this.

**Strategies:**
- **Model alias layer.** Never hardcode model names. Use aliases (`"default-fast"`, `"default-smart"`) that map to specific model versions. Change the mapping without code deploys.
- **Eval-gated migration.** Before switching to a new model, run your eval suite. Only switch if quality metrics hold.
- **Shadow mode.** Run the new model in parallel, compare outputs, switch when confident.
- **Deprecation alerts.** Monitor provider announcements. Track model deprecation dates in your configuration.

### Multi-Region Deployment

For global applications:
- Deploy your gateway in multiple regions.
- Use the geographically closest provider endpoint.
- Failover across regions if one region's provider endpoint is down.
- Be aware of data residency requirements (EU data may need EU endpoints).

### Runbook Essentials

Every production LLM system should have a runbook covering:

1. **Provider outage:** Detection, failover steps, communication template.
2. **Cost spike:** Investigation steps, emergency kill switches, rate limiting.
3. **Quality degradation:** Detection (eval scores dropping), rollback to previous model/prompt, escalation path.
4. **Rate limit exhaustion:** Prioritization, degradation plan, provider upgrade process.
5. **Data incident:** PII in logs, prompt injection exfiltrating data, containment steps.
6. **Model behavior change:** Provider silently updates model, outputs change. Detection and response.

---

## Interview Angle

### Deep-Dive Questions

1. "You're self-hosting a 70B model. Walk me through your infrastructure choices."
   - GPU selection, quantization, inference engine (vLLM), batching strategy, scaling approach. Include actual numbers.

2. "Design a multi-model architecture for a customer support system."
   - Router or cascade. Which models at each tier. How you measure confidence. Cost analysis.

3. "How would you handle a scenario where your primary LLM provider goes down?"
   - Circuit breaker, fallback chain, model alias layer, gradual recovery. Mention real outage experiences if you have them.

4. "Your LLM costs have tripled this month. Walk me through your investigation and optimization."
   - Cost monitoring dashboard, per-feature breakdown, prompt audit, model routing, caching analysis, batch processing.

5. "You need to serve 10,000 concurrent users with sub-2-second latency. Design the system."
   - Capacity planning math, caching strategy, streaming, model selection, auto-scaling, geographic distribution.
