# Interview Cheat Sheet — Quick-Fire Reference

Use this the night before. One-liner answers, key numbers, comparison tables, and anti-patterns.

---

## "Explain Like I'm Interviewing" — 30-Second Answers

### Fundamentals

| Question | Answer |
|---|---|
| How do LLMs work? | Neural networks trained to predict the next token on massive text corpora; they learn statistical patterns in language and generate text by repeatedly sampling from a probability distribution over the vocabulary. |
| What is a transformer? | Architecture that processes all tokens in parallel using self-attention (every token attends to every other token), replacing the sequential processing of RNNs. |
| What is attention? | Mechanism where each token computes relevance scores (via Q/K dot products) against all other tokens, then uses those weights to aggregate information (Values). Multi-head runs this in parallel with different learned projections. |
| What is tokenization? | Converting text into subword units (tokens) the model processes. Uses BPE or similar algorithms. Matters because pricing, context limits, and many edge cases are per-token. |
| What is the context window? | Maximum tokens (input + output) the model processes in one call. It is the model's working memory -- everything it needs must be in the prompt or its weights. |
| What is the KV cache? | Cached Key/Value matrices from previous tokens during generation. Avoids recomputing attention over the full sequence for each new token. Explains why first-token latency is higher than subsequent tokens. |
| What are embeddings? | Dense vector representations where semantic similarity maps to geometric proximity. Foundation for semantic search and RAG. |
| What is hallucination? | Model generates plausible but factually incorrect text because it optimizes for probable, not true. Mitigate with RAG, constrained output, citations, and human review. |
| What is temperature? | Scales logits before softmax. 0 = deterministic, higher = more random. Controls creativity vs. consistency. |
| What are reasoning models? | Models like o1/o3 that trade inference-time compute for accuracy by generating extended chain-of-thought reasoning. Best for hard math/logic/code. Overkill for simple tasks. |

### Prompt Engineering

| Question | Answer |
|---|---|
| What is chain-of-thought? | Asking the model to reason step by step before answering. Dramatically helps multi-step reasoning. Uses more tokens. |
| What is few-shot prompting? | Including example (input, output) pairs in the prompt to calibrate the model's behavior. Usually 3-5 diverse examples suffice. |
| How do you get structured output? | Provider-enforced schemas (OpenAI JSON mode, Anthropic tool use) are most reliable. Always validate with Pydantic. Retry with error message as fallback. |
| What is prompt injection? | User-provided content that manipulates model behavior. Defend with delimiters, instruction hierarchy, input validation, output validation, and least privilege. |
| What is prompt chaining? | Breaking a complex task into a pipeline of simpler LLM calls, each with a focused prompt. More reliable and debuggable than one giant prompt. |

### RAG

| Question | Answer |
|---|---|
| What is RAG? | Retrieve relevant documents at query time and include them in the prompt to ground the model's response in specific, current information. |
| What is chunking? | Splitting documents into appropriately-sized pieces for embedding and retrieval. Common strategies: fixed-size, recursive/structural, semantic. |
| What is hybrid search? | Combining vector (semantic) search with keyword (BM25) search, merged via Reciprocal Rank Fusion. Better recall than either alone. |
| What is reranking? | Using a cross-encoder to re-score retrieved chunks for more precise relevance ranking after initial retrieval. |
| What is the "lost in the middle" problem? | Models attend more to the beginning and end of context; information in the middle gets less attention. Place important content at the start or end. |

### Agents

| Question | Answer |
|---|---|
| What is function calling? | Model outputs structured tool call requests; your code executes them and returns results. Model never executes anything itself. |
| What is an agent loop? | Send messages to LLM, if it returns tool calls execute them and loop back, repeat until the model returns text or you hit a max iteration limit. |
| What is ReAct? | Thought-Action-Observation loop: model reasons about what to do, calls a tool, processes the result, and repeats until done. |
| What is MCP? | Model Context Protocol -- an open standard for connecting AI applications to tools and data sources. Universal connector instead of one-off integrations. |

### Fine-tuning

| Question | Answer |
|---|---|
| When to fine-tune? | To change model behavior (style, format, domain reasoning). Not for knowledge injection -- use RAG for that. |
| What is LoRA? | Low-Rank Adaptation: trains tiny matrices (0.1-1% of params) added to frozen base weights. Same quality as full fine-tuning at a fraction of the cost. |
| What is DPO? | Direct Preference Optimization: simpler alternative to RLHF that directly optimizes on preference data without a separate reward model. |
| What is SFT? | Supervised Fine-Tuning: training on (instruction, response) pairs to teach instruction-following behavior. |

### Production

| Question | Answer |
|---|---|
| How do you handle streaming? | Server-Sent Events push tokens as generated. Buffer for JSON parsing. Handle partial tool calls, stream failures, and backpressure. |
| How do you optimize cost? | Model tiering (cheapest model per task), caching (response + provider prompt caching), token optimization (concise prompts, summarized history). |
| How do you evaluate? | Offline: curated test set run on every change. Online: LLM-as-judge on production samples, user feedback signals, A/B tests. |
| How do you handle safety? | Defense in depth: input filtering, output validation, least privilege for tools, human-in-the-loop for destructive actions, kill switches, audit logging. |

---

## System Design Interview Template

Follow this structure for any LLM system design question:

```
STEP 1: CLARIFY REQUIREMENTS (2 min)
  Functional: Core task? Knowledge needs? Actions? Interface?
  Non-functional: Latency? Scale? Accuracy? Cost?
  Safety: What can go wrong? Users? Compliance?

STEP 2: HIGH-LEVEL ARCHITECTURE (3 min)
  Draw the data flow. Identify the pattern:
    - Simple Prompt (classification, extraction)
    - RAG (knowledge-grounded Q&A)
    - Agent (multi-step, tool use)
    - Pipeline (staged processing)
  Most real systems combine multiple patterns.

STEP 3: DEEP DIVE (5-10 min)
  Pick 2-3 most interesting/complex components.
  For each: what it does, why this approach, tradeoffs.
  Show concrete details: prompt snippets, schemas, data models.

STEP 4: PRODUCTION CONCERNS (2-3 min)
  Reliability: retries, fallbacks, circuit breakers
  Performance: streaming, caching, model routing
  Cost: tiering, token optimization, monitoring
  Observability: logging, tracing, alerting

STEP 5: EVALUATION (2 min)
  Offline: test set, automated scoring, regression checks
  Online: sampled quality monitoring, user feedback, A/B tests

STEP 6: COST ESTIMATION (1 min)
  Tokens x price x volume = $/day
  Is this financially viable?
```

---

## Key Numbers to Know

### Model Context Windows (as of early 2026)

| Model | Context Window | Max Output |
|---|---|---|
| GPT-4o | 128K tokens | 16K tokens |
| GPT-4o-mini | 128K tokens | 16K tokens |
| Claude Sonnet 3.5/4 | 200K tokens | 8K tokens |
| Claude Haiku 3.5 | 200K tokens | 8K tokens |
| Claude Opus 4 | 200K tokens | 32K tokens |
| Gemini 1.5 Pro | 1-2M tokens | 8K tokens |
| Gemini Flash | 1M tokens | 8K tokens |
| Llama 3.1 (405B) | 128K tokens | varies |
| DeepSeek-R1 | 128K tokens | 8K tokens |

### Approximate Pricing (per 1M tokens, early 2026)

| Model | Input | Output |
|---|---|---|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |
| Claude Opus 4 | $15.00 | $75.00 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini Flash | $0.075 | $0.30 |
| text-embedding-3-small | $0.02 | N/A |
| text-embedding-3-large | $0.13 | N/A |

### Useful Conversions

```
1 token ≈ 0.75 English words (or ~4 characters)
1,000 tokens ≈ 750 words ≈ 1.5 pages of text
1M tokens ≈ 750K words ≈ 3,000 pages ≈ 5-6 novels

A typical chat message: 50-200 tokens
A typical RAG prompt: 2,000-5,000 tokens
A 10-page document: ~4,000 tokens
An entire codebase (medium project): 200K-500K tokens
```

### Latency Benchmarks

```
Time to first token (TTFT):
  Cloud API (fast model):     100-300ms
  Cloud API (capable model):  200-800ms
  Reasoning model (o1-class): 2-30 seconds
  Self-hosted (Llama on GPU): 50-200ms

Token generation speed:
  Cloud API:        30-100 tokens/second
  Self-hosted:      10-50 tokens/second (depends on hardware)

Embedding:
  Single text:      10-50ms via API
  Batch of 100:     100-500ms via API

Vector search (pgvector, 100K vectors):  5-20ms
Vector search (Pinecone, 1M vectors):    10-50ms
Cross-encoder reranking (20 passages):   50-200ms
```

---

## Comparison Tables

### RAG vs Fine-tuning vs Long Context

| Dimension | RAG | Fine-tuning | Long Context |
|---|---|---|---|
| **Best for** | Updatable knowledge | Behavior/style changes | Small, static datasets |
| **Data freshness** | Real-time updates | Stale after training | Current at query time |
| **Setup effort** | Medium (pipeline) | High (data prep + training) | Low (just stuff it in) |
| **Per-query cost** | Medium (retrieval + generation) | Low (inference only) | High (long prompts) |
| **Upfront cost** | Medium (embedding + indexing) | High (training run) | None |
| **Accuracy for facts** | High (grounded in sources) | Unreliable | High (if in context) |
| **Scalability** | Scales to millions of docs | Fixed at training time | Limited by context window |
| **Traceability** | High (can cite sources) | Low (knowledge baked in) | High (in context) |

### Agent Frameworks

| Framework | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **LangGraph** | Graph-based flows, checkpoints, human-in-the-loop | Complex API, steep learning curve | Complex stateful workflows |
| **CrewAI** | Simple multi-agent setup, role-based | Less flexible for custom patterns | Multi-agent role-play |
| **AutoGen** | Research-oriented, conversational agents | Heavy, Microsoft-centric | Experimental multi-agent |
| **Custom loop** | Full control, no dependencies | More code to write | Production systems |
| **Anthropic/OpenAI SDK** | Simple, well-documented | Single-provider | Most production use cases |

### Vector Databases

| Database | Managed? | Hybrid Search? | Best For |
|---|---|---|---|
| **Pinecone** | Yes | Yes | Zero-ops, scale |
| **Weaviate** | Both | Built-in | Hybrid search |
| **Qdrant** | Both | Yes | Performance |
| **pgvector** | Self (Postgres ext.) | Via Postgres FTS | Existing Postgres stack |
| **ChromaDB** | Self | No | Prototyping |
| **FAISS** | Self (library) | No | Research, custom |

---

## Red Flags and Anti-Patterns

Things that signal you do NOT know what you are doing:

### Architecture Anti-Patterns

- **Fine-tuning for knowledge injection.** Use RAG. Fine-tuning is for behavior, not facts.
- **One model for everything.** Use the cheapest model that meets your quality bar per task. Haiku for classification, Sonnet for generation.
- **No eval suite.** If you cannot measure quality, you cannot improve it. "It looks good" is not a metric.
- **Sending raw user input directly into prompts without delimiters.** This is an injection vulnerability.
- **Ignoring cost until production.** Do the math early. A $10/query system serving 100K queries/day is $1M/day.
- **Using LangChain in production because you used it for prototyping.** Evaluate whether you need the abstraction. Often you do not.

### Prompt Engineering Anti-Patterns

- **Prompt engineering by vibes.** Change one thing, measure the impact, keep what works. No evals = guessing.
- **Enormous monolithic prompts.** Break complex tasks into chains of focused prompts.
- **Ignoring the "lost in the middle" effect.** Put important instructions at the start and end, not buried in context.
- **Setting temperature > 0 for deterministic tasks.** Classification, extraction, and structured output should use temperature 0.
- **Not versioning prompts.** Prompts are code. Version them, test them, review them.

### RAG Anti-Patterns

- **Chunks that are too small.** A 100-token chunk has no context. Start at 500-1000 tokens.
- **Chunks that are too large.** A 3000-token chunk dilutes relevance. The embedding represents the whole chunk.
- **Not using metadata filtering.** If you have document types or dates, filter before vector search.
- **Skipping reranking.** A cross-encoder reranker is a cheap, high-impact improvement for retrieval precision.
- **Embedding query != embedding document.** Some embedding models have separate query/document modes. Use them correctly.

### Agent Anti-Patterns

- **No iteration limit.** An unbounded agent loop will run forever (and drain your wallet).
- **Crashing on tool errors.** Return structured errors to the model. It can often recover.
- **Letting the model execute arbitrary code or SQL.** Sandbox everything. Validate tool inputs.
- **Tools with poor descriptions.** The model uses the description to decide when to call a tool. Bad description = wrong tool usage.
- **Multi-agent when single-agent suffices.** Start with one agent. Split only when complexity demands it.

### Production Anti-Patterns

- **No observability.** If you are not logging prompts, responses, latency, and cost, you cannot debug or optimize.
- **No fallback for API outages.** LLM APIs go down. Have retries, circuit breakers, and fallback models.
- **Streaming as an afterthought.** For user-facing chat, streaming is table stakes. Design for it from the start.
- **Ignoring prompt caching.** Providers offer discounted rates for cached prompt prefixes. Free money for RAG and agent workloads.

---

## Buzzword Decoder

Accurate definitions for overloaded terms:

| Term | What it actually means |
|---|---|
| **AGI** | Artificial General Intelligence -- AI with human-level capability across all domains. Does not exist yet. Not a product feature. |
| **Agentic** | A system where the LLM autonomously decides what actions to take in a loop, rather than just generating text in a single pass. |
| **Alignment** | Training models to be helpful, harmless, and honest. Concretely: RLHF, DPO, constitutional AI. |
| **Chain-of-thought** | Making the model show its reasoning step by step. Improves accuracy on reasoning tasks. |
| **Context window** | Maximum tokens the model processes per call. Not the same as "memory" -- there is no persistence between calls. |
| **Distillation** | Training a smaller model to mimic a larger model's outputs. A way to get big-model quality at small-model cost. |
| **Embedding** | Dense vector representation of text for similarity search. Not the same as an LLM response. |
| **Fine-tuning** | Further training a model on domain-specific data. Changes behavior, not just knowledge. |
| **Grounding** | Connecting model outputs to verifiable sources (via RAG, tool use, or citations). Reduces hallucination. |
| **Guardrails** | Safety mechanisms that constrain model behavior: input/output filtering, tool permissions, human oversight. |
| **Hallucination** | Model generates confident-sounding false information. Not a bug; an inherent property of next-token prediction. |
| **Inference** | Running a trained model to generate predictions/text. As opposed to training. |
| **LoRA** | Low-Rank Adaptation. Fine-tuning a tiny fraction of model parameters. Not a model; an adapter applied to a base model. |
| **MCP** | Model Context Protocol. Standard for connecting AI apps to tools/data. Think USB-C for AI integrations. |
| **Multimodal** | Model that handles multiple input types: text + images + audio + video. |
| **Prompt caching** | Provider-side caching of repeated prompt prefixes for reduced cost. Different from response caching. |
| **Quantization** | Reducing model weight precision (32-bit to 8-bit or 4-bit) to reduce memory and increase speed. Slight quality tradeoff. |
| **RAG** | Retrieval-Augmented Generation. Retrieve documents, include in prompt, generate grounded response. |
| **Reasoning model** | Model trained to think step-by-step at inference time (o1, o3, R1). Trades latency for accuracy. |
| **RLHF** | Reinforcement Learning from Human Feedback. Training models on human preferences. |
| **Semantic search** | Finding content by meaning rather than keywords. Powered by embeddings and vector similarity. |
| **Structured output** | Forcing the model to produce valid JSON/schema-conformant output. Provider-enforced is best. |
| **Token** | The atomic unit the model processes. Subword piece, not a word. Everything (cost, limits, latency) is per-token. |
| **Tool use** | Same as function calling. Model requests actions; your code executes them. |
| **Vector database** | Database optimized for storing and querying dense vectors (embeddings). Not a regular database. |
