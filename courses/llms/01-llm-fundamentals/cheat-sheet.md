# LLM Fundamentals Cheat Sheet

Quick reference for interview prep. Scan this before your interview.

---

## Model Comparison Table

| Model | Provider | Context | Max Output | Input $/1M | Output $/1M | Strengths | Release |
|---|---|---|---|---|---|---|---|
| GPT-4o | OpenAI | 128K | 16K | ~$2.50 | ~$10 | All-around, multimodal, fast | 2024 |
| GPT-4o-mini | OpenAI | 128K | 16K | ~$0.15 | ~$0.60 | Cheapest quality model | 2024 |
| o3 | OpenAI | 200K | 100K | ~$10 | ~$40 | Best reasoning | 2025 |
| o3-mini | OpenAI | 200K | 100K | ~$1.10 | ~$4.40 | Cheap reasoning | 2025 |
| Claude Opus 4 | Anthropic | 200K | 32K | ~$15 | ~$75 | Top reasoning, coding | 2025 |
| Claude Sonnet 4 | Anthropic | 200K | 16K | ~$3 | ~$15 | Best value, strong coding | 2025 |
| Claude Haiku 3.5 | Anthropic | 200K | 8K | ~$0.80 | ~$4 | Fast, cheap | 2024 |
| Gemini 2.0 Flash | Google | 1M | 8K | ~$0.10 | ~$0.40 | Huge context, cheap | 2025 |
| Gemini 2.5 Pro | Google | 1M | 64K | ~$1.25 | ~$10 | Strong reasoning | 2025 |
| Llama 3.3 70B | Meta | 128K | -- | Self-host | Self-host | Open, fine-tunable | 2024 |
| DeepSeek-V3 | DeepSeek | 128K | 8K | ~$0.27 | ~$1.10 | Cheap, MoE, strong code | 2024 |
| DeepSeek-R1 | DeepSeek | 128K | 8K | ~$0.55 | ~$2.19 | Open reasoning model | 2025 |
| Mistral Large 2 | Mistral | 128K | -- | ~$2 | ~$6 | Multilingual, European | 2024 |

*Prices are approximate and change frequently. Check provider pricing pages for current rates.*

---

## Token / Cost Quick Reference

### Estimation Rules

```
English text:     1 token  ~=  4 characters  ~=  0.75 words
Code:             1 token  ~=  3 characters  ~=  0.4 words
JSON:             Highly variable (verbose keys waste tokens)
Non-English:      1.5x - 3x more tokens than English equivalent
```

### Quick Cost Formulas

```
Total cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)

Where:
  input_price_per_token  = model_input_rate / 1,000,000
  output_price_per_token = model_output_rate / 1,000,000

Example (GPT-4o):
  Prompt: 2000 tokens, Response: 500 tokens
  Cost = (2000 * $2.50/1M) + (500 * $10/1M)
       = $0.005 + $0.005
       = $0.01 per request

  At 10,000 requests/day = $100/day = ~$3,000/month
```

### Common Document Token Estimates

| Content Type | Approximate Tokens |
|---|---|
| 1 page of text (~500 words) | ~670 tokens |
| 1 email | ~200-500 tokens |
| 1 typical API response (JSON) | ~300-1000 tokens |
| 1 PDF page | ~500-800 tokens |
| 1 code file (~200 lines) | ~600-1200 tokens |
| 1 high-res image (GPT-4o) | ~1000 tokens |
| 1 high-res image (Claude) | ~1600 tokens |

---

## Parameter Tuning Guide

### Temperature

| Value | Use Case | Why |
|---|---|---|
| 0 | Classification, extraction, structured output, deterministic code | Reproducible, highest-probability output |
| 0.1-0.3 | Factual Q&A, code generation, data analysis | Slight variation, still focused |
| 0.5-0.7 | General conversation, summaries, documentation | Balanced |
| 0.8-1.2 | Creative writing, brainstorming, diverse suggestions | More variety and surprise |
| >1.2 | Rarely useful | Output becomes incoherent |

### Top-p (Nucleus Sampling)

| Value | Use Case |
|---|---|
| 0.1-0.3 | Very constrained (classification, yes/no) |
| 0.5-0.7 | Focused but some variety |
| 0.9 | General purpose (most common setting) |
| 1.0 | No filtering (use temperature alone for control) |

### Recommended Presets

| Task | Temperature | Top-p | Max Tokens | Notes |
|---|---|---|---|---|
| JSON extraction | 0 | 1.0 | 500-2000 | Use structured output if available |
| Classification | 0 | 1.0 | 10-50 | Short max_tokens prevents rambling |
| Code generation | 0.1-0.2 | 0.95 | 2000-8000 | Low temp for correctness |
| Summarization | 0.3 | 0.9 | 500-2000 | Moderate creativity |
| Conversation | 0.7 | 0.9 | 1000-4000 | Natural-sounding |
| Creative writing | 1.0 | 0.95 | 2000-8000 | High variety |
| Brainstorming | 1.0-1.2 | 0.95 | 2000-4000 | Maximum diversity |

---

## Terminology Glossary

### Architecture

| Term | Definition |
|---|---|
| **Transformer** | Neural network architecture using self-attention; foundation of all modern LLMs. |
| **Self-attention** | Mechanism where each token computes relevance scores against every other token using Q/K/V matrices. |
| **Multi-head attention** | Running multiple attention computations in parallel, each capturing different relationship patterns. |
| **Feed-forward network (FFN)** | Dense layers in each transformer block; stores much of the model's factual knowledge. |
| **Residual connection** | Skip connection (`x + sublayer(x)`) that enables training of very deep networks. |
| **Layer normalization** | Normalizes activations across features; stabilizes training. Modern LLMs use Pre-LN. |
| **Decoder-only** | Architecture that generates text left-to-right with causal masking. GPT, Claude, Llama. |
| **Causal mask** | Attention mask preventing tokens from attending to future positions. |
| **Parameters** | Learned weights. "70B parameters" = 70 billion floating-point numbers. |
| **MoE (Mixture of Experts)** | Architecture activating only a subset of parameters per token; larger capacity at similar compute. |

### Tokenization

| Term | Definition |
|---|---|
| **Token** | Subword unit the model processes; learned from training data via BPE or similar. |
| **BPE** | Byte Pair Encoding; builds vocabulary by iteratively merging frequent character pairs. |
| **SentencePiece** | Tokenizer treating input as raw bytes; language-agnostic. Used by Llama, Gemini. |
| **WordPiece** | Google's tokenization algorithm used in BERT; similar to BPE with likelihood-based merges. |
| **tiktoken** | OpenAI's fast BPE tokenizer implementation (Rust). Used for GPT models. |
| **Special tokens** | Reserved tokens for structure: `<\|endoftext\|>`, `[INST]`, message boundaries. |

### Training

| Term | Definition |
|---|---|
| **Pre-training** | Training on massive text to predict next tokens. Learns language, facts, reasoning. |
| **SFT** | Supervised Fine-Tuning on (instruction, response) pairs. Teaches instruction-following. |
| **RLHF** | Reinforcement Learning from Human Feedback. Humans rank outputs to train a reward model. |
| **DPO** | Direct Preference Optimization. Simpler alternative to RLHF; learns directly from preference pairs. |
| **Constitutional AI** | Anthropic's approach: model self-critiques against explicit principles. |
| **LoRA** | Low-Rank Adaptation. Efficient fine-tuning via small trainable adapter matrices. |
| **Distillation** | Training a smaller model to mimic a larger model's outputs. |
| **Alignment** | Making models helpful, harmless, honest via RLHF/DPO/Constitutional AI. |

### Generation

| Term | Definition |
|---|---|
| **Temperature** | Scales logits before softmax. 0 = deterministic, >1 = more random. |
| **Top-p** | Nucleus sampling: sample from smallest token set with cumulative probability >= p. |
| **Top-k** | Only consider the k most probable next tokens. |
| **Logits** | Raw model output scores before softmax normalization. |
| **Greedy decoding** | Always pick highest-probability token. Equivalent to temperature=0. |
| **Stop sequence** | String that signals the model to stop generating. |
| **Streaming** | Returning tokens via SSE as they are generated. Reduces perceived latency. |
| **TTFT** | Time to First Token. Key latency metric for streaming applications. |

### Embeddings and Retrieval

| Term | Definition |
|---|---|
| **Embedding** | Dense vector representation where semantic similarity maps to geometric proximity. |
| **Cosine similarity** | Angle-based similarity metric for vectors. Standard for text embeddings. |
| **Vector database** | Database optimized for storing and querying high-dimensional vectors (Pinecone, Qdrant, pgvector). |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant docs, inject into prompt. |
| **Chunking** | Splitting documents into smaller pieces for embedding and retrieval. |
| **Hybrid search** | Combining vector (semantic) and keyword (BM25) search. |

### Inference and Production

| Term | Definition |
|---|---|
| **KV cache** | Caching key/value tensors during generation to avoid recomputation. Essential optimization. |
| **Quantization** | Reducing weight precision (FP16 to INT4) for faster, cheaper inference. |
| **GQA / MQA** | Grouped/Multi-Query Attention: sharing KV heads to reduce cache size. |
| **Prompt caching** | Provider-side caching of repeated prompt prefixes for cost/latency savings. |
| **Hallucination** | Model generating plausible but incorrect information. Fundamental limitation. |
| **Prompt injection** | Adversarial input that overrides model instructions. |
| **Structured output** | Constraining model output to match a schema (JSON mode, tool calls). |
| **Context window** | Maximum tokens (input + output) the model processes per request. |

---

## Common Interview Questions

### Conceptual

**Q: How does a transformer process text differently from an RNN?**
A: Transformers process all tokens in parallel via self-attention, while RNNs process sequentially. This enables massive parallelism on GPUs and eliminates the sequential bottleneck that limited RNNs to short effective contexts.

**Q: Why do LLMs hallucinate?**
A: Pre-training optimizes for plausible next-token prediction, not factual accuracy. The model generates text that sounds right based on training patterns, even when it doesn't correspond to reality. There's no internal "fact-checking" mechanism.

**Q: What's the difference between temperature and top-p?**
A: Temperature scales the entire logit distribution (changing the sharpness), while top-p truncates the distribution (removing low-probability tokens). Temperature is global; top-p is adaptive -- it narrows more when the model is confident.

**Q: Why does context window size matter for application design?**
A: It determines how much information (history, retrieved documents, instructions) can fit in a single request. It directly impacts RAG architecture, conversation memory strategies, and cost.

**Q: Explain the training pipeline from base model to production assistant.**
A: Pre-training on internet text (next-token prediction) produces a text completer. SFT on instruction/response pairs teaches helpfulness. RLHF/DPO alignment training teaches safety and preference following. Optional domain fine-tuning or tool-use training for specialization.

### Applied / System Design

**Q: How would you choose between GPT-4o and Claude Sonnet for a production system?**
A: Evaluate on your specific task via blind evals. Consider: cost (similar tier), context window (Claude has 200K vs 128K), structured output support, streaming behavior, rate limits, and your existing infrastructure. Run A/B tests on real workloads.

**Q: How would you handle a document that exceeds the context window?**
A: Options: (1) Chunk and retrieve relevant sections via RAG, (2) Hierarchical summarization (map-reduce), (3) Sliding window processing, (4) Use a model with a larger context window (Gemini 1M+). Choice depends on the task -- retrieval for Q&A, summarization for overviews.

**Q: When would you use a reasoning model (o3) vs a standard model (GPT-4o)?**
A: Reasoning models for complex math, multi-step logic, hard coding problems where accuracy justifies 5-20x higher cost and latency. Standard models for everything else -- conversation, extraction, classification, simple generation. Route dynamically based on task complexity.

**Q: How do you estimate and control LLM costs in production?**
A: Track input/output tokens per request. Use model routing (cheap models for easy tasks). Apply prompt caching for repeated prefixes. Minimize output tokens via specific instructions and stop sequences. Set max_tokens appropriately. Monitor and alert on cost anomalies.

**Q: What's the "lost in the middle" problem and how do you mitigate it?**
A: Models attend more to the beginning and end of context, with degraded attention in the middle. Mitigation: put critical information at the start/end, use RAG to place only relevant chunks near the query, keep context concise rather than dumping everything in.

---

## Architecture Quick Reference

```
Transformer Block:
  Input -> LayerNorm -> Multi-Head Attention -> + Residual -> LayerNorm -> FFN -> + Residual -> Output

Attention:
  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

Multi-Head:
  MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O
  where head_i = Attention(QW_Qi, KW_Ki, VW_Vi)

Complexity:
  Attention: O(n^2 * d) time, O(n^2) memory
  FFN: O(n * d^2) time, O(d^2) memory
  Total per layer: O(n^2 * d + n * d^2)
```

---

## Cost Optimization Checklist

1. [ ] Use the cheapest model that meets quality requirements
2. [ ] Implement model routing (easy tasks -> cheap models)
3. [ ] Enable prompt caching (stable system prompts)
4. [ ] Set appropriate max_tokens (don't over-allocate)
5. [ ] Use stop sequences to prevent over-generation
6. [ ] Trim conversation history to what's needed
7. [ ] Use concise JSON keys in context
8. [ ] Batch non-real-time requests where possible
9. [ ] Monitor token usage per request type
10. [ ] Evaluate distilled/fine-tuned models for high-volume tasks
