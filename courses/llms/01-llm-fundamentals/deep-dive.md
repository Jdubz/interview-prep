# Deep Dive: Advanced LLM Fundamentals

Extended topics that come up in senior-level interviews and system design discussions.

---

## Table of Contents

1. [Scaling Laws](#scaling-laws)
2. [KV Cache](#kv-cache)
3. [Positional Encoding](#positional-encoding)
4. [Emergent Abilities](#emergent-abilities)
5. [Inference Optimization](#inference-optimization)
6. [Multi-Modal Models](#multi-modal-models)
7. [Reasoning Models](#reasoning-models)
8. [Model Distillation](#model-distillation)

---

## Scaling Laws

Scaling laws describe the relationship between model performance and three key variables:
model size (parameters), dataset size (tokens), and compute budget (FLOPs).

### Kaplan et al. (OpenAI, 2020) -- Original Scaling Laws

Found power-law relationships:

```
Loss ~= C * N^(-0.076)     (N = parameters)
Loss ~= C * D^(-0.095)     (D = dataset tokens)
Loss ~= C * F^(-0.050)     (F = compute FLOPs)
```

Key claim: Performance improves predictably with scale, and model size matters more than
dataset size for a fixed compute budget.

### Chinchilla (DeepMind, 2022) -- Compute-Optimal Training

Chinchilla challenged the "bigger model is always better" assumption. Key finding:

```
Optimal training:  tokens ~= 20 * parameters

Example:
  70B parameter model should be trained on ~1.4T tokens
  (Chinchilla: 70B params, 1.4T tokens -- matched 280B Gopher with 4x fewer params)
```

**Why this matters in interviews:**

| Model | Parameters | Training Tokens | Chinchilla-Optimal? |
|---|---|---|---|
| GPT-3 | 175B | 300B | Undertrained (~2x tokens per param) |
| Chinchilla | 70B | 1.4T | Yes (20x) |
| Llama 2 70B | 70B | 2T | Over-trained (by design -- cheaper inference) |
| Llama 3 70B | 70B | 15T+ | Very over-trained (inference efficiency) |

**The Llama strategy**: Meta deliberately trains smaller models on far more data than
Chinchilla-optimal. Why? A 70B model trained on 15T tokens outperforms a 175B model trained
on 1.4T tokens at the same inference cost. Training is a one-time cost; inference is ongoing.
When you're serving billions of requests, smaller-but-well-trained models win economically.

### Implications for Applied Engineers

- **You can't just scale your way out of problems.** A model needs both enough parameters
  AND enough training data. More of one doesn't compensate for a deficit in the other.
- **Inference cost dominates.** Train once, serve forever. The industry has shifted toward
  smaller, more efficient models for this reason.
- **Diminishing returns are real.** Going from 7B to 70B is a big jump in quality. Going
  from 70B to 700B is a smaller jump. Know when "good enough" is good enough.

---

## KV Cache

The KV cache is the single most important inference optimization for autoregressive LLMs.
Understanding it is essential for reasoning about latency, memory, and throughput.

### The Problem

During autoregressive generation, the model generates one token at a time. Each new token
requires attending to ALL previous tokens. Without caching, generating token N requires
re-computing the K and V projections for tokens 1 through N-1 -- work that was already
done when generating the previous tokens.

```
Without KV Cache:
  Generate token 1: compute K,V for [t1]
  Generate token 2: compute K,V for [t1, t2]       <-- recomputes t1
  Generate token 3: compute K,V for [t1, t2, t3]   <-- recomputes t1, t2
  Generate token N: compute K,V for [t1, ..., tN]   <-- recomputes everything

  Total K,V computations: O(N^2)

With KV Cache:
  Generate token 1: compute K,V for [t1], cache it
  Generate token 2: compute K,V for [t2], append to cache, attend to [t1, t2]
  Generate token 3: compute K,V for [t3], append to cache, attend to [t1, t2, t3]
  Generate token N: compute K,V for [tN], append to cache, attend to [t1, ..., tN]

  Total K,V computations: O(N)
```

### Memory Cost

The KV cache grows linearly with sequence length, and the memory is substantial:

```
KV cache memory per token per layer:
  = 2 (K and V) * num_heads * head_dim * bytes_per_param

For Llama 2 70B (80 layers, 64 heads, head_dim=128, FP16):
  Per token: 2 * 64 * 128 * 2 bytes * 80 layers = 2.6 MB
  For 4096 tokens: ~10.5 GB
  For 128K tokens: ~330 GB  (more than the model weights!)

This is why long-context inference is so expensive on memory.
```

### Two Phases of Inference

**Prefill (prompt processing)**:
- Process all input tokens in parallel (like training)
- Build the initial KV cache for the entire prompt
- Compute-bound (lots of matrix multiplications)
- Latency scales roughly linearly with prompt length

**Decode (token generation)**:
- Generate one token at a time
- Each step: compute Q for new token, attend to full KV cache, run through FFN
- Memory-bound (reading KV cache dominates)
- Latency is roughly constant per token (but memory access is the bottleneck)

```
Time breakdown for a typical request:

|<-- Prefill (parallel) -->|<-- Decode (sequential, one token at a time) -->|
|      Process prompt      | t1 | t2 | t3 | t4 | ... | tN |
|      ~100ms for 4K       | ~20ms each                    |
```

This is why Time to First Token (TTFT) and tokens-per-second are different metrics.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

These architectural changes reduce KV cache size:

**Standard Multi-Head Attention (MHA)**:
- Each head has its own K, V projections
- KV cache size: `2 * num_heads * head_dim * seq_len * num_layers`

**Multi-Query Attention (MQA)** (Shazeer, 2019):
- All heads share ONE set of K, V projections
- Q still has per-head projections
- KV cache reduced by `num_heads`x (e.g., 32x for 32 heads)
- Slight quality loss

**Grouped-Query Attention (GQA)** (Ainslie et al., 2023):
- Compromise: group heads and share K, V within groups
- E.g., 32 Q heads grouped into 8 KV groups (4 Q heads per KV group)
- KV cache reduced by 4x
- Minimal quality loss

```
MHA:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8    (8 heads)
      K1  K2  K3  K4  K5  K6  K7  K8    (8 KV heads)

GQA:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8    (8 Q heads)
      K1  K1  K2  K2  K3  K3  K4  K4    (4 KV groups)

MQA:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8    (8 Q heads)
      K1  K1  K1  K1  K1  K1  K1  K1    (1 KV head)
```

| Architecture | Used By | KV Cache Size |
|---|---|---|
| MHA | GPT-3, older models | Baseline (100%) |
| GQA | Llama 2/3, Mistral, Gemma | ~12-25% of MHA |
| MQA | Falcon, PaLM, some GPT variants | ~3-6% of MHA |

---

## Positional Encoding

Transformers have no inherent notion of token order. Positional encodings inject position
information so the model knows that "cat sat" and "sat cat" are different.

### Sinusoidal (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Added directly to token embeddings. The key insight: each dimension oscillates at a different
frequency, so the combination uniquely identifies each position. Relative positions can be
computed via linear transformation.

**Limitation**: Theoretically generalizes to unseen lengths, but in practice performance
degrades beyond training lengths.

### Learned Position Embeddings (GPT-2/3)

Simply learn a position embedding table: `PE[pos]` is a learned vector for position `pos`.

**Limitation**: Fixed to the maximum training length. Cannot extrapolate.

### RoPE (Rotary Position Embeddings) -- Used by Most Modern LLMs

Encodes position by rotating the Q and K vectors:

```
For a pair of dimensions (q_2i, q_2i+1) at position m:

  q_2i'    = q_2i * cos(m*theta_i) - q_2i+1 * sin(m*theta_i)
  q_2i+1'  = q_2i * sin(m*theta_i) + q_2i+1 * cos(m*theta_i)

  where theta_i = 1 / 10000^(2i/d)
```

**Why RoPE won**: The dot product between rotated Q and K vectors naturally captures
relative position (the rotation angles subtract). This means:
- Relative position information is embedded in the attention scores
- Can theoretically extrapolate to longer sequences
- No additional parameters needed

**Extending RoPE for longer contexts**:
- **NTK-aware scaling**: Modify the base frequency to better utilize high-frequency
  components for longer sequences
- **YaRN**: Combines NTK interpolation with attention scaling
- **Dynamic NTK**: Adjust scaling dynamically based on sequence length

Used by: Llama 2/3, Mistral, Qwen, DeepSeek, Gemma, and most modern open-source models.

### ALiBi (Attention with Linear Biases)

Instead of modifying embeddings, ALiBi adds a position-dependent bias directly to attention
scores:

```
attention_score(i, j) = q_i * k_j - m * |i - j|

where m is a head-specific slope (fixed, not learned)
```

Each head uses a different slope, so some heads focus on local context and others on
distant context. No modification to embeddings at all.

**Strengths**: Extremely simple, no added parameters, strong length extrapolation.
**Used by**: BLOOM, MPT, some Falcon variants.

### Comparison

| Method | Relative Position | Extrapolation | Used By |
|---|---|---|---|
| Sinusoidal | Via linear transform | Poor | Original Transformer |
| Learned | No (absolute only) | None | GPT-2/3 |
| RoPE | Native (rotation) | Moderate (with scaling) | Llama, Mistral, most modern |
| ALiBi | Via linear bias | Strong | BLOOM, MPT |

---

## Emergent Abilities

Abilities that appear suddenly at certain model scales, rather than improving gradually.

### In-Context Learning (ICL)

Models can learn new tasks from examples in the prompt without any weight updates:

```
System: Classify the sentiment.

User:
"I love this movie!" -> Positive
"Terrible experience." -> Negative
"The food was okay." -> Neutral
"Best concert ever!" -> ?

Model: Positive
```

This is not in the training data. The model is performing a new task specified entirely by
the prompt examples. This ability emerges strongly around the 6-10B parameter scale.

### Chain-of-Thought (CoT) Reasoning

Prompting the model to "think step by step" dramatically improves reasoning performance:

```
Without CoT:
Q: "If a train travels at 60mph for 2.5 hours, how far does it go?"
A: "120 miles"  <-- wrong

With CoT:
Q: "If a train travels at 60mph for 2.5 hours, how far does it go? Think step by step."
A: "Speed is 60 mph. Time is 2.5 hours. Distance = speed x time = 60 x 2.5 = 150 miles."
```

CoT works because it forces the model to allocate compute tokens to intermediate reasoning
steps rather than trying to jump directly to the answer. Each token generated becomes context
for the next, enabling multi-step reasoning.

**Emerges strongly around 100B+ parameters.** Smaller models often produce incoherent chains
of thought that don't improve accuracy.

### Instruction Following

The ability to follow complex, multi-part instructions emerges with scale:
- Small models: Can follow simple one-step instructions
- Medium models: Can follow multi-step instructions
- Large models: Can follow nuanced, conditional instructions with complex formatting

### Why Emergence Matters for Engineers

- **Model selection**: Some tasks require a minimum model size. Don't use a 7B model for
  complex reasoning tasks and blame your prompt when it fails.
- **Prompt design**: CoT prompting is free performance for reasoning tasks, but only works
  well with capable models.
- **Evaluation**: Test your application across model tiers. If your prompts require emergent
  abilities, they won't work with smaller models.

---

## Inference Optimization

Understanding inference optimization helps you reason about latency, cost, and deployment
architecture in system design interviews.

### Quantization

Reducing the precision of model weights to use less memory and compute faster.

```
Full Precision (FP32):    32 bits per parameter    (baseline)
Half Precision (FP16):    16 bits per parameter    (2x compression, ~same quality)
BFloat16 (BF16):          16 bits per parameter    (preferred for training, wider range)
INT8 (8-bit):              8 bits per parameter    (4x compression, minimal quality loss)
INT4 (4-bit):              4 bits per parameter    (8x compression, some quality loss)
```

**Popular quantization methods**:

| Method | Bits | Approach | Used For |
|---|---|---|---|
| GPTQ | 4-bit | Post-training, GPU-optimized | GPU inference |
| GGUF (llama.cpp) | 2-6 bit | Post-training, CPU-friendly | Local/CPU inference |
| AWQ | 4-bit | Activation-aware, preserves important weights | GPU inference |
| bitsandbytes | 4/8-bit | Dynamic, integrated with HuggingFace | Training and inference |
| SmoothQuant | 8-bit | Migrates difficulty from activations to weights | Production serving |
| FP8 | 8-bit | Native FP8 on H100/H200 GPUs | High-throughput serving |

**Practical impact**: A 70B parameter model at FP16 requires ~140GB of VRAM (2 bytes * 70B).
At INT4, it needs ~35GB -- fitting on a single A100 80GB GPU instead of two.

### Speculative Decoding

Use a small, fast "draft" model to generate candidate tokens, then verify them in parallel
with the large model.

```
1. Draft model generates N candidate tokens quickly:  [t1, t2, t3, t4, t5]
2. Large model verifies all N tokens in a single forward pass (parallel)
3. Accept tokens until the first rejection:            [t1, t2, t3] accepted, t4 rejected
4. Large model generates the correct t4
5. Repeat

Speedup: If draft model acceptance rate is ~70%, get ~2-3x throughput improvement
without any quality loss (same output as running the large model alone).
```

**Key insight**: Verification is parallel (prefill-like) while generation is sequential.
Speculative decoding converts sequential generation into parallel verification.

### Continuous Batching

Traditional batching waits for all requests in a batch to finish before starting new ones.
Continuous batching (iteration-level batching) adds new requests as soon as existing ones
finish, maximizing GPU utilization.

```
Traditional batching:
  Request A: |====|
  Request B: |========|
  Request C:           |====|       <-- waits for B to finish
  GPU idle:      XXXX              <-- A finishes but GPU waits for B

Continuous batching:
  Request A: |====|
  Request B: |========|
  Request C:      |====|           <-- starts as soon as there's capacity
  GPU idle:  (none)                <-- always processing something
```

Implemented by vLLM, TGI (HuggingFace), TensorRT-LLM.

### PagedAttention (vLLM)

Inspired by virtual memory paging in operating systems:

- Traditional KV cache allocates contiguous memory per sequence -> fragmentation and waste
- PagedAttention stores KV cache in non-contiguous "pages" (blocks)
- Pages can be shared across sequences (e.g., shared system prompts)
- Dramatically reduces memory waste (near-zero fragmentation)

**Impact**: 2-4x higher throughput by fitting more concurrent requests in the same GPU memory.

### Prefix Caching (Prompt Caching)

Cache the KV cache for common prompt prefixes at the provider level:

```
Request 1: [System Prompt | User A's question]
  --> Computes KV cache for entire prompt

Request 2: [System Prompt | User B's question]
  --> Reuses cached KV for System Prompt, only computes for User B's question

Savings: Skip prefill for the shared prefix (often 50-90% of input tokens)
```

Both OpenAI and Anthropic offer automatic prompt caching. Anthropic's requires a minimum
prefix length (1024 tokens) and provides a 90% input cost discount on cached tokens.

### Optimization Summary

| Technique | What it Optimizes | Typical Speedup | Quality Impact |
|---|---|---|---|
| Quantization (INT4) | Memory, compute | 2-4x throughput | Slight degradation |
| Speculative decoding | Latency, throughput | 2-3x tokens/sec | None (exact) |
| Continuous batching | Throughput | 2-5x requests/sec | None |
| PagedAttention | Memory efficiency | 2-4x concurrent users | None |
| FlashAttention | Compute, memory | 2-4x attention speed | None (exact) |
| Prompt caching | Latency, cost | 50-90% input savings | None |
| KV cache (GQA/MQA) | Memory | 4-32x cache reduction | Minimal |

---

## Multi-Modal Models

Models that process multiple input types (text, images, audio, video) in a unified
architecture.

### How Images Are Tokenized

Most vision-language models (VLMs) follow this pattern:

```
Image (e.g., 1024x1024 pixels)
    |
    v
[Vision Encoder] (e.g., ViT -- Vision Transformer)
    |
    Splits image into patches (e.g., 14x14 or 16x16 pixel patches)
    Each patch is linearly projected into an embedding
    Run through transformer layers
    |
    v
Image token embeddings (e.g., 576 tokens for a 384x384 image with 16x16 patches)
    |
    v
[Projection Layer / Adapter]
    Maps vision embeddings into the LLM's embedding space
    |
    v
[LLM Backbone]
    Image tokens are interleaved with text tokens
    Attend to each other via standard self-attention
    |
    v
Text output (descriptions, answers, analysis)
```

**Key point**: Images become tokens in the same embedding space as text. The LLM doesn't
"see" pixels -- it processes learned representations of image patches alongside text tokens.

### Image Token Costs

Images consume significant context budget:

| Provider | Low Detail | High Detail |
|---|---|---|
| GPT-4o | 85 tokens | 170-1105 tokens (varies by resolution) |
| Claude 3.5+ | ~1600 tokens (typical photo) | Up to ~6000 tokens |
| Gemini | ~258 tokens (low) | Varies by resolution |

This means a single high-res image can cost as much as a full page of text.

### Audio Tokenization

Audio models follow a similar pattern:
1. Audio waveform is converted to spectrograms or mel-frequency features
2. An audio encoder (Whisper-like) produces audio embeddings
3. Audio embeddings are projected into the LLM's embedding space
4. Processed alongside text tokens

**Whisper approach**: Splits audio into 30-second chunks, uses a spectrogram representation,
then runs through an encoder-decoder transformer.

### Architecture Patterns

**Early fusion**: Image/audio tokens mixed with text tokens from the start. Used by GPT-4o,
Gemini. The LLM jointly attends to all modalities.

**Late fusion**: Separate encoders process each modality, results are combined later. Simpler
but less cross-modal understanding.

**Cross-attention**: LLM attends to encoded image/audio via cross-attention layers (rather
than self-attention). Used in some encoder-decoder architectures. Flamingo-style models.

### Multi-Modal Model Comparison

| Model | Modalities | Key Strength |
|---|---|---|
| GPT-4o | Text, image, audio (native) | Natively multimodal, strong across all |
| Claude 3.5/4 | Text, image | Strong image understanding, document analysis |
| Gemini 2.0 | Text, image, audio, video | Broadest modality support |
| Llama 3.2 Vision | Text, image | Open-source multimodal |
| Qwen2-VL | Text, image, video | Strong open-source VLM |

---

## Reasoning Models

A new category of models (starting with o1) that allocate extra inference-time compute
to "think" through problems before responding.

### How Reasoning Models Work

Standard models map input directly to output in a single forward pass (per token).
Reasoning models generate an internal chain of thought ("thinking tokens") before producing
the visible response.

```
Standard Model:
  Input: "What is 23 * 47?"
  Output: "1081"
  Compute: ~N forward passes (one per output token)

Reasoning Model:
  Input: "What is 23 * 47?"
  [Internal thinking -- NOT shown to user]:
    "Let me break this down.
     23 * 47 = 23 * (50 - 3)
     23 * 50 = 1150
     23 * 3 = 69
     1150 - 69 = 1081"
  Output: "1081"
  Compute: ~5x-20x more forward passes (thinking + output tokens)
```

### Key Reasoning Models

| Model | Provider | Key Feature |
|---|---|---|
| o1 | OpenAI | First production reasoning model |
| o3 | OpenAI | Improved reasoning, configurable "effort" |
| o3-mini | OpenAI | Cheaper reasoning, tunable effort levels |
| DeepSeek-R1 | DeepSeek | Open-source reasoning model, shows thinking process |
| Claude with extended thinking | Anthropic | Claude Opus/Sonnet with visible thinking |
| Gemini 2.0 Flash Thinking | Google | Fast reasoning with visible thought process |

### When to Use Reasoning Models

**Good for**:
- Complex math and logic problems
- Multi-step code generation and debugging
- Scientific reasoning
- Planning and strategy
- Tasks where accuracy matters more than speed

**Bad for**:
- Simple tasks (classification, extraction) -- overkill and slow
- Latency-sensitive applications (thinking tokens add seconds to minutes)
- High-volume, low-complexity tasks (cost of thinking tokens adds up)
- Creative writing (thinking doesn't help much with creativity)

### Thinking Tokens and Cost

Thinking tokens are generated and processed but often hidden from the user. They still
consume compute and contribute to cost:

```
User prompt:          200 tokens
Thinking tokens:      2000 tokens (internal, billed but not shown by some models)
Visible response:     300 tokens

Total billed output: 2300 tokens (or 300, depending on provider)
Total latency: dominated by the 2000 thinking tokens
```

For o1/o3, thinking tokens are billed at output token rates. This makes reasoning models
5-20x more expensive per request than standard models for the same visible output length.

### Configurable Effort (o3)

OpenAI's o3 allows you to set a "reasoning effort" level:

- **Low**: Minimal thinking, fast, cheap. Good for moderately complex tasks.
- **Medium**: Balanced thinking.
- **High**: Maximum thinking, slow, expensive. For the hardest problems.

This lets you trade off cost/latency against reasoning depth per-request.

### Engineering Implications

1. **Model routing becomes critical**: Use cheap models for simple tasks, reasoning models
   only when needed. A smart router can cut costs 10x.
2. **Timeout handling**: Reasoning can take 30-120+ seconds. Your application needs
   appropriate timeouts and UX patterns (progress indicators, streaming partial results).
3. **Prompt design differs**: Reasoning models often perform better with less prompt
   engineering. Overly detailed instructions can actually conflict with the model's own
   reasoning process. Let the model think.
4. **Evaluation changes**: You need to evaluate both the final answer AND the reasoning
   process. A correct answer with flawed reasoning is a red flag.

---

## Model Distillation

Training a smaller "student" model to replicate the behavior of a larger "teacher" model.

### How Distillation Works

```
Teacher Model (e.g., GPT-4, 1.8T params)
    |
    | Generate high-quality outputs for training prompts
    v
Training Dataset: (prompt, teacher_response) pairs
    |
    | Train student model to match teacher outputs
    v
Student Model (e.g., 7B params)
    Produces similar quality to teacher at a fraction of the cost
```

**Soft label distillation**: Instead of just matching the teacher's final output (hard
labels), the student learns from the teacher's full probability distribution over the
vocabulary (soft labels). This transfers more information -- including the teacher's
uncertainty and alternative choices.

```
Teacher output distribution for next token:
  "is": 0.45, "was": 0.30, "will": 0.10, "has": 0.08, ...

Hard label: student learns "is" is correct
Soft label: student learns the FULL distribution -- much richer signal
```

### Distillation Approaches

**Output distillation**: Student learns from teacher's generated text.
- Simplest approach
- Can be done via APIs (don't need teacher weights)
- Limited by teacher's output quality

**Logit distillation**: Student matches teacher's output probability distributions.
- Requires access to teacher's logits
- Richer learning signal
- Not possible with closed-source APIs (typically)

**Feature distillation**: Student matches teacher's internal representations.
- Requires access to teacher's intermediate activations
- Most information transfer
- Only possible with open-weight models

### When to Use Distillation

| Scenario | Why Distillation |
|---|---|
| High-volume production | A distilled 7B model costs 50x less to run than a 175B model |
| Latency-critical | Smaller models are faster (fewer FLOPs per token) |
| Edge deployment | Distilled models fit on consumer hardware |
| Domain specialization | Distill a generalist's knowledge for a specific domain |
| Cost reduction | Replace expensive API calls with a self-hosted distilled model |

### Real-World Examples

- **GPT-4o-mini**: Likely distilled from GPT-4o (OpenAI hasn't confirmed exact method)
- **Phi-3**: Microsoft's small models trained on GPT-4 generated data
- **DeepSeek-R1 distilled models**: 7B/14B/32B models distilled from the full R1
- **Llama 3.2 1B/3B**: Distilled from Llama 3.1 70B/405B
- **Gemma**: Google's small models benefiting from Gemini's capabilities

### Distillation vs Fine-Tuning

| | Distillation | Fine-Tuning |
|---|---|---|
| Goal | Replicate a larger model's general ability | Adapt to a specific task/domain |
| Teacher | Large, capable model | Not applicable (uses labeled data) |
| Data | Teacher-generated outputs | Task-specific labeled examples |
| Result | Smaller model, broad capability | Same-size model, specialized capability |
| When | Need a cheaper model for general tasks | Need domain-specific behavior |

### Important Caveats

**Terms of Service**: Many API providers explicitly prohibit using their model outputs to
train competing models. OpenAI's ToS, for example, restricts using outputs to develop
models that compete with OpenAI. Check the ToS before distilling from commercial APIs.

**Quality ceiling**: A distilled model cannot exceed its teacher's capability. It can only
approach it. For the highest quality, you still need the largest models.

**Distribution shift**: If the student is deployed on inputs very different from the
distillation dataset, quality can degrade more sharply than with the original large model.
Larger models tend to be more robust to distribution shift.

### Interview Application

When asked "How would you reduce inference costs for a production LLM application?":

1. Start with model routing (use cheap models for easy tasks)
2. Apply quantization to self-hosted models
3. Consider distillation for high-volume, well-defined task patterns
4. Use prompt caching for repeated prefixes
5. Optimize context lengths (shorter prompts = cheaper)
6. Evaluate batch processing for non-real-time workloads
