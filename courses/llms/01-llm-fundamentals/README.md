# Module 01: LLM Fundamentals

Core knowledge for AI engineering interviews. Every section maps to questions you will be asked.

---

## Table of Contents

1. [Transformer Architecture](#transformer-architecture)
2. [Tokenization](#tokenization)
3. [Attention Mechanism](#attention-mechanism)
4. [Context Windows](#context-windows)
5. [Training Pipeline](#training-pipeline)
6. [Model Families Comparison](#model-families-comparison)
7. [Key Parameters](#key-parameters)
8. [Embeddings](#embeddings)
9. [Mental Models for Interviews](#mental-models-for-interviews)

---

## Transformer Architecture

The transformer (Vaswani et al., 2017 -- "Attention Is All You Need") replaced RNNs and LSTMs as
the dominant architecture for sequence modeling. Every major LLM is built on it.

### Why Transformers Won

RNNs process tokens sequentially -- token 500 must wait for tokens 1-499. Transformers process all
tokens in parallel via self-attention, then stack layers deep. This parallelism maps perfectly to
GPU hardware, enabling training on trillions of tokens.

Think of it this way: an RNN is like a `for` loop processing a linked list. A transformer is like a
matrix multiplication -- the same operation expressed in a way that hardware can parallelize.

### Architecture Overview

```
Input: "The cat sat on the mat"
    |
    v
[Tokenizer] --> token IDs: [464, 3797, 3332, 319, 262, 2603]
    |
    v
[Token Embedding Table]  +  [Positional Encoding]
    |                            |
    +----------------------------+
    |
    v
+=========================================+
|       Transformer Block  (x N)          |  N = 32 for 7B, 80 for 175B, etc.
|                                         |
|  +-----------------------------------+  |
|  |  Layer Norm                        |  |
|  +-----------------------------------+  |
|  |  Multi-Head Self-Attention         |  |  <-- "Which tokens matter for each token?"
|  +-----------------------------------+  |
|  |  + Residual Connection             |  |  <-- Skip connection around attention
|  +-----------------------------------+  |
|  |  Layer Norm                        |  |
|  +-----------------------------------+  |
|  |  Feed-Forward Network (FFN)        |  |  <-- 2 linear layers with activation
|  +-----------------------------------+  |
|  |  + Residual Connection             |  |  <-- Skip connection around FFN
|  +-----------------------------------+  |
+=========================================+
    |
    v
[Layer Norm] --> [Linear Projection] --> logits (vocab_size)
    |
    v
[Softmax] --> probability distribution over vocabulary
    |
    v
Next token prediction
```

### Component Breakdown

**Token Embeddings**: A lookup table (matrix of shape `[vocab_size, d_model]`) that maps each
token ID to a dense vector. If your vocab is 100K tokens and `d_model` is 4096, this is a
100K x 4096 matrix. Think of it as a `Map<number, Float32Array>` -- each token ID gets a
learned vector.

**Positional Encoding**: Transformers have no inherent notion of order (unlike RNNs). Positional
encodings inject sequence position information. Original paper used sinusoidal functions; modern
models use learned positions or RoPE (rotary position embeddings). Without this, the model
treats "the cat sat on the mat" and "mat the on sat cat the" identically.

**Layer Normalization**: Normalizes activations across the feature dimension (not the batch
dimension like BatchNorm). Stabilizes training in deep networks. Modern LLMs use "Pre-LN" --
normalize before each sub-layer, not after. This is critical for training stability at scale.

**Residual Connections**: The `x + sublayer(x)` pattern. Identical to skip connections in
ResNets. They solve the vanishing gradient problem in deep networks by giving gradients a
direct path backward through the network. Without residuals, a 96-layer transformer would be
untrainable.

**Feed-Forward Network (FFN)**: Two linear transformations with a nonlinearity:

```
FFN(x) = W2 * activation(W1 * x + b1) + b2
```

Typically `W1` projects from `d_model` to `4 * d_model` (expansion), then `W2` projects back.
This is where a large portion of the model's factual knowledge is stored -- think of it as a
learned key-value memory. Modern models use SwiGLU or GeGLU activations instead of ReLU.

**Output Head**: Final layer norm, then a linear projection to vocabulary size, producing logits
(unnormalized scores) for each token in the vocabulary.

### Decoder-Only vs Encoder-Decoder

| Architecture | Models | Key Difference |
|---|---|---|
| Decoder-only | GPT-4, Claude, Llama, Mistral | Causal attention mask (can only look left). Generates token by token. |
| Encoder-decoder | T5, BART, original Transformer | Encoder sees full input bidirectionally; decoder generates autoregressively. |
| Encoder-only | BERT, RoBERTa | Bidirectional attention. Used for embeddings/classification, not generation. |

Almost all modern LLMs are decoder-only. The causal mask means each token can only attend to
previous tokens -- this is what makes left-to-right generation work.

### Interview Anchor

> "A transformer block is self-attention (dynamic routing of information between tokens) followed
> by a feed-forward network (per-token learned transformation), with residual connections and
> layer norms for training stability. Stack N of these blocks, and you get an LLM."

---

## Tokenization

LLMs operate on tokens, not characters or words. Tokenization is the process of converting raw
text into a sequence of integer IDs from a fixed vocabulary.

### Tokenization Algorithms

**BPE (Byte Pair Encoding)** -- Used by GPT models (via tiktoken), Llama, Mistral:
1. Start with individual bytes/characters as the base vocabulary
2. Count all adjacent pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size (e.g., 100K)

**WordPiece** -- Used by BERT, some older models:
- Similar to BPE but uses likelihood maximization instead of frequency for merges
- Prefixes subwords with `##` to indicate continuation

**SentencePiece** -- Used by Llama, Gemini, T5:
- Treats input as a raw byte stream (no pre-tokenization)
- Supports both BPE and unigram models
- Language-agnostic -- doesn't assume whitespace-separated words

**Unigram** -- Used by SentencePiece's unigram mode:
- Starts with a large vocabulary, iteratively removes tokens that contribute least to
  likelihood
- Probabilistic -- can produce multiple tokenizations for the same input

### tiktoken vs SentencePiece

| Feature | tiktoken (OpenAI) | SentencePiece (Google/Meta) |
|---|---|---|
| Used by | GPT-3.5/4, Claude (similar) | Llama, Gemini, T5 |
| Base approach | Byte-level BPE | BPE or Unigram on raw bytes |
| Pre-tokenization | Regex-based splitting first | None (raw byte stream) |
| Speed | Very fast (Rust implementation) | Fast (C++ implementation) |
| Whitespace | Explicit space tokens | Uses special char for spaces |

### Token Examples

```
Model: GPT-4 (cl100k_base vocabulary, ~100K tokens)

"Hello, world!"        -->  ["Hello", ",", " world", "!"]                    = 4 tokens
"implementation"       -->  ["implement", "ation"]                           = 2 tokens
"XMLHttpRequest"       -->  ["XML", "Http", "Request"]                       = 3 tokens
" "                    -->  [" "]                                            = 1 token
"    "  (4 spaces)     -->  ["    "]                                         = 1 token
"こんにちは"            -->  ["こん", "にち", "は"]                             = 3 tokens
"123456"               -->  ["123", "456"]                                   = 2 tokens
"<|endoftext|>"        -->  ["<|endoftext|>"]                                = 1 special token
```

### Why Tokenization Matters in Practice

**Cost**: You pay per token. Non-English text and code often produce more tokens per semantic
unit. A 1000-word English document might be ~1300 tokens, but the same content in Japanese
could be 2000+ tokens.

**Context budget**: Your context window is measured in tokens. A JSON blob with verbose keys
burns tokens fast: `{"user_email_address": "..."}` costs more than `{"email": "..."}`.

**Model behavior quirks**:
- Arithmetic is hard because numbers get split unpredictably: "12345" might become ["123", "45"]
- Spelling tasks fail because the model doesn't see individual characters
- Counting characters/words is approximate because they don't map 1:1 to tokens

**Special tokens**: Models use reserved tokens for structure:
- `<|im_start|>`, `<|im_end|>` -- message boundaries (ChatML format)
- `<|endoftext|>` -- document separator
- `[INST]`, `[/INST]` -- instruction delimiters (Llama)
- `<tool_call>`, `</tool_call>` -- function calling delimiters

### Rough Estimation Rules

```
English text:  1 token ~= 0.75 words  (~= 4 characters)
Code:          1 token ~= 0.4 words   (more tokens due to syntax/whitespace)
JSON:          Heavily depends on key verbosity
Non-English:   1.5x to 3x more tokens than equivalent English
```

---

## Attention Mechanism

Attention is the core innovation that makes transformers work. It allows every token to
dynamically route information from every other token, creating a fully connected information
graph at each layer.

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V
```

Step by step:

```
Input: sequence of token embeddings X  [seq_len, d_model]

1. Project to Q, K, V:
   Q = X * W_Q    [seq_len, d_k]      "What am I looking for?"
   K = X * W_K    [seq_len, d_k]      "What do I contain?"
   V = X * W_V    [seq_len, d_v]      "What information do I provide if matched?"

2. Compute attention scores:
   scores = Q * K^T                    [seq_len, seq_len]
   -- Each entry (i,j) = how much token i should attend to token j

3. Scale:
   scores = scores / sqrt(d_k)
   -- Prevents dot products from growing large with dimension,
   -- which would push softmax into saturated regions with near-zero gradients

4. (Optional) Mask:
   scores[i][j] = -inf  where j > i   (causal mask for autoregressive models)
   -- Ensures token i cannot peek at future tokens j > i

5. Normalize:
   weights = softmax(scores)           [seq_len, seq_len]
   -- Each row sums to 1; these are the attention weights

6. Weighted sum:
   output = weights * V                [seq_len, d_v]
   -- Each token's output is a weighted combination of all Value vectors
```

**Analogy for the audience**: Think of Q/K/V like a database query. Q is your query, K is the
index, and V is the stored data. The dot product Q*K^T is the relevance score, softmax
normalizes it into weights, and the weighted sum of V is your result. Except here, every token
is simultaneously a query AND a key-value entry.

### Multi-Head Attention

Instead of one attention computation, run `h` attention heads in parallel:

```
head_i = Attention(X * W_Qi, X * W_Ki, X * W_Vi)

MultiHead(X) = Concat(head_1, ..., head_h) * W_O
```

Each head uses smaller dimensions: `d_k = d_model / h`. So with `d_model=4096` and `h=32`,
each head operates on 128-dimensional projections.

**Why multiple heads?** Different heads learn different relationship patterns:
- Head 3 might track syntactic dependencies (subject-verb agreement)
- Head 12 might track coreference ("it" -> "the cat")
- Head 27 might focus on positional proximity

This is like having multiple specialized indices on your database instead of one.

### Causal Masking

In decoder-only models, the attention mask prevents looking at future tokens:

```
Attention mask (4 tokens):

     t1  t2  t3  t4
t1 [  0  -inf -inf -inf ]    t1 can only see t1
t2 [  0    0  -inf -inf ]    t2 can see t1, t2
t3 [  0    0    0  -inf ]    t3 can see t1, t2, t3
t4 [  0    0    0    0  ]    t4 can see t1, t2, t3, t4

(0 = allowed, -inf = masked; after softmax, -inf becomes 0 weight)
```

This is what makes autoregressive generation work -- during training, the model predicts every
next token simultaneously, but each prediction can only use preceding context.

### Attention Complexity

**Time complexity**: O(n^2 * d) where n = sequence length, d = dimension
**Memory complexity**: O(n^2) for the attention matrix

This is why context windows have limits. Doubling the sequence length quadruples the attention
computation. For a 128K token context with 32 heads, you need to compute and store attention
matrices of size 128K x 128K per head per layer.

**Practical impact**:
- 4K context: baseline
- 32K context: 64x more attention computation than 4K
- 128K context: 1024x more attention computation than 4K

This has driven research into efficient attention: FlashAttention (IO-aware exact attention),
sliding window attention, sparse attention, and linear attention approximations.

### Why Attention is the Key Innovation

Before attention, sequence models had a fixed-size bottleneck -- an RNN had to compress the
entire input into a single hidden state vector. Attention removes this bottleneck by letting the
model access any part of the input directly, with learned relevance weighting.

This is why LLMs can handle instructions like "Given the following 50-page document, answer this
specific question" -- attention can learn to route directly from the question tokens to the
relevant passage tokens, regardless of distance.

---

## Context Windows

The context window is the maximum number of tokens the model can process in a single
request. It includes everything: system prompt, conversation history, user message, and the
model's response.

### How Context Windows Work

Think of the context window as working memory. The model has no persistent state between API
calls -- everything it needs to "know" for this request must be within the context window
(or baked into its weights from training).

```
+---------------------------------------------------+
|              Context Window (128K tokens)          |
|                                                   |
|  [System Prompt]           ~500 tokens            |
|  [Conversation History]    ~2000 tokens           |
|  [Retrieved Documents]     ~8000 tokens           |
|  [User Message]            ~200 tokens            |
|  [Model Response]          ~2000 tokens           |
|                                                   |
|  Used: ~12,700 tokens                             |
|  Available: ~115,300 tokens                       |
+---------------------------------------------------+
```

### Current Context Windows (as of early 2026)

| Model | Context Window | Max Output | Notes |
|---|---|---|---|
| GPT-4o | 128K | 16K | Good balance of speed and capability |
| GPT-4o-mini | 128K | 16K | Fast, cheap, same context |
| o1 / o3 | 128K-200K | 32K-100K | Thinking tokens consume context |
| Claude 3.5 Sonnet | 200K | 8K | Large input, moderate output |
| Claude Opus 4 | 200K | 32K | Extended thinking uses output budget |
| Gemini 2.0 Flash | 1M | 8K | Largest production context |
| Gemini 2.0 Pro | 2M | 8K | Research-scale context |
| Llama 3.3 70B | 128K | varies | Open source, self-hosted |
| Mistral Large | 128K | varies | Strong European alternative |
| DeepSeek-V3 | 128K | 8K | Strong open model, MoE |

### The "Lost in the Middle" Problem

Models don't attend equally across the entire context. Research shows a U-shaped attention
curve: information at the beginning and end of the context gets more attention than information
in the middle.

```
Recall Performance vs. Position in Context:

High  |  *                                      *
      |   *                                    *
      |    **                                **
      |      ***                          ***
      |         ****                  ****
Low   |             ******************
      +------------------------------------------
      Beginning          Middle              End
```

**Practical implication**: Put the most important information (instructions, key constraints)
at the beginning and end of your prompt. Don't bury critical instructions in the middle of
a large context.

### Strategies for Long Contexts

**Chunking + Retrieval (RAG)**:
Don't stuff everything into context. Retrieve only relevant chunks.
Best for: large document collections, knowledge bases.

**Sliding Window**:
Process a long document in overlapping windows, aggregating results.
Best for: summarization, extraction over very long documents.

**Hierarchical Summarization**:
Summarize sections, then summarize summaries. Map-reduce pattern.
Best for: getting a high-level view of large documents.

**Context Compression**:
Use a smaller/faster model to compress conversation history or documents before passing to
the main model.
Best for: long conversations, reducing cost.

**Prompt Caching**:
Both OpenAI and Anthropic cache repeated prompt prefixes. If your system prompt + few-shot
examples are the same across requests, subsequent calls are faster and cheaper.
Best for: high-volume applications with stable system prompts.

---

## Training Pipeline

Understanding how LLMs are trained explains their behavior, capabilities, and limitations.

### Stage 1: Pre-training (Next Token Prediction)

```
Objective: Given tokens [t1, t2, ..., tn], predict t(n+1)
Data: Trillions of tokens from the internet, books, code, etc.
Compute: Thousands of GPUs for weeks to months
Cost: $10M - $100M+ for frontier models
```

The model learns to predict the next token in a sequence. Through this simple objective, it
acquires:
- Grammar and syntax
- Factual knowledge (stored in FFN weights)
- Reasoning patterns (from mathematical/logical text)
- Code structure (from GitHub, Stack Overflow)
- Multilingual capability (from multilingual corpora)

**Key insight**: Pre-training produces a text completion engine, not an assistant. Ask it a
question and it might complete with another question, because that's a likely continuation.
It knows facts but doesn't know to be helpful.

### Stage 2: Supervised Fine-Tuning (SFT)

```
Data: Curated (instruction, response) pairs -- typically 10K-100K examples
Goal: Teach the model to follow instructions and produce helpful responses
```

Human annotators write ideal responses to diverse prompts. The model learns the format:
"When someone asks X, respond like Y." This is what turns a completion engine into a chatbot.

SFT is relatively cheap compared to pre-training -- you're adjusting the model's behavior
pattern, not teaching it new knowledge.

### Stage 3: Alignment (RLHF / DPO / Constitutional AI)

**RLHF (Reinforcement Learning from Human Feedback)**:
1. Generate multiple responses to each prompt
2. Human raters rank them (best to worst)
3. Train a reward model on these rankings
4. Use PPO (Proximal Policy Optimization) to optimize the LLM against the reward model

**DPO (Direct Preference Optimization)**:
- Skip the reward model -- directly optimize from preference pairs
- Simpler, more stable training
- "Given prompt P, response A is better than response B" -> update weights to prefer A
- Increasingly popular as a replacement for PPO-based RLHF

**Constitutional AI (Anthropic's approach)**:
1. Define a set of principles ("constitution")
2. Model generates responses, then self-critiques against the principles
3. Model revises its own responses
4. Train on the revised outputs
- Reduces reliance on human labeling for alignment
- Principles are explicit and auditable

### Stage 4: Specialized Training (Optional)

**Domain fine-tuning**: Further SFT on domain-specific data (legal, medical, financial).

**Tool use training**: Teaching the model to emit structured tool calls (function calling).

**Long-context training**: Extending context window via continued pre-training with long
sequences and adjusted positional encodings.

**Reasoning training**: Training on chain-of-thought traces, process reward models. Used for
o1/o3, DeepSeek-R1.

### The Full Pipeline Visualized

```
Internet Text (trillions of tokens)
    |
    v
[Pre-training] --> Base Model (text completer, not helpful)
    |
    v
[SFT on instruction data] --> Instruction-following Model
    |
    v
[RLHF / DPO / Constitutional AI] --> Aligned Model (helpful, harmless, honest)
    |
    v
[Optional: domain fine-tuning, tool use training, reasoning training]
    |
    v
Production Model (e.g., GPT-4o, Claude Sonnet, etc.)
```

### Why Training Explains Behavior

| Behavior | Explanation |
|---|---|
| Hallucination | Pre-training optimizes for plausible continuations, not truth |
| Instruction following | Learned in SFT stage |
| Refusal to help with harm | Learned in RLHF/alignment stage |
| Knowledge cutoff | Pre-training data has a fixed end date |
| Better at English than other languages | Training data skewed toward English |
| Sycophancy | RLHF can over-optimize for "user satisfaction" vs. accuracy |

---

## Model Families Comparison

Knowing the landscape is critical for system design interviews: "Which model would you use
for X and why?"

### Major Model Families (as of early 2026)

**OpenAI GPT Series**

| Model | Context | Strengths | Weaknesses | Cost (input/output per 1M tokens) |
|---|---|---|---|---|
| GPT-4o | 128K | All-around strong, multimodal, fast | Expensive for high volume | ~$2.50 / $10 |
| GPT-4o-mini | 128K | Very fast, very cheap | Less capable on complex reasoning | ~$0.15 / $0.60 |
| o1 | 200K | Strong reasoning, math, code | Slow, expensive, thinking tokens add cost | ~$15 / $60 |
| o3 | 200K | Best reasoning model | Very expensive, very slow | ~$10 / $40 |
| o3-mini | 200K | Good reasoning, cheaper | Less capable than o3 | ~$1.10 / $4.40 |

**Anthropic Claude Series**

| Model | Context | Strengths | Weaknesses | Cost (input/output per 1M tokens) |
|---|---|---|---|---|
| Claude Opus 4 | 200K | Top-tier reasoning, long context, coding | Most expensive Claude | ~$15 / $75 |
| Claude Sonnet 4 | 200K | Best quality/cost ratio, strong coding | Slower than Haiku | ~$3 / $15 |
| Claude Haiku 3.5 | 200K | Very fast, cheapest Claude | Less capable on complex tasks | ~$0.80 / $4 |

**Google Gemini Series**

| Model | Context | Strengths | Weaknesses | Cost (input/output per 1M tokens) |
|---|---|---|---|---|
| Gemini 2.0 Flash | 1M | Huge context, fast, multimodal | Less precise than Pro | ~$0.10 / $0.40 |
| Gemini 2.0 Pro | 2M | Largest context window available | API access varies | ~$1.25 / $5 |
| Gemini 2.5 Pro | 1M | Strong reasoning, code | Newer, evolving | varies |

**Open Source / Open Weight**

| Model | Context | Strengths | Weaknesses | Cost |
|---|---|---|---|---|
| Llama 3.3 70B | 128K | Strong open model, fine-tunable | Requires infrastructure | Self-hosted |
| Mistral Large 2 | 128K | Strong European model, multilingual | Smaller community | Self-hosted or API |
| DeepSeek-V3 | 128K | MoE, very efficient, strong at code | China-based, policy considerations | Very cheap API or self-hosted |
| DeepSeek-R1 | 128K | Open reasoning model, strong math | Slow (reasoning tokens) | Self-hosted or API |
| Qwen 2.5 72B | 128K | Strong multilingual, code | Alibaba-based | Self-hosted or API |

### Model Selection Decision Framework

```
Task Requirements Analysis:

Is it a simple task?          --> GPT-4o-mini / Haiku / Gemini Flash
  (classification, extraction,
   simple generation)

Does it need reasoning?       --> o3 / o3-mini / DeepSeek-R1 / Claude Opus 4
  (math, logic, complex code,
   multi-step problems)

Does it need huge context?    --> Gemini 2.0 Flash/Pro (1M-2M tokens)
  (entire codebases, long
   document analysis)

Is cost the primary concern?  --> GPT-4o-mini / Gemini Flash / DeepSeek-V3
  (high volume, low margins)

Does it need self-hosting?    --> Llama 3.3 / Mistral / DeepSeek-V3
  (data sovereignty, air-gapped,
   fine-tuning needed)

Best general-purpose?         --> Claude Sonnet 4 / GPT-4o
  (balanced quality/cost/speed)
```

---

## Key Parameters

These parameters control the model's generation behavior. Knowing them at a technical
level (not just "temperature = creativity") is expected in interviews.

### Temperature

Scales the logits (raw model output scores) before softmax:

```
P(token_i) = exp(logit_i / T) / SUM_j(exp(logit_j / T))

T -> 0:  Probability concentrates on the highest-logit token (deterministic)
T = 1:   Original distribution from the model
T > 1:   Flattened distribution (more randomness)
```

**Intuition**: Temperature doesn't change which token the model thinks is best. It changes
how much probability mass spreads to alternatives. At T=0, the model always picks its top
choice. At T=1.5, even low-probability tokens get a real shot.

| Temperature | Use Case | Why |
|---|---|---|
| 0 | Classification, extraction, structured output | Deterministic, reproducible |
| 0.1 - 0.3 | Factual Q&A, code generation | Slight variation, still focused |
| 0.5 - 0.7 | General conversation, writing | Balanced creativity and coherence |
| 0.8 - 1.2 | Creative writing, brainstorming | Diverse, surprising outputs |

### Top-p (Nucleus Sampling)

Sample only from the smallest set of tokens whose cumulative probability >= p.

```
Sorted tokens by probability: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, ...]

top_p = 0.80:  Consider tokens 1-3 (0.40 + 0.25 + 0.15 = 0.80)
               Remaining tokens zeroed out, redistribute probability among top 3

top_p = 0.95:  Consider tokens 1-5 (0.40 + 0.25 + 0.15 + 0.10 + 0.05 = 0.95)
```

**Key property**: Adaptive. When the model is confident (one token has P=0.90), top_p=0.95
considers only ~1 token. When uncertain (flat distribution), it considers many. This is why
top_p is generally preferred over top_k.

### Top-k

Only consider the k highest-probability tokens. Simple but not adaptive -- considers exactly
k tokens whether the model is confident or not.

### Frequency Penalty and Presence Penalty

Both reduce repetition, but differently:

```
frequency_penalty:  Subtract (penalty * count_of_token_so_far) from logit
                    Penalizes proportional to how many times a token has appeared
                    Good for: reducing word repetition in long outputs

presence_penalty:   Subtract a flat penalty if the token has appeared at all
                    Binary -- appeared or not, regardless of count
                    Good for: encouraging topic diversity
```

Typical ranges: 0.0 (off) to 2.0 (strong). Most applications use 0.0-0.5.

### Stop Sequences

Strings that trigger the model to stop generating. Critical for structured outputs:

```
stop_sequences = ["\n\n", "END", "```"]

Model generates: "The answer is 42.\n\n"  --> stops at "\n\n"
```

Without stop sequences, models may generate unwanted continuations (additional examples,
explanations you didn't ask for, etc.).

### Max Tokens

Hard limit on output length. If the model hits this limit, the response is truncated
mid-sentence. Set it appropriately:
- Classification: 10-50
- Short answers: 200-500
- Long generation: 2000-8000
- Extended analysis: 4000-16000

**Cost implication**: You only pay for tokens actually generated, but max_tokens reserves
compute capacity. Some providers charge for the full max_tokens if using certain features.

### Parameter Interaction

```
                 Low Temperature            High Temperature
                 (T < 0.3)                  (T > 0.8)
  +-----------+--------------------------+------------------------+
  | Low top_p |  Very deterministic       |  Slightly random but   |
  | (< 0.5)   |  (classification)         |  constrained           |
  +-----------+--------------------------+------------------------+
  | High top_p|  Mostly deterministic     |  Very creative/random  |
  | (> 0.9)   |  with occasional variety  |  (brainstorming)       |
  +-----------+--------------------------+------------------------+

  Rule: Don't set both temperature AND top_p aggressively.
  Pick one to tune; leave the other at default.

  OpenAI default: temperature=1, top_p=1
  Anthropic default: temperature=1, top_p varies
```

---

## Embeddings

Embeddings map text to dense vectors in a high-dimensional space where semantic similarity
corresponds to geometric proximity.

### What Embeddings Are

An embedding model takes text and produces a fixed-size vector (typically 256-3072 dimensions).
The model is trained so that semantically similar texts produce vectors that are close together.

```
embed("How do I reset my password?")  -->  [0.023, -0.187, 0.445, ..., 0.091]  (1536 dims)
embed("I forgot my login credentials") --> [0.019, -0.201, 0.438, ..., 0.087]  (1536 dims)
embed("Best pizza in New York")        --> [-0.312, 0.044, -0.156, ..., 0.223] (1536 dims)

cosine_similarity(password_reset, forgot_login) = 0.94  (very similar)
cosine_similarity(password_reset, pizza)        = 0.12  (unrelated)
```

**Analogy**: If you've used feature vectors in ML or TF-IDF vectors in search, embeddings are
the deep learning version -- they capture semantic meaning, not just keyword overlap.

### Embedding Models vs Generation Models

| | Embedding Models | Generation Models (LLMs) |
|---|---|---|
| Input | Text | Text (prompt) |
| Output | Fixed-size vector | Variable-length text |
| Use case | Search, similarity, clustering | Generation, Q&A, reasoning |
| Cost | Very cheap (~$0.02 per 1M tokens) | 10-1000x more expensive |
| Examples | text-embedding-3-small, Cohere embed-v3 | GPT-4o, Claude Sonnet |
| Architecture | Usually encoder-only or dual encoder | Decoder-only |

### Similarity Metrics

**Cosine Similarity**: Measures the angle between two vectors. Ranges from -1 to 1 (in
practice, 0 to 1 for most embedding models). Standard choice for text embeddings.

```
cos_sim(A, B) = (A . B) / (||A|| * ||B||)
```

**Dot Product**: Like cosine similarity but affected by vector magnitude. Use when magnitude
carries meaning (e.g., relevance strength).

**Euclidean Distance**: Straight-line distance. Less common for text embeddings but used in
some clustering applications.

### Where Embeddings Are Used

1. **Semantic search / RAG**: Embed query and documents, find nearest neighbors
2. **Clustering**: Group similar content (support tickets, feedback)
3. **Classification**: Use embedding as features for a classifier
4. **Deduplication**: Find near-duplicate content
5. **Recommendation**: "Users who liked X" via embedding similarity
6. **Anomaly detection**: Flag content far from known clusters

### Key Embedding Models

| Model | Dimensions | Max Tokens | Provider | Notes |
|---|---|---|---|---|
| text-embedding-3-small | 1536 | 8191 | OpenAI | Good default, cheap |
| text-embedding-3-large | 3072 | 8191 | OpenAI | Higher quality, more storage |
| embed-v3 | 1024 | 512 | Cohere | Strong multilingual |
| voyage-3 | 1024 | 32K | Voyage AI | Long context embeddings |
| BGE-M3 | 1024 | 8192 | BAAI | Open source, multilingual |
| Gemini embedding | 768 | 2048 | Google | Integrated with Google Cloud |

---

## Mental Models for Interviews

These are the conceptual anchors that help you reason about any LLM question.

### 1. LLMs Are Probabilistic Text Completers

Not databases. Not reasoning engines. They predict the next token given previous tokens.
"Reasoning" emerges from pattern matching over the training distribution. When a model
"reasons" through a math problem, it's reproducing reasoning patterns it learned from
training data -- which is powerful but fundamentally different from symbolic computation.

### 2. The Context Window Is Working Memory

Everything the model needs must be in the context window or encoded in its weights. There is
no persistent memory between API calls (unless you build it). This is why conversation history
management, RAG, and context window optimization are core engineering concerns.

### 3. Tokens Are the Atomic Unit

Cost, context limits, latency, and many edge cases tie back to tokenization. When someone
asks "why can't the model count letters in a word?" -- it's tokenization. When costs spike
unexpectedly -- check token counts. When non-English performance degrades -- it's tokenization
efficiency.

### 4. Training Stage Determines Behavior

Pre-training gives knowledge. SFT gives instruction-following. Alignment gives safety and
helpfulness. Understanding which stage is responsible for a behavior tells you how to
address issues -- prompt engineering works for instruction-following, fine-tuning addresses
knowledge gaps, and alignment issues need training-level interventions.

### 5. All LLM Applications Are Prompt Engineering + Orchestration

At the application layer, you're combining: prompt design, model selection, context management
(RAG, history), output parsing, error handling, and tool/agent orchestration. The model itself
is a component -- the engineering is in how you wire it together.

### 6. Cost = f(Input Tokens, Output Tokens, Model Tier)

```
cost = (input_tokens * input_price) + (output_tokens * output_price)
```

Output tokens are typically 2-6x more expensive than input tokens. Reducing output length
(via specific instructions, stop sequences, max_tokens) often matters more than reducing
input length. Prompt caching reduces input costs for repeated prefixes.
