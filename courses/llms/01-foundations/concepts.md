# How LLMs Work

## The 30-Second Explanation

Large Language Models are neural networks trained on massive text corpora to predict the next token in a sequence. They learn statistical patterns in language — grammar, facts, reasoning patterns, code structure — and generate text by repeatedly predicting "what comes next" given everything before it.

---

## Transformers

The transformer architecture (Vaswani et al., 2017 — "Attention Is All You Need") is the foundation of all modern LLMs.

**Key idea:** Instead of processing text sequentially (like RNNs), transformers process all tokens in parallel and use *attention* to learn which tokens are relevant to each other.

**Architecture overview:**

```
Input Text
    ↓
Tokenization → Token IDs → Token Embeddings + Positional Encodings
    ↓
┌─────────────────────────────┐
│   Transformer Block (×N)    │  ← Stack of identical layers
│  ┌───────────────────────┐  │
│  │  Multi-Head Attention  │  │  ← "Which tokens matter for this token?"
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │   Feed-Forward Network │  │  ← "What does this combination mean?"
│  └───────────────────────┘  │
│  + Layer Norm + Residuals   │
└─────────────────────────────┘
    ↓
Output Probabilities (over entire vocabulary)
```

**Why it matters for interviews:** Understanding transformers lets you reason about context window limits, why certain tasks are hard (e.g., counting), and why prompt structure matters.

---

## Tokenization

LLMs don't see characters or words — they see **tokens**, which are subword units learned from the training data.

**How it works:**
- Algorithms like BPE (Byte Pair Encoding) or SentencePiece build a vocabulary by merging frequent character pairs
- Common words become single tokens; rare words get split into pieces
- Typical vocabulary: 32K–100K tokens

**Examples (approximate, varies by model):**
```
"Hello world"     → ["Hello", " world"]           (2 tokens)
"tokenization"    → ["token", "ization"]           (2 tokens)
"🎉"             → ["🎉"]                         (1 token, sometimes more)
"XMLHttpRequest"  → ["XML", "Http", "Request"]     (3 tokens)
```

**Why it matters:**
- **Cost** — you're billed per token (input + output)
- **Context limits** — measured in tokens, not words (~0.75 words per token for English)
- **Edge cases** — tokenization affects math (numbers split unpredictably), code (whitespace-sensitive), and non-English text (more tokens per word)

---

## Attention Mechanism

Attention is *the* core innovation that makes transformers work. It lets each token "look at" every other token and decide how much to weight each one.

**Self-attention in plain English:**

For the input "The cat sat on the mat because **it** was tired":
- The token "it" needs to figure out what it refers to
- Attention computes a relevance score between "it" and every other token
- "cat" gets a high score → the model understands "it" = "cat"

**How it's computed:**

Each token produces three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

The `QKᵀ` dot product measures relevance. Softmax normalizes into weights. Those weights are applied to Values to produce the output.

**Multi-head attention:** Run multiple attention computations in parallel (each "head" can focus on different relationship types — syntax, semantics, position, etc.).

**Why it matters:** Attention is why prompt structure matters. Placing instructions near relevant content, using clear delimiters, and structuring prompts well all help the model attend to the right information.

---

## Context Windows

The **context window** is the maximum number of tokens the model can process in a single request (input + output combined).

**Current landscape (approximate):**
| Model Family | Context Window |
|---|---|
| GPT-4o | 128K tokens |
| Claude 3.5+ | 200K tokens |
| Gemini 1.5 Pro | 1M–2M tokens |
| Llama 3 | 8K–128K tokens |

**Key concepts:**

- **Input tokens** — your prompt, system message, conversation history
- **Output tokens** — the model's response (often capped separately, e.g., 4K–8K)
- **Lost in the middle** — models tend to pay more attention to the beginning and end of the context; information in the middle can be overlooked
- **Effective vs. stated context** — a model may accept 128K tokens but perform degraded on tasks requiring precise recall beyond ~32K

**Why it matters:**
- Determines how much history/context you can include
- Directly affects RAG design (how many retrieved chunks fit)
- Long contexts cost more and increase latency

---

## Temperature & Sampling

After the model computes probabilities for the next token, **sampling parameters** control how that distribution is converted into an actual token choice.

### Temperature

Controls randomness. Technically, it scales the logits before softmax:

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

| Temperature | Behavior | Use Case |
|---|---|---|
| 0 | (Near-)deterministic, highest probability token | Factual Q&A, classification, structured output |
| 0.3–0.7 | Balanced creativity and coherence | General conversation, writing |
| 1.0+ | High randomness, more diverse/surprising outputs | Creative writing, brainstorming |

### Top-p (Nucleus Sampling)

Instead of considering all tokens, only sample from the smallest set whose cumulative probability ≥ p.

- `top_p=0.9` → consider tokens comprising the top 90% of probability mass
- Dynamically adjusts the candidate pool — narrow for confident predictions, wider for uncertain ones

### Top-k

Only consider the top k most probable tokens. Simpler but less adaptive than top-p.

### In Practice

- **Deterministic tasks** (extraction, classification): temperature=0
- **Creative tasks**: temperature=0.7–1.0, top_p=0.9
- Don't set both temperature and top_p aggressively — they compound

---

## Embeddings

Embeddings are dense vector representations of text in a high-dimensional space where **semantic similarity maps to geometric proximity**.

**Key properties:**
- Typical dimensions: 256–3072 (depends on model)
- Similar meanings → nearby vectors (measured by cosine similarity)
- Capture semantic meaning, not just lexical overlap

**Example:**
```
embed("king") - embed("man") + embed("woman") ≈ embed("queen")

cosine_similarity(embed("dog"), embed("puppy"))   ≈ 0.92  (high)
cosine_similarity(embed("dog"), embed("bicycle"))  ≈ 0.15  (low)
```

**Where they're used:**
- **Semantic search** — find relevant documents by meaning, not keywords
- **RAG** — retrieve context to augment LLM prompts
- **Clustering** — group similar content
- **Classification** — use as features for downstream models

**Embedding models vs. LLMs:**
- Embedding models (e.g., OpenAI `text-embedding-3-small`, Cohere `embed-v3`) are specialized for producing good vectors
- LLMs generate text; embedding models map text → vectors
- Much cheaper to run than LLMs

---

## Training Pipeline (High Level)

Understanding the training stages helps explain model behavior:

### 1. Pre-training
- Train on massive internet text (books, web, code)
- Objective: predict the next token
- Result: a model that can complete text but isn't "helpful"
- This is where most factual knowledge is learned

### 2. Supervised Fine-Tuning (SFT)
- Train on curated (prompt, response) pairs
- Teaches the model to follow instructions and be helpful
- Relatively small dataset compared to pre-training

### 3. RLHF / RLAIF (Alignment)
- **RLHF:** Reinforcement Learning from Human Feedback
- **RLAIF:** ...from AI Feedback
- Human raters rank model outputs → train a reward model → optimize the LLM against it
- This is what makes models refuse harmful requests, be honest about uncertainty, etc.

### 4. (Optional) Domain Fine-Tuning
- Further fine-tune on domain-specific data
- Useful for specialized tasks (legal, medical, code)

**Why it matters:** Training explains why models hallucinate (pre-training optimizes for plausible text, not truth), why they're "helpful" (SFT + RLHF), and the limits of fine-tuning vs. prompting.

---

## Key Mental Models

1. **LLMs are probabilistic text completers** — not databases, not reasoning engines. Their "reasoning" emerges from pattern completion over training data.

2. **The context window is the model's working memory** — everything it needs must be in the prompt or its weights. It has no persistent memory between calls.

3. **Tokens are the atomic unit** — cost, limits, and many edge cases tie back to tokenization.

4. **Garbage in, garbage out** — prompt quality directly determines output quality. The model is extremely sensitive to how you ask.

5. **Models don't "know what they don't know"** — they generate confident-sounding text even when wrong (hallucination). Always verify factual claims for critical applications.
