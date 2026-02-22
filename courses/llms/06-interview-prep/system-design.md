# LLM System Design Interview Guide

## Framework for LLM System Design Questions

When asked to "design an LLM-powered system," use this framework to structure your answer. It works for any scenario — customer support bot, document Q&A, code assistant, content moderation, etc.

---

## Step 1: Clarify Requirements (2 minutes)

Ask these questions before designing:

**Functional:**
- What's the core task? (Q&A, generation, classification, action-taking?)
- What knowledge does it need? (Company-specific, general, real-time?)
- What actions can it take? (Read-only, or can it modify state?)
- What's the user interface? (Chat, API, embedded, batch?)

**Non-Functional:**
- Latency requirements? (Real-time chat vs. background processing)
- Scale? (10 users vs. 10M users)
- Accuracy requirements? (Casual helpfulness vs. zero-tolerance for errors)
- Cost sensitivity? (Startup exploring vs. enterprise at scale)

**Safety:**
- What can go wrong? (Bad advice, data leaks, harmful content)
- Who are the users? (Internal team vs. public-facing)
- Compliance requirements? (PII, HIPAA, SOC2)

---

## Step 2: High-Level Architecture (3 minutes)

Draw the big picture. Most LLM systems follow one of these patterns:

### Pattern A: Simple Prompt (Classification, Extraction)

```
User Input → Prompt Assembly → LLM → Parse Output → Response
```

**Use when:** Task is well-defined, no external knowledge needed, single-step.

### Pattern B: RAG (Knowledge-Grounded Q&A)

```
User Input → Query Processing → Retrieval → Prompt + Context → LLM → Response
```

**Use when:** The system needs domain-specific or up-to-date knowledge.

### Pattern C: Agent (Multi-Step, Tool Use)

```
User Input → Agent Loop (LLM ↔ Tools) → Response
```

**Use when:** The system needs to take actions, query multiple sources, or reason through multi-step problems.

### Pattern D: Pipeline (Complex Processing)

```
Input → Stage 1 (Classify) → Stage 2 (Retrieve) → Stage 3 (Generate) → Stage 4 (Validate) → Output
```

**Use when:** Complex tasks benefit from decomposition into specialized stages.

Most real systems are **combinations** — e.g., a customer support bot might use:
- Classification (Pattern A) for intent routing
- RAG (Pattern B) for knowledge questions
- Agent (Pattern C) for account actions

---

## Step 3: Deep Dive on Key Components (5-10 minutes)

For each component, discuss **what**, **why**, and **trade-offs**.

### Knowledge Layer (if applicable)

```
Documents → Chunking → Embedding → Vector DB
                                        ↑
User Query → Embedding → Similarity Search → Top-K Chunks → Prompt
```

**Discuss:**
- Chunking strategy and chunk size (trade-off: precision vs. context)
- Embedding model choice
- Vector DB selection (Pinecone for managed, pgvector for simplicity, etc.)
- Hybrid search if exact matching matters
- Reranking for improved precision
- How you'll keep the knowledge base updated

### Model Selection

| Component | Model Tier | Reasoning |
|---|---|---|
| Intent classification | Small/cheap | High volume, simple task |
| Knowledge Q&A | Medium | Needs comprehension, moderate volume |
| Complex reasoning | Large | Accuracy-critical, lower volume |
| Embedding | Embedding model | Specialized, very cheap |

### Prompt Design

- System prompt: role, constraints, output format
- Context injection: how retrieved content is formatted
- Output structure: JSON for programmatic consumption, text for user-facing

### Safety & Guardrails

- Input filtering (PII detection, injection prevention)
- Output validation (schema checking, content filtering)
- Tool permissions (what actions require human approval?)
- Fallback behavior (what happens when the model fails?)

---

## Step 4: Production Considerations (2-3 minutes)

### Reliability
- **Retry logic** with exponential backoff for API failures
- **Fallback models** (if primary model is down, use backup)
- **Graceful degradation** (if RAG fails, acknowledge the limitation)
- **Circuit breakers** for downstream dependencies

### Performance
- **Streaming** for real-time chat interfaces
- **Caching** for repeated or similar queries
- **Async processing** for non-real-time tasks
- **Model routing** — cheap model for simple tasks, expensive for complex

### Cost
- **Token optimization** — concise prompts, summarized history
- **Model tiering** — right-size per task
- **Caching** — avoid redundant API calls
- **Monitoring** — track cost per request, per feature

### Observability
- **Log** all inputs, outputs, model parameters, latency, cost
- **Trace** multi-step pipelines end-to-end
- **Alert** on error rate spikes, cost anomalies, latency degradation
- **Eval dashboards** — track quality metrics over time

---

## Step 5: Evaluation Strategy (2 minutes)

- **Offline evals:** Curated test set, run before every deployment
- **Online metrics:** User satisfaction, task completion rate, escalation rate
- **LLM-as-judge:** Automated quality scoring for open-ended outputs
- **Regression testing:** Catch quality drops from prompt or model changes

---

## Example: Design a Customer Support Bot

Here's the framework applied to a common interview question.

### Requirements
- Answer product and policy questions from a knowledge base
- Perform account actions (check order status, initiate refunds)
- Handle 10K conversations/day
- Public-facing — safety is critical

### Architecture

```
┌─────────────────────────────────────────────┐
│                 User Message                 │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│        Intent Classification (cheap model)   │
│  → billing | technical | account | general   │
└───────┬────────────┬────────────┬───────────┘
        ↓            ↓            ↓
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Knowledge  │ │  Account   │ │  Fallback  │
│ RAG Path   │ │ Agent Path │ │  (escalate │
│            │ │            │ │  to human) │
└───────────┘ └───────────┘ └───────────┘
        ↓            ↓            ↓
┌─────────────────────────────────────────────┐
│          Response + Safety Filter            │
└─────────────────────────────────────────────┘
```

### Knowledge Path (RAG)
- **Ingestion:** Chunk support docs (recursive, ~500 tokens, 10% overlap), embed with `text-embedding-3-small`, store in Pinecone
- **Retrieval:** Embed query → vector search (top 5) → rerank (top 3) → inject into prompt
- **Generation:** System prompt constrains the model to only use provided sources, cite them, and say "I don't know" if unsure

### Account Path (Agent)
- **Tools:** `lookup_customer`, `get_order_status`, `initiate_refund`
- **Permissions:** Read operations automatic; writes (refund) require confirmation
- **Loop:** Max 5 iterations, circuit breaker on 3 consecutive errors

### Safety
- **Input:** PII detection (redact before logging), injection defense (delimiters)
- **Output:** Content filter, never expose internal system details
- **Actions:** Refunds require explicit user confirmation
- **Escalation:** Low confidence → hand off to human agent

### Evaluation
- **Test set:** 200 curated (question, expected_answer) pairs across all intents
- **Metrics:** Answer accuracy, intent classification accuracy, resolution rate
- **Online:** Customer satisfaction scores, escalation rate, average handle time

---

## Common LLM System Design Questions

| Question | Key Patterns |
|---|---|
| Customer support bot | RAG + Agent + Intent routing |
| Document Q&A system | RAG + Citation |
| Code assistant | RAG (codebase) + Agent (tools) + Streaming |
| Content moderation | Classification pipeline + Human-in-the-loop |
| Search engine with AI answers | RAG + Hybrid search + Summarization |
| Data analysis assistant | Agent + Tool use (SQL, charts) |
| Meeting summarizer | Pipeline (transcribe → extract → summarize) |
| Email drafting assistant | RAG (context) + Generation + Tone control |

---

## Tips for the Interview

1. **Start with requirements** — don't jump into architecture without understanding the problem
2. **Draw a diagram** — even in a verbal interview, describe the data flow clearly
3. **Discuss trade-offs** — every decision has pros and cons; show you think about them
4. **Start simple, add complexity** — "I'd start with X and add Y if Z becomes a problem"
5. **Mention evaluation early** — shows you think about quality, not just building
6. **Be honest about what you don't know** — "I haven't used X in production, but I understand it does Y"
7. **Connect to real experience** — "In a similar system I worked on, we found that..."
