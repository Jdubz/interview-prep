# Interview Questions & Answers

Organized by topic. Each question includes a concise answer suitable for a verbal interview response, plus key points to hit.

---

## Foundations

### Q: How do LLMs work at a high level?

**Answer:** LLMs are neural networks based on the transformer architecture, trained on massive text corpora to predict the next token in a sequence. They process input through layers of self-attention (which lets each token consider relationships with all other tokens) and feed-forward networks. The output is a probability distribution over the vocabulary, and sampling parameters like temperature control how that distribution is converted to actual text.

**Key points to hit:**
- Transformer architecture, self-attention mechanism
- Next-token prediction as the core training objective
- Pre-training → fine-tuning → alignment (RLHF) pipeline
- Probabilistic, not deterministic lookup

### Q: What's the difference between tokens and words?

**Answer:** Tokens are subword units — the actual atomic pieces the model processes. Algorithms like BPE split text into common subword chunks. "Tokenization" might be two tokens ("token" + "ization"), while "the" is one. This matters because: pricing is per-token, context windows are measured in tokens, and tokenization affects how well the model handles things like math, code, and non-English text.

### Q: Explain the attention mechanism.

**Answer:** Attention lets each token compute a relevance score against every other token in the sequence. Each token produces a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what information do I provide?"). The dot product of Queries and Keys gives relevance scores, which are normalized via softmax and used to weight the Values. Multi-head attention runs this in parallel multiple times so different heads can capture different relationship types — syntax, semantics, coreference, etc.

### Q: What are embeddings and why are they useful?

**Answer:** Embeddings are dense vector representations of text where semantic similarity maps to geometric proximity — "king" and "queen" are close, "king" and "bicycle" are far. They're produced by specialized embedding models (cheaper and faster than LLMs) and are fundamental to semantic search, RAG, clustering, and classification. Typical dimensions range from 256 to 3072.

### Q: What is a context window and what are its practical implications?

**Answer:** The context window is the maximum number of tokens (input + output) a model can process in a single call. It's effectively the model's working memory. Practical implications: it determines how much conversation history, retrieved documents, or examples you can include. There's also the "lost in the middle" phenomenon — models attend more to the beginning and end, so placement matters. And longer contexts increase cost and latency.

### Q: What is hallucination and how do you mitigate it?

**Answer:** Hallucination is when a model generates plausible-sounding but factually incorrect information. It happens because models optimize for *probable* text, not *true* text. Mitigation strategies:
1. **RAG** — ground responses in retrieved source documents
2. **Constrained output** — force structured output, limit scope
3. **Temperature 0** — reduce randomness for factual tasks
4. **Ask the model to cite sources** — easier to verify
5. **Self-consistency** — generate multiple answers and check agreement
6. **Human-in-the-loop** — final human review for critical applications

---

## Prompt Engineering

### Q: What are the key prompt engineering techniques you use?

**Answer:** The core techniques I reach for:
1. **System prompts** — set role, constraints, output format
2. **Few-shot examples** — anchor the model's understanding of the task
3. **Chain-of-thought** — improve reasoning by requiring step-by-step thinking
4. **Delimiters** — separate instructions from data (prevents injection, aids parsing)
5. **Output structuring** — enforce JSON/schema for programmatic consumption
6. **Prompt chaining** — break complex tasks into reliable smaller steps

The choice depends on the task: classification gets few-shot + structured output, complex reasoning gets CoT + chaining, extraction gets delimiters + JSON.

### Q: When would you use few-shot vs. zero-shot prompting?

**Answer:** Zero-shot works well for tasks the model has seen extensively in training — standard classification, summarization, translation. I start with zero-shot because it's cheaper (fewer tokens) and simpler. I switch to few-shot when: the task has ambiguous boundaries that examples clarify, I need a very specific output format, or the domain is specialized enough that the model needs calibration. Usually 3-5 diverse, balanced examples are enough.

### Q: How do you handle prompt injection?

**Answer:** Prompt injection is when user-provided content manipulates the model's behavior. Defenses:
1. **Delimiters** — clearly separate instructions from user data with XML tags or markers
2. **Input validation** — sanitize/filter user input before including it in prompts
3. **Instruction hierarchy** — system prompts are harder to override than user messages
4. **Output validation** — verify the response conforms to expected format/content
5. **Separate models** — use one model to check if input looks like an injection attempt
6. **Principle of least privilege** — limit what actions the model can trigger

No single defense is bulletproof; use defense in depth.

### Q: What's the difference between temperature and top_p?

**Answer:** Both control randomness in token selection. **Temperature** scales the logits before softmax — 0 is deterministic, higher values flatten the distribution. **Top-p (nucleus sampling)** restricts the candidate pool to tokens whose cumulative probability reaches p (e.g., 0.9 = top 90%). Top-p is adaptive — it narrows when the model is confident and widens when uncertain. In practice, I adjust temperature for most tasks and leave top-p at the default, since they compound when both are set aggressively.

---

## RAG (Retrieval-Augmented Generation)

### Q: What is RAG and why would you use it?

**Answer:** RAG augments an LLM's generation with relevant documents retrieved at query time. Instead of relying solely on the model's training data, you: (1) embed the user's query, (2) search a vector database for similar documents, (3) include those documents in the prompt as context. This lets you give the model up-to-date or domain-specific information without fine-tuning, reduces hallucination by grounding responses in source material, and gives you control over what knowledge the model uses.

### Q: Walk me through a RAG pipeline.

**Answer:**
1. **Ingestion:** Split documents into chunks (by paragraph, semantic boundary, or fixed size with overlap), embed each chunk using an embedding model, store vectors + metadata in a vector database
2. **Retrieval:** Embed the user's query, do a similarity search (cosine distance) against the vector DB, optionally rerank results with a cross-encoder
3. **Generation:** Construct a prompt with the retrieved chunks as context, send to the LLM, the model generates a response grounded in the retrieved content

Key decisions: chunk size (too small = no context, too large = noise), number of chunks to retrieve (balance context quality vs. token cost), and whether to use hybrid search (vector + keyword).

### Q: What are common chunking strategies?

**Answer:**
- **Fixed-size:** Split by token/character count with overlap. Simple, predictable, but may split mid-sentence.
- **Recursive/structural:** Split on paragraph → sentence → character boundaries. Respects document structure.
- **Semantic:** Use embeddings to find natural topic boundaries. Best quality but more complex.
- **Document-aware:** Split on headings, code blocks, or other structural markers.

I usually start with recursive splitting (paragraph boundaries, ~500-1000 tokens, 10-20% overlap) and tune from there based on retrieval quality.

### Q: How do you evaluate RAG quality?

**Answer:** Three dimensions:
1. **Retrieval quality:** Are the right chunks being retrieved? Measure recall@k, precision@k, MRR (Mean Reciprocal Rank)
2. **Generation quality:** Is the model using the context correctly? Check faithfulness (does the answer match the sources?) and relevance (does it answer the question?)
3. **End-to-end:** Does the user get a good answer? Human evaluation, automated metrics like RAGAS framework

Common failure modes: retrieving irrelevant chunks, model ignoring retrieved context, model hallucinating beyond the provided context.

---

## Agents & Tool Use

### Q: What is function calling / tool use?

**Answer:** Function calling lets the LLM request structured actions from external systems. You define available tools with schemas (name, description, parameters), the model outputs a structured tool call instead of text when appropriate, your code executes the tool and returns the result, and the model incorporates the result into its response. This extends the LLM beyond text generation — it can query databases, call APIs, perform calculations, and interact with the real world.

### Q: What's the agent loop pattern?

**Answer:** An agent loop is: (1) send the conversation to the LLM, (2) if the model requests a tool call, execute it and append the result, (3) repeat until the model returns a final text response (or a max iteration limit is hit). This lets the model plan and execute multi-step tasks autonomously — it might search a database, use those results to call an API, then summarize the outcome. Key considerations: max iterations to prevent infinite loops, error handling for failed tools, and guardrails on which tools can be called.

### Q: What is ReAct and how does it work?

**Answer:** ReAct (Reasoning + Acting) is a pattern where the model explicitly alternates between thinking and acting. Each step has: **Thought** (reason about what to do next), **Action** (call a tool), **Observation** (tool result). This chain continues until the model has enough information to produce a final answer. The key insight is that explicit reasoning steps improve tool-use accuracy — the model "plans" before acting rather than jumping to a tool call.

### Q: How do you handle errors in agent systems?

**Answer:** Multiple layers:
1. **Tool-level:** Catch exceptions, return structured error messages the model can reason about
2. **Retry logic:** Let the model try a different approach when a tool fails (e.g., reformulate a query)
3. **Iteration limits:** Cap the agent loop to prevent infinite retries
4. **Graceful degradation:** If tools fail, have the model explain what it couldn't do rather than hallucinating
5. **Guardrails:** Validate tool inputs before execution, validate outputs before passing to the model
6. **Human escalation:** For high-stakes actions, require human approval before execution

---

## Production & Architecture

### Q: How do you handle LLM costs in production?

**Answer:**
1. **Model selection:** Use the cheapest model that's good enough per task (e.g., Haiku for classification, Sonnet for generation)
2. **Caching:** Cache responses for identical/similar prompts (semantic caching)
3. **Token optimization:** Minimize prompt length, use concise system prompts, summarize conversation history
4. **Prompt chaining:** Use cheap models for filtering/routing, expensive models only when needed
5. **Batching:** Where supported, batch requests for throughput pricing
6. **Monitoring:** Track cost per request, per feature, per user

### Q: How would you evaluate an LLM application?

**Answer:** Build an eval pipeline:
1. **Test set:** Curate (input, expected output) pairs representing real-world usage
2. **Automated metrics:** Exact match (classification), BLEU/ROUGE (summarization), LLM-as-judge (open-ended)
3. **Regression testing:** Run the test set on every prompt change or model update
4. **A/B testing:** Compare approaches with real users
5. **Failure analysis:** Categorize errors to find systematic issues

The key principle: **eval-driven development**. Define your success metrics before writing prompts, then iterate.

### Q: What is streaming and why does it matter?

**Answer:** Streaming returns tokens as they're generated rather than waiting for the full response. UX-wise, it makes the application feel faster because the user sees output immediately (time-to-first-token matters more than total generation time). Technically, it uses SSE (Server-Sent Events) or WebSockets. Implementation considerations: you need to handle partial JSON (for structured output), manage backpressure, and decide how to handle tool calls mid-stream.

### Q: How do you make LLM applications safe?

**Answer:**
1. **Input safety:** Content filtering, prompt injection defense, input length limits
2. **Output safety:** Content filtering, PII detection, output validation against expected schemas
3. **Architectural safety:** Principle of least privilege for tool access, human-in-the-loop for destructive actions, audit logging
4. **Operational safety:** Rate limiting, cost caps, monitoring for unusual patterns
5. **Alignment techniques:** System prompts with clear boundaries, constitutional AI principles

---

## System Design

### Q: Design a customer support chatbot.

**Answer (framework):**
1. **Requirements:** Scope (which products/topics?), autonomy level (answer directly vs. escalate?), integrations (ticketing system, knowledge base, order DB?)
2. **Architecture:**
   - RAG over knowledge base for product/policy questions
   - Tool use for account actions (check order status, initiate refund)
   - Conversation memory (summarize or window-based)
   - Routing layer: intent classification → specialized handlers
3. **Safety:** Never expose sensitive data, require auth for account actions, escalate when confidence is low
4. **Evaluation:** Customer satisfaction, resolution rate, escalation rate, accuracy on test cases

### Q: How would you approach building an LLM feature from scratch?

**Answer (my process):**
1. **Define the task precisely** — what does good output look like? Create 20+ examples.
2. **Start simple** — zero-shot prompt with a good model. Measure baseline.
3. **Iterate on prompts** — few-shot, CoT, structured output. Measure each change.
4. **Add retrieval if needed** — RAG for knowledge-grounded tasks.
5. **Add tools if needed** — function calling for actions beyond text.
6. **Build evals** — automated test suite, run on every change.
7. **Optimize** — cheaper models where possible, caching, token reduction.
8. **Monitor in production** — cost, latency, quality metrics, failure rates.

Start simple, measure everything, add complexity only when the evals justify it.
