# Prompt Engineering Deep Dive

Advanced techniques and production considerations beyond the fundamentals. This material targets interview questions that probe depth — "How would you handle X at scale?" and "What are the trade-offs of approach Y?"

---

## Self-Consistency

Self-consistency (Wang et al., 2023) improves CoT accuracy by sampling multiple reasoning paths and taking the majority vote.

### How It Works

1. Prompt the model with the same CoT prompt multiple times (temperature > 0)
2. Each sample produces a different reasoning path and potentially different answer
3. Extract the final answer from each sample
4. Take the majority vote

```
Prompt (sent N times with temperature=0.7):
  "A store has 15 apples. They sell 6, receive 20, sell 8. How many remain?
   Let's think step by step."

Sample 1: 15 - 6 = 9, 9 + 20 = 29, 29 - 8 = 21. Answer: 21
Sample 2: 15 - 6 = 9, 9 + 20 = 29, 29 - 8 = 21. Answer: 21
Sample 3: 15 - 6 = 9, 9 + 20 = 29, 29 - 8 = 22. Answer: 22 (arithmetic error)
Sample 4: 15 - 6 = 9, 9 + 20 = 29, 29 - 8 = 21. Answer: 21
Sample 5: 15 - 6 = 9, 9 + 20 = 29, 29 - 8 = 21. Answer: 21

Majority vote: 21 (4 out of 5)
```

### Why It Works

- Different reasoning paths may make different errors, but correct reasoning is more common
- Errors are typically random (different arithmetic mistakes), while correct reasoning converges
- Analogous to ensemble methods in ML — multiple weak signals produce a strong signal

### When to Use Self-Consistency

| Good Fit | Poor Fit |
|----------|----------|
| Math/logic with verifiable answers | Open-ended generation (no "correct" answer) |
| Classification with clear categories | Tasks where diversity of output is desired |
| Factual questions with one right answer | Low-latency requirements |
| High-stakes decisions worth the extra cost | Simple tasks where single-shot is sufficient |

### Production Considerations

- **Cost:** N samples = N times the API cost. Typical N: 5-10.
- **Latency:** Send samples in parallel to avoid N times the latency.
- **Answer extraction:** You need reliable parsing of the final answer from each CoT trace. Use structured output or a clear answer format.
- **Weighted voting:** Instead of simple majority, weight by model confidence or reasoning length. Some implementations use the model's log probabilities.

---

## Tree-of-Thought (ToT)

Tree-of-Thought (Yao et al., 2023) extends CoT from a single linear chain to a tree of reasoning branches. The model explores multiple approaches, evaluates them, and prunes unpromising paths.

### Core Idea

```
                    Problem
                   /       \
              Approach A    Approach B
              /      \         |
          Step A1   Step A2   Step B1
            |         X        /    \
          Step A1a         Step B1a  Step B1b
            |                  |       X
          Solution         Solution
```

X = pruned (evaluated as unpromising)

### Implementation Pattern

**Step 1: Generate candidate approaches**
```
Given this problem, propose 3 distinct approaches to solve it.
For each approach, describe the first step and briefly explain why
this approach might work.

Problem: [problem description]
```

**Step 2: Evaluate and select**
```
Evaluate each of these approaches for the given problem.
Rate each on a scale of 1-5 for:
- Likelihood of reaching the correct answer
- Efficiency (fewer steps is better)

Approaches:
1. [approach 1]
2. [approach 2]
3. [approach 3]

Select the most promising approach(es) to continue.
```

**Step 3: Expand the best branches**
```
Continue developing approach [N]. Given the current progress:
[current state]

What are the possible next steps? List 2-3 options.
```

**Step 4: Repeat evaluation and expansion until solution**

### BFS vs DFS Exploration

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| BFS (breadth-first) | Expand all nodes at depth D before moving to D+1 | When you want to explore many approaches before committing |
| DFS (depth-first) | Follow one branch deeply before backtracking | When solutions require many steps and branching factor is small |
| Best-first | Always expand the highest-rated node | When evaluation is reliable and you want efficiency |

### Practical Considerations

ToT is research-grade — rarely used directly in production due to:
- High token cost (many LLM calls per problem)
- Complex orchestration logic
- Most production tasks don't benefit over simpler CoT + self-consistency

**Interview angle:** Know the concept and when it theoretically helps. In practice, recommend it for complex planning/reasoning tasks where the cost is justified (e.g., automated code architecture decisions, complex data pipeline design).

---

## Meta-Prompting

Prompts that generate or optimize other prompts. This is the "compiler" level of prompt engineering.

### Prompt Generation

```
System: You are a prompt engineer. Given a task description, generate
an effective prompt that will produce reliable results from an LLM.

The generated prompt should:
- Be specific and unambiguous
- Include output format specification
- Handle edge cases explicitly
- Use few-shot examples if the task is ambiguous

Task: I need to extract product features from customer reviews.
Features should be categorized as positive, negative, or neutral.
Input will be review text. Output needs to be structured JSON.

Generate the prompt:
```

### Automatic Prompt Optimization (APO)

A systematic approach to improving prompts:

```
Step 1: Run baseline prompt on evaluation set
Step 2: Identify failure cases
Step 3: Ask the model to analyze failures and suggest improvements

Prompt for Step 3:
  You are a prompt optimization expert. Here is a prompt being used for
  [task description]:

  <current_prompt>
  {prompt}
  </current_prompt>

  It produced these incorrect results:

  <failures>
  Input: {input_1} → Expected: {expected_1} → Got: {actual_1}
  Input: {input_2} → Expected: {expected_2} → Got: {actual_2}
  </failures>

  Analyze why the prompt fails on these cases and suggest a revised
  prompt that would handle them correctly while maintaining performance
  on other inputs.

Step 4: Evaluate revised prompt on full evaluation set
Step 5: Repeat until convergence or budget exhausted
```

### DSPy-Style Optimization

DSPy (Declarative Self-improving Language Programs) takes this further — it treats prompt optimization as a machine learning problem:

1. Define the task as a module with typed inputs/outputs
2. Provide a small training set
3. The optimizer automatically searches for the best prompt/few-shot examples
4. Compiles to an optimized prompt or fine-tuned model

**Interview point:** DSPy represents the future direction — prompts as optimizable programs rather than hand-crafted strings. Know the concept even if you haven't used the library.

---

## Constitutional AI Prompting

Self-critique and revision patterns inspired by Anthropic's Constitutional AI paper.

### The Pattern

```
Step 1 (Generate): Produce initial response
Step 2 (Critique): Review the response against principles
Step 3 (Revise): Fix identified issues
```

### Implementation

**Step 1 — Generate:**
```
[Your normal task prompt]
```

**Step 2 — Critique:**
```
Review your response against these principles:
1. Is the information factually accurate?
2. Does it contain any harmful or biased content?
3. Is it complete — does it address all parts of the question?
4. Is the tone appropriate for the audience?

For each principle, rate compliance (PASS/FAIL) and explain any issues.
```

**Step 3 — Revise:**
```
Based on your critique, provide a revised response that addresses
all identified issues. If no issues were found, return the original
response unchanged.
```

### Production Application

This pattern is useful for:
- Content moderation: generate → check for policy violations → revise
- Factual accuracy: generate → check claims → add caveats or corrections
- Compliance: generate → check against regulatory requirements → revise
- Quality assurance: generate → check against style guide → revise

The critique step can use a different (often cheaper) model than the generation step.

### Self-Refinement Loop

For iterative improvement, run critique-revise multiple times:

```python
response = generate(prompt)
for i in range(max_iterations):
    critique = evaluate(response, principles)
    if critique.all_pass:
        break
    response = revise(response, critique)
```

Diminishing returns after 2-3 iterations in practice.

---

## Prompt Caching

Modern providers cache prompt prefixes to reduce cost and latency for repeated calls with shared context.

### How It Works

When you send a prompt, the provider computes the internal representation (KV cache) for each token. If subsequent requests share the same prefix, that computation can be reused.

```
Request 1: [System prompt + 10K context + Question A]
Request 2: [System prompt + 10K context + Question B]
                    ↑ cached prefix ↑        ↑ new ↑
```

The shared prefix (system prompt + context) is computed once. Only the differing suffix requires new computation.

### Provider-Specific Behavior

**Anthropic (Prompt Caching):**
- Explicit cache control via `cache_control` markers in the message
- Mark breakpoints where caching should apply
- Cached tokens cost ~90% less on cache hits
- Minimum cacheable prefix: 1024 tokens (Claude Sonnet), 2048 tokens (Claude Haiku)
- Cache TTL: 5 minutes (extended on each hit)

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant with expertise in...",
            "cache_control": {"type": "ephemeral"}  # Cache this prefix
        }
    ],
    messages=[...]
)
```

**OpenAI (Automatic Caching):**
- Automatic — no explicit markers needed
- Caches the longest matching prefix
- 50% discount on cached input tokens
- Minimum prefix: 1024 tokens
- Available on GPT-4o and later models

### Designing Cache-Friendly Prompts

Structure prompts so the static parts come first:

```
BAD (cache-unfriendly):
  "Analyze {user_input} using these rules: [10K tokens of rules]"
  ↑ changes every request — nothing is cacheable

GOOD (cache-friendly):
  "[10K tokens of rules] Now analyze the following: {user_input}"
  ↑ static prefix (cached) ↑                       ↑ varies ↑
```

**Design principles:**
1. System prompt and static context first
2. Few-shot examples second (same examples across requests)
3. Dynamic input last
4. Don't include timestamps or request IDs early in the prompt
5. Batch similar requests together to maximize cache hits

### Cost Impact

For a RAG system sending 10K context tokens with each query:

| Scenario | Input Cost (relative) |
|----------|----------------------|
| No caching | 1.0x |
| Anthropic cached hit | ~0.1x for cached portion |
| OpenAI cached hit | ~0.5x for cached portion |

At scale, this is significant. A system making 100K requests/day with 10K-token prompts saves substantially.

---

## Prompt Optimization Workflow

A systematic, reproducible approach to improving prompts.

### Step 1: Define Evaluation Criteria

Before writing a single prompt, define what "good" means:

```
Task: Classify support tickets
Metrics:
  - Accuracy (% correct classifications)
  - Consistency (same input → same output across runs)
  - Latency (time to response)
  - Cost (tokens consumed)
  - Edge case handling (performance on ambiguous inputs)
```

### Step 2: Build an Evaluation Set

Minimum viable eval set: 50-100 labeled examples covering:
- Happy path (60%)
- Edge cases (20%)
- Known failure modes (20%)

```python
eval_set = [
    {"input": "I was charged twice for my subscription",
     "expected": "billing", "difficulty": "easy"},
    {"input": "The button doesn't work on my phone but works on desktop",
     "expected": "technical", "difficulty": "medium"},
    {"input": "I want to cancel and get a refund for the last charge",
     "expected": "billing",  # or "account"? Edge case.
     "difficulty": "hard", "notes": "ambiguous: billing + account"},
]
```

### Step 3: Baseline

Run the simplest possible prompt and measure:

```
Baseline prompt: "Classify this support ticket: {input}"
Baseline accuracy: 72%
Baseline issues: Inconsistent categories, no handling of ambiguous cases
```

### Step 4: Iterate

Apply techniques one at a time and measure impact:

| Version | Change | Accuracy | Notes |
|---------|--------|----------|-------|
| v0 | Baseline | 72% | No category definitions |
| v1 | Added category definitions | 81% | +9% from clarity |
| v2 | Added 3 few-shot examples | 87% | +6% from examples |
| v3 | Added CoT for edge cases | 89% | +2%, but 3x tokens |
| v4 | Removed CoT, added 2 edge-case examples | 90% | Better accuracy, lower cost than v3 |

### Step 5: Regression Testing

When you change a prompt, run the full eval set. Improvements on one category often cause regressions in another.

### Step 6: A/B Testing in Production

Deploy the new prompt to a fraction of traffic, compare metrics against the current prompt, and roll forward only if the new prompt wins across all key metrics.

---

## Multi-Turn Prompt Design

Designing effective conversational experiences with LLMs.

### Conversation Structure

```
System: [Persistent instructions, persona, constraints]
User: [First message]
Assistant: [First response]
User: [Follow-up, possibly referencing previous response]
Assistant: [Response with full conversation context]
...
```

The model sees the entire conversation history on every turn. This means:
- Every previous message consumes input tokens
- Earlier messages may get "lost" as context grows
- Contradictions between turns confuse the model

### Context Management Strategies

**Strategy 1: Sliding Window**
Keep only the last N turns. Simple, but loses early context.

```python
def sliding_window(messages: list, max_turns: int = 10) -> list:
    system = messages[0]  # Always keep system prompt
    recent = messages[-(max_turns * 2):]  # Keep last N turns (user + assistant)
    return [system] + recent
```

**Strategy 2: Summarize and Compress**
Periodically summarize the conversation and replace old messages with the summary.

```
System: [original system prompt]

[Summary of conversation so far:
User asked about X. You recommended Y because Z.
User then asked about A, and you clarified B.]

User: [latest message]
```

**Strategy 3: Key-Value Memory**
Extract and maintain structured state from the conversation:

```python
conversation_state = {
    "user_name": "Alex",
    "issue": "billing dispute",
    "order_id": "ORD-12345",
    "status": "investigating",
    "resolution_attempted": ["refund_offered"],
}
```

Inject this state into each prompt rather than replaying the full conversation.

### Multi-Turn Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Unbounded history | Token costs explode, early context lost | Implement context management strategy |
| No conversation state | Model forgets key facts | Extract and inject structured state |
| System prompt drift | Instructions weakened by long conversations | Reinforce critical instructions periodically |
| Ambiguous references | "Do that thing from earlier" | Ask for clarification or maintain explicit state |

---

## Provider-Specific Techniques

What works differently across major providers.

### OpenAI (GPT-4o, o1, o3)

**Strengths:**
- Best structured output support (json_schema with constrained decoding)
- Strong function/tool calling
- Good at following complex multi-part instructions
- o1/o3 models have built-in extended reasoning (no need for explicit CoT)

**Specific techniques:**
- Use `response_format: { type: "json_schema", json_schema: {...} }` for guaranteed valid output
- Use `logprobs: true` to get token-level confidence (useful for classification)
- For o1/o3 reasoning models: don't use CoT prompting (they do it internally). Provide the problem clearly and let the model reason.
- `prediction` parameter for anticipated output (faster for code editing tasks)

### Anthropic (Claude Sonnet, Opus, Haiku)

**Strengths:**
- Excellent instruction following, especially from system prompts
- Strong with XML-structured prompts
- Very good at refusing harmful requests (fewer guardrail workarounds needed)
- Explicit prompt caching control
- Extended thinking for complex reasoning tasks

**Specific techniques:**
- Use XML tags liberally — Claude responds exceptionally well to `<tags>`
- Place the most important instructions at the start and end of the system prompt
- Use `tool_choice: { type: "tool", name: "..." }` for structured output via tools
- Leverage prompt caching with `cache_control` for repeated context
- For complex reasoning: enable extended thinking instead of manual CoT
- Prefill the assistant message to steer output format:

```python
messages = [
    {"role": "user", "content": "Extract entities from: ..."},
    {"role": "assistant", "content": "{"}  # Forces JSON output starting with {
]
```

### Open Source (Llama, Mistral, etc.)

**Strengths:**
- Full control over inference (temperature, sampling, stopping)
- Can use constrained decoding libraries (Outlines, Guidance)
- No API costs (but infrastructure costs)
- Fine-tuning is straightforward

**Specific techniques:**
- Use the exact chat template the model was trained with (Jinja templates in tokenizer config)
- Constrained decoding with Outlines for guaranteed format compliance
- Shorter, more direct prompts often work better than elaborate ones
- System prompts may need to be in the model-specific format
- Smaller models need more few-shot examples and clearer instructions
- Grammar-based sampling for structured output (llama.cpp grammar feature)

### Cross-Provider Compatibility Tips

If your system needs to work across providers:
1. Test prompts on each target provider — they respond differently
2. Avoid provider-specific features in the prompt text itself
3. Abstract the API layer: same prompt, different API calls
4. Maintain per-provider prompt variants for critical paths
5. Use a prompt template system that supports provider-specific overrides

---

## Prompt Versioning and Management

Treating prompts as a first-class software artifact.

### Prompts as Code

Prompts should be:
- Version controlled (git)
- Code reviewed
- Tested against evaluation sets
- Deployed with rollback capability

```
prompts/
  classification/
    v1.txt          # Original prompt
    v2.txt          # Added few-shot examples
    v3.txt          # Optimized for edge cases
    eval_set.json   # Test cases
    results/
      v1_results.json
      v2_results.json
      v3_results.json
```

### Prompt Template System

Separate the template from the variables:

```python
# Template (version controlled, reviewed)
CLASSIFICATION_TEMPLATE = """
Classify the support ticket into one of these categories:
{categories}

{few_shot_examples}

Ticket: {input}
Category:
"""

# Variables (may come from config, database, or runtime)
categories = load_categories()      # Can change without prompt code change
examples = select_examples(input)   # Dynamic example selection
```

### A/B Testing Prompts

```python
def get_prompt_version(user_id: str, experiment: str) -> str:
    """Route users to prompt variants for A/B testing."""
    bucket = hash(user_id + experiment) % 100
    if bucket < 10:  # 10% get new prompt
        return load_prompt("classification_v4")
    return load_prompt("classification_v3")  # 90% get current prompt
```

Track metrics per variant:
- Accuracy (if you have labels or human feedback)
- User satisfaction (thumbs up/down)
- Task completion rate
- Latency and cost

### Gradual Rollout

```
Day 1: Deploy v4 to 5% of traffic
Day 2: Monitor metrics, compare to v3 baseline
Day 3: If metrics are good, increase to 25%
Day 5: If still good, increase to 50%
Day 7: Full rollout or rollback
```

### Prompt Registries

For production systems, maintain a central registry:

```python
class PromptRegistry:
    """Central store for prompt templates with versioning and A/B support."""

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        """Fetch a prompt template by name and version."""
        ...

    def register(self, name: str, template: str, metadata: dict) -> str:
        """Register a new prompt version. Returns version ID."""
        ...

    def evaluate(self, name: str, version: str, eval_set: list) -> dict:
        """Run evaluation set against a prompt version."""
        ...
```

Platforms like LangSmith, Braintrust, and PromptLayer provide managed versions of this.

---

## Advanced Patterns

### Retry with Feedback

When structured output fails validation, feed the error back:

```
[Original prompt produces invalid JSON]

Your previous response was not valid JSON. The error was:
{error_message}

Your response was:
{previous_response}

Please fix the issue and return valid JSON matching the required schema.
```

Typically resolves on the first retry. Cap at 2-3 retries to avoid infinite loops.

### Prompt Ensembles

Different prompts for the same task, combined for robustness:

```python
prompts = [
    "Classify this as positive/negative/neutral: {text}",
    "What is the sentiment of this text? (positive/negative/neutral): {text}",
    "Rate the sentiment: {text}\nOptions: positive, negative, neutral",
]

results = [await classify(p.format(text=input)) for p in prompts]
final = majority_vote(results)
```

Different phrasings may activate different model behaviors. Ensembling smooths out phrasing-dependent errors.

### Least-to-Most Prompting

Decompose a complex problem into sub-problems, solve from simplest to most complex:

```
Step 1: "What are the sub-problems needed to solve this?"
  → [sub-problem A, sub-problem B, sub-problem C]

Step 2: "Solve sub-problem A: ..."
  → solution A

Step 3: "Given solution A, solve sub-problem B: ..."
  → solution B

Step 4: "Given solutions A and B, solve sub-problem C: ..."
  → solution C (final answer)
```

Each step builds on previous solutions. Effective for math word problems and compositional tasks.

### Directional Stimulus Prompting

Provide a hint that guides reasoning without giving the answer:

```
Standard: "What causes memory leaks in Python?"

With directional stimulus:
"What causes memory leaks in Python?
Hint: Think about circular references and the garbage collector."
```

The hint narrows the search space without biasing toward a specific answer.

---

## Interview Deep-Dive Questions

Questions interviewers ask to probe depth, with guidance on strong answers:

**Q: "How would you ensure an LLM always returns valid JSON?"**
Strong answer: Layer multiple strategies — (1) prompt engineering with schema in the prompt, (2) provider features like OpenAI's json_schema mode, (3) output validation with retry on failure, (4) fallback to a simpler prompt or format.

**Q: "Your prompt works 95% of the time. How do you get to 99%?"**
Strong answer: Build an eval set from the 5% failures, analyze patterns, iterate on the prompt, consider self-consistency for critical paths, add retry logic with error feedback, consider fine-tuning if prompt engineering plateaus.

**Q: "How do you handle prompt injection in a user-facing product?"**
Strong answer: Defense in depth — input sanitization, delimiters, instruction hierarchy in system prompt, dual-LLM detection, output validation. Acknowledge it's fundamentally an unsolved problem and discuss monitoring for new attack patterns.

**Q: "How do you decide between prompt engineering and fine-tuning?"**
Strong answer: Start with prompt engineering — it's faster, cheaper, and more flexible. Fine-tune when: (1) prompt engineering plateaus below your accuracy target, (2) you need to reduce latency (fine-tuned models need shorter prompts), (3) you have a large labeled dataset, (4) the task is well-defined and stable (fine-tuning is expensive to iterate).

**Q: "How would you design a prompt system for a multi-language customer support bot?"**
Strong answer: Discuss language detection as a first step, prompt templates per language vs. single prompt with language instructions, few-shot examples per language, testing coverage across languages, and the reality that model quality varies by language.

---

*Back to [Core Knowledge](README.md) | Next: [Cheat Sheet](cheat-sheet.md)*
