# Prompt Engineering Cheat Sheet

Quick reference for interviews and daily work. Print this, bookmark it, internalize it.

---

## Decision Tree: Which Prompting Approach?

```
START: What is the task?
│
├─ Simple, well-defined task (classification, extraction, translation)?
│   ├─ Model gets it right zero-shot? → ZERO-SHOT. Done.
│   ├─ Output format is inconsistent? → Add FORMAT SPECIFICATION
│   └─ Results are wrong/inconsistent? → Add FEW-SHOT EXAMPLES (3-5)
│
├─ Requires reasoning (math, logic, multi-step)?
│   ├─ Single clear reasoning path? → CHAIN-OF-THOUGHT
│   ├─ High stakes, need reliability? → CoT + SELF-CONSISTENCY (5-10 samples)
│   └─ Complex planning, many possible approaches? → TREE-OF-THOUGHT
│
├─ Complex task with multiple sub-tasks?
│   ├─ Can be decomposed into sequential steps? → PROMPT CHAINING
│   ├─ Sub-tasks are independent? → PARALLEL PROMPT EXECUTION
│   └─ Need different models/configs per step? → PROMPT CHAINING with routing
│
├─ Need guaranteed output format?
│   ├─ Using OpenAI? → json_schema with strict: true
│   ├─ Using Anthropic? → tool_choice with forced tool
│   ├─ Using open source? → Constrained decoding (Outlines/Guidance)
│   └─ Any provider? → Format in prompt + validation + retry
│
├─ Output quality not good enough?
│   ├─ Specific known failure modes? → NEGATIVE PROMPTING + edge case examples
│   ├─ General quality issues? → Add ROLE/PERSONA to system prompt
│   └─ Already optimized prompt? → Consider FINE-TUNING
│
└─ Security concern (untrusted input)?
    └─ DELIMITERS + SYSTEM PROMPT hardening + INPUT SANITIZATION + DUAL-LLM check
```

---

## Prompt Templates

### Classification (Single-Label)

```
System: Classify the input into exactly one category.
Categories:
- billing: payment, charges, invoices, refunds, subscriptions
- technical: bugs, errors, crashes, performance issues
- account: login, password, settings, profile changes
- general: questions, feedback, feature requests

Respond with ONLY the category name, nothing else.

User: {input}
```

### Classification (Multi-Label with Confidence)

```
System: Tag the input with all applicable labels and rate confidence.

Labels: security, performance, ux, accessibility, seo, mobile

Return JSON:
{
  "labels": ["label1", "label2"],
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence"
}
```

### Entity Extraction

```
System: Extract entities from the document. Return JSON:
{
  "people": [{"name": "...", "role": "..."}],
  "organizations": ["..."],
  "dates": ["YYYY-MM-DD"],
  "amounts": ["$X,XXX.XX"],
  "locations": ["..."]
}

Rules:
- Empty array if no entities found for a field
- Only extract explicitly stated information
- Standardize dates to ISO 8601
- Standardize currency to USD with two decimal places

<document>
{input}
</document>
```

### Summarization (Length-Controlled)

```
Summarize the following in exactly {n} bullet points.
Each bullet: one sentence, focus on the most important information.
No introduction or conclusion — just the bullets.

<text>
{input}
</text>
```

### RAG / Q&A with Context

```
System: Answer using ONLY the provided context.
If the answer isn't in the context, say "The provided context does not
contain enough information to answer this question."
Do not use external knowledge. Cite sources as [Source N].

<context>
{retrieved_chunks_with_source_labels}
</context>

User: {question}
```

### Code Review

```
System: Review the code for bugs, security issues, and performance problems.

For each issue:
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- Location: file:line or function name
- Problem: what is wrong
- Fix: how to fix it

If no issues found, say "No issues found."
Ignore style and formatting.

<code>
{code}
</code>
```

### Chain-of-Thought

```
{problem_statement}

Think through this step by step:
1. First, identify what we know
2. Then, determine what we need to find
3. Work through the logic
4. State your final answer

ANSWER: <your answer>
```

### Self-Consistency Wrapper

```python
# Send the same CoT prompt N times with temperature > 0
responses = await asyncio.gather(*[
    call_llm(prompt, temperature=0.7) for _ in range(N)
])
answers = [extract_answer(r) for r in responses]
final_answer = majority_vote(answers)
```

---

## Common Pitfalls and Fixes

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Vague instructions | Output format varies across calls | Specify exact format with examples |
| No output format | Responses include explanatory text you can't parse | Add "Return ONLY..." or use json_schema |
| Prompt too long | Model ignores middle instructions | Put critical instructions first and last |
| All examples same label | Model always predicts that label | Balance examples across all labels |
| Temperature too high | Different answers each time for same input | Use temperature=0 for deterministic tasks |
| Temperature too low | Outputs are repetitive and uncreative | Increase temperature for creative/diverse tasks |
| No edge case handling | Crashes on empty/malformed input | Add "If input is empty, return..." |
| CoT on simple tasks | Wastes tokens, no accuracy gain | Only use CoT for reasoning-heavy tasks |
| Contradictory instructions | Model picks one randomly | Review prompt for conflicts |
| Ignoring instruction hierarchy | Prompt injection via user input | Put security constraints in system prompt |
| No retry logic | Single failure = system failure | Retry with error feedback (2-3 attempts) |
| Prompt in wrong language | Poor results for non-English tasks | Use target language in prompt if model supports it |
| Stuffing context | Irrelevant context dilutes answers | Only include relevant retrieved chunks |
| Missing delimiters | Model confuses instructions with data | Wrap user data in `<tags>` or ``` |
| Not testing prompt changes | "Improved" prompt regresses on edge cases | Maintain eval set, test every change |

---

## Provider Comparison

| Feature | OpenAI | Anthropic | Open Source |
|---------|--------|-----------|-------------|
| **System prompt** | `role: "system"` | `system` parameter | Model-specific template |
| **JSON mode** | `response_format: json_object` | Not native (use tool_choice) | Outlines / grammar |
| **Strict schema** | `json_schema` + `strict: true` | Tool `input_schema` | Constrained decoding |
| **Function calling** | Native, parallel | Native (tool use) | Varies by model |
| **Prompt caching** | Automatic (50% discount) | Explicit markers (90% discount) | N/A (self-hosted) |
| **Log probabilities** | `logprobs: true` | Not available | Full access |
| **Stop sequences** | `stop` parameter | `stop_sequences` parameter | Full control |
| **Built-in reasoning** | o1/o3 models | Extended thinking | N/A |
| **Prefill / assistant msg** | Not supported | Supported (steer output) | Depends on framework |
| **Max context** | 128K (GPT-4o) | 200K (Claude) | Varies (8K-128K) |
| **Streaming** | SSE | SSE | Full control |
| **Batch API** | Yes (50% discount) | Yes (50% discount) | Self-managed |

---

## Delimiter Quick Reference

| Delimiter | Syntax | Best For |
|-----------|--------|----------|
| XML tags | `<context>...</context>` | Claude, structured sections, nesting |
| Triple backticks | ` ```code``` ` | Code blocks (all providers) |
| Triple quotes | `"""text"""` | Large text blocks |
| Triple dashes | `---` | Section breaks |
| Markdown headers | `### Section` | Multi-part prompts |
| Square brackets | `[Source 1]` | Source citations |
| Curly braces | `{variable}` | Template variables |

---

## Stop Sequences Quick Reference

```python
# OpenAI
client.chat.completions.create(stop=["END", "\n\n"])

# Anthropic
client.messages.create(stop_sequences=["END", "\n\n"])

# Common stop sequences:
"END"           # Explicit end marker
"\n\n"          # Double newline (end of paragraph)
"```"           # End of code block
"---"           # Section break
"ANSWER:"       # Stop before the answer (for CoT parsing)
"Human:"        # Prevent role confusion
```

---

## Temperature Guide

| Temperature | Use Case | Example |
|-------------|----------|---------|
| 0.0 | Deterministic tasks, classification, extraction | "Classify this ticket" |
| 0.1-0.3 | Mostly deterministic, slight variation | Code generation, structured summaries |
| 0.5-0.7 | Balanced creativity/consistency | General writing, explanations |
| 0.7-1.0 | Creative tasks, brainstorming | Marketing copy, story generation |
| 1.0+ | Maximum diversity (self-consistency sampling) | Multiple reasoning paths |

---

## Token Estimation Rules of Thumb

- 1 token ~ 4 characters in English
- 1 token ~ 0.75 words in English
- 100 tokens ~ 75 words
- 1 page of text ~ 300-400 tokens
- Code is typically more token-dense than prose
- Non-English text uses more tokens per word
- JSON structure adds ~20-30% overhead vs. plain text

---

## Cost Optimization Checklist

1. Start with the smallest model that meets quality requirements
2. Use prompt caching (static context first, dynamic input last)
3. Minimize few-shot examples (use only as many as needed)
4. Strip CoT reasoning from output if only the answer is needed
5. Use batch APIs for non-real-time workloads (50% discount)
6. Set appropriate max_tokens (don't over-allocate)
7. Cache responses for identical inputs (application-level)
8. Use streaming to improve perceived latency without changing cost
9. Consider fine-tuning to replace long prompts with shorter ones
10. Monitor token usage and set alerts for anomalies

---

## Key Numbers to Know

| Metric | Value |
|--------|-------|
| Few-shot examples sweet spot | 3-5 |
| Self-consistency samples | 5-10 |
| Max retries for structured output | 2-3 |
| Prompt caching min prefix (Anthropic) | 1024 tokens |
| Prompt caching min prefix (OpenAI) | 1024 tokens |
| CoT accuracy improvement (typical) | 10-30% on reasoning tasks |
| Self-consistency improvement over single CoT | 5-15% |
| Constitutional AI revision iterations | 2-3 max |
| A/B test initial traffic split | 5-10% |

---

*Back to [Core Knowledge](README.md) | See also: [Deep Dive](deep-dive.md)*
