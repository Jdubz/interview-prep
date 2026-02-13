# Prompt Engineering Techniques

## Why Prompt Engineering Matters

The same model can give wildly different results depending on how you prompt it. Prompt engineering is the most accessible, lowest-cost way to improve LLM output quality — no fine-tuning, no infrastructure changes. In an interview, demonstrating prompt engineering skill shows you can get real value from LLMs in production.

---

## Zero-Shot Prompting

Ask the model to perform a task with no examples — just instructions.

```
Classify the following customer message as "billing", "technical", or "general".

Message: "I can't log into my account after the latest update"

Category:
```

**When it works well:**
- Tasks the model has seen extensively in training (summarization, classification, translation)
- When the task is clearly defined and unambiguous

**When it struggles:**
- Novel or ambiguous task definitions
- When you need a very specific output format
- Domain-specific tasks with non-obvious conventions

---

## Few-Shot Prompting

Provide examples of (input, output) pairs before the actual task. The model learns the pattern from examples.

```
Classify customer messages.

Message: "Why was I charged twice?"
Category: billing

Message: "The app crashes when I open settings"
Category: technical

Message: "Do you have a mobile app?"
Category: general

Message: "I can't log into my account after the latest update"
Category:
```

**Best practices:**
- Use 2–5 examples (more isn't always better — it costs tokens)
- Examples should be diverse and representative
- Order can matter — place the most representative examples first
- Match the exact format you want in the output

**Pitfall:** If all your examples have the same label, the model may be biased toward that label. Balance your examples.

---

## Chain-of-Thought (CoT)

Ask the model to reason step-by-step before giving a final answer. This dramatically improves performance on tasks requiring logic, math, or multi-step reasoning.

### Explicit CoT

```
Q: A store has 15 apples. They sell 6 in the morning and receive a shipment
of 20 in the afternoon. Then they sell 8 more. How many apples do they have?

Think step by step:

1. Start: 15 apples
2. Sell 6: 15 - 6 = 9
3. Receive 20: 9 + 20 = 29
4. Sell 8: 29 - 8 = 21

Answer: 21
```

### Zero-Shot CoT

Simply append "Let's think step by step" or "Think through this carefully":

```
How many r's are in the word "strawberry"? Let's think step by step.
```

### When to use CoT:
- Math and logic problems
- Multi-step reasoning
- Tasks where the model needs to "show its work"
- Complex classifications where the reasoning matters

### When NOT to use CoT:
- Simple factual lookups
- Tasks where speed/cost matters more than accuracy
- When you need short, direct answers (CoT increases output tokens)

---

## System Prompts

The system prompt sets the model's persona, constraints, and behavioral rules. It's processed before the user message and typically given elevated priority by the model.

```
System: You are a senior code reviewer. You:
- Focus on bugs, security issues, and performance problems
- Ignore style/formatting unless it affects readability
- Rate severity as: critical, warning, or info
- Always suggest a fix for each issue found
- Respond in JSON format
```

**Best practices:**
- Be specific about what the model should and shouldn't do
- Define the output format
- Set the persona/role if it helps constrain behavior
- Place your most important instructions here (models weight system prompts heavily)

**Provider differences:**
- OpenAI: explicit `role: "system"` message
- Anthropic: separate `system` parameter
- Open source: varies; often uses special tokens like `<<SYS>>` or `<|system|>`

---

## Output Structuring

Constraining the model's output format is critical for programmatic consumption.

### JSON Output

```
Extract the entities from this text and return them as JSON.

Text: "John Smith from Acme Corp called about the Q4 report on January 15th."

Return a JSON object with keys: "person", "organization", "document", "date".
Only return the JSON, no other text.
```

**Stronger enforcement:**
- Many providers support `response_format: { type: "json_object" }` or JSON schema constraints
- For critical applications, always validate the output programmatically

### Structured Formats

```
Analyze this code and respond in exactly this format:

SUMMARY: <one sentence>
ISSUES:
- [severity] description
RECOMMENDATION: <one sentence>
```

**Tips:**
- Show the exact format in the prompt (the model mirrors what it sees)
- Use delimiters (`---`, `###`, XML tags) to separate sections
- For code output, specify the language and any conventions

---

## Role Prompting

Assign the model a specific role or expertise to improve response quality and relevance.

```
You are an experienced database architect with 15 years of experience
in PostgreSQL performance optimization. A junior developer asks you:

"Our query is slow on a table with 10M rows. It joins 3 tables and
filters by date range. What should I check?"
```

**Why it works:**
- Activates relevant patterns from training data
- Constrains the response space (a "DBA" won't suggest frontend solutions)
- Often produces more expert-level, specific answers

**Common roles:**
- Domain expert (security researcher, data scientist, etc.)
- Persona (helpful assistant, strict reviewer, devil's advocate)
- Audience-aware ("explain to a 5-year-old", "explain to a senior engineer")

---

## Delimiters and Structure

Use clear delimiters to separate instructions from data. This prevents prompt injection and helps the model parse complex prompts.

```
Summarize the following document.

<document>
{user_provided_text}
</document>

Provide a 3-sentence summary.
```

**Common delimiter patterns:**
- XML tags: `<context>...</context>` — works very well, especially with Claude
- Triple backticks: ` ``` ` — good for code
- Markdown headers: `### Instructions`, `### Input`
- Special tokens: `---`, `===`

**Why delimiters matter:**
- Prevent the model from confusing instructions with data
- Critical defense against prompt injection
- Make complex prompts readable and maintainable

---

## Prompt Chaining

Break complex tasks into a sequence of simpler prompts, where each step's output feeds into the next.

```
Step 1: Extract key facts from the document → list of facts
Step 2: For each fact, assess if it's supported by evidence → annotated facts
Step 3: Generate a summary using only well-supported facts → final summary
```

**Benefits:**
- Each step is simpler and more reliable
- Easier to debug (inspect intermediate outputs)
- Can use different models/temperatures per step
- Natural fit for Python pipelines

**Drawback:** More API calls = more latency and cost.

---

## Negative Prompting

Tell the model what NOT to do. Often more effective than only saying what to do.

```
Explain quantum computing to a software engineer.

Do NOT:
- Use analogies involving cats (Schrödinger's cat is overused)
- Assume physics background
- Include mathematical notation
- Exceed 200 words
```

---

## Iterative Refinement

Use the model to critique and improve its own output.

```
[First call]
Write a Python function that validates email addresses.

[Second call]
Review this function for edge cases and security issues:
{previous_output}

List any problems and provide a corrected version.
```

This is the foundation of more advanced patterns like self-consistency and constitutional AI.

---

## Quick Reference: Choosing Techniques

| Task | Recommended Techniques |
|---|---|
| Classification | Few-shot + structured output |
| Data extraction | System prompt + delimiters + JSON output |
| Summarization | System prompt + length constraints |
| Code generation | Role prompting + CoT + examples |
| Complex reasoning | CoT + prompt chaining |
| Creative writing | Role + temperature adjustment |
| Reliable pipelines | Structured output + chaining + validation |
