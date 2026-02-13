# Common Prompt Patterns

Reusable patterns for tasks that come up frequently. Each shows the prompt structure and key considerations.

---

## Classification

Assign input to one or more categories.

### Single-Label Classification

```
System: Classify the customer message into exactly one category:
- billing: payment, charges, invoices, subscriptions
- technical: bugs, errors, performance, features not working
- account: login, password, profile, settings
- general: everything else

Respond with ONLY the category name.

User: I keep getting a 500 error when I try to export my data
Assistant: technical
```

**Key considerations:**
- Define categories with examples/descriptions to reduce ambiguity
- Use temperature=0 for consistency
- For edge cases, few-shot examples help more than longer descriptions

### Multi-Label Classification

```
System: Tag the following text with ALL applicable labels.
Labels: security, performance, ux, accessibility, seo, mobile

Return as a JSON array of strings. Only include labels that clearly apply.

User: The page takes 8 seconds to load on mobile and images have no alt text
Assistant: ["performance", "mobile", "accessibility"]
```

### Classification with Confidence

```
System: Classify this support ticket and rate your confidence.

Return JSON:
{
  "category": "billing" | "technical" | "account" | "general",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence explanation"
}

If confidence is "low", also suggest the next most likely category.
```

**Why add confidence:** Lets you route low-confidence classifications to a human reviewer.

---

## Data Extraction

Pull structured data from unstructured text.

### Entity Extraction

```
System: Extract all entities from the text. Return JSON:
{
  "people": [{ "name": "...", "role": "..." }],
  "companies": ["..."],
  "dates": ["... (ISO 8601)"],
  "monetary_values": ["... (standardized)"],
  "locations": ["..."]
}

If a field has no entities, use an empty array.
Only extract what's explicitly stated — do not infer.

User: <document>
Sarah Chen, CTO of Acme Corp, announced a $50M Series C on March 15, 2024
at their San Francisco headquarters.
</document>
```

**Key considerations:**
- Use delimiters (`<document>`) to separate instructions from data
- Specify date/currency formats explicitly
- "Do not infer" prevents the model from making assumptions

### Form/Field Extraction

```
System: Extract the following fields from the email. Return JSON with these exact keys:
- sender_name: string
- sender_email: string
- subject: string
- action_items: string[] (list of specific tasks requested)
- urgency: "high" | "medium" | "low"
- deadline: string | null (ISO 8601 if mentioned, null otherwise)

If a field cannot be determined, use null (or empty array for arrays).
```

### Relation Extraction

```
System: Extract relationships between entities in the text.
Return as a JSON array of triples:
[
  { "subject": "...", "predicate": "...", "object": "..." }
]

Example: "Apple acquired Beats" → { "subject": "Apple", "predicate": "acquired", "object": "Beats" }
```

---

## Summarization

Condense text while preserving key information.

### Length-Controlled Summary

```
System: Summarize the following text in exactly 3 bullet points.
Each bullet should be one sentence, focusing on the most important information.
Do not include any introduction or conclusion — just the bullet points.

User: <text>
{long_text}
</text>
```

### Audience-Targeted Summary

```
System: Summarize this technical document for two audiences.

EXECUTIVE SUMMARY (2-3 sentences):
- Focus on business impact, costs, and timeline
- No technical jargon

TECHNICAL SUMMARY (3-5 bullet points):
- Focus on architecture decisions, trade-offs, and implementation details
- Assume the reader is a senior engineer
```

### Incremental Summarization

For very long documents, summarize in chunks and then summarize the summaries:

```
Step 1: Summarize pages 1-10 → summary_1
Step 2: Summarize pages 11-20 → summary_2
...
Final: Summarize [summary_1, summary_2, ...] → final_summary
```

---

## Code Generation

### Generate with Constraints

```
System: You are a senior Python engineer. Generate code that:
- Uses modern Python (type hints, no Any)
- Handles errors explicitly (no silent failures)
- Includes JSDoc comments for public functions
- Uses async/await (no raw promises)

Do not generate tests unless asked. Do not wrap in a class unless the problem requires state.

User: Write a function that fetches paginated data from a REST API
and returns all results. The API uses cursor-based pagination with
a `next_cursor` field in the response.
```

### Code Review

```
System: Review this code for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues

For each issue found:
- Cite the specific line(s)
- Explain the problem
- Provide a fix

If no issues are found in a category, say "None found."
Ignore style/formatting.
```

### Code Transformation

```
System: Convert the following JavaScript code to Python.
- Add proper type annotations (no `Any` types)
- Use dataclasses for object shapes
- Use Literal types or Enums where string unions are appropriate
- Preserve all functionality exactly

Return only the Python code, no explanation.
```

---

## Question Answering

### Closed-Book (From Training Data)

```
System: Answer the following question concisely and accurately.
If you're not confident in the answer, say "I'm not certain, but..."
Do not make up facts.
```

### Open-Book (RAG Context)

```
System: Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information to answer fully,
say what you can and clearly state what's missing.

Do not use any knowledge beyond what's in the context.
Cite sources using [Source N] notation.

<context>
[Source 1: API Documentation]
{chunk_1}

[Source 2: FAQ]
{chunk_2}
</context>
```

### Comparative Q&A

```
System: Compare the items below based on the user's criteria.
Structure your response as a table with clear pros/cons.
Be objective — state facts, not opinions.
End with a brief recommendation based on the user's stated needs.
```

---

## Transformation

### Translation with Context

```
System: Translate the following text to {target_language}.
- Maintain the original tone (formal/informal)
- Preserve formatting (bullet points, headers)
- For technical terms, include the English original in parentheses
- If a phrase doesn't translate directly, explain the adaptation
```

### Format Conversion

```
System: Convert the following markdown table to a JSON array of objects.
Use the header row as keys (camelCase).
Preserve data types: numbers as numbers, dates as ISO 8601 strings,
booleans as booleans, everything else as strings.
```

### Style Transfer

```
System: Rewrite the following text to match the target style.
Preserve all factual content — only change the tone and language level.

Target style: Professional but approachable. Short sentences.
Reading level: 8th grade. No jargon.

User: <original>
The implementation leverages a microservices architecture predicated on
event-driven communication paradigms, facilitating loose coupling and
enhanced horizontal scalability characteristics.
</original>
```

---

## Pattern Composition

Real-world prompts often combine patterns. A customer support system might combine:

1. **Classification** (route the query)
2. **Extraction** (pull out order IDs, product names)
3. **Q&A with RAG** (answer from knowledge base)
4. **Summarization** (condense the response)

The key is **prompt chaining** — each pattern as a discrete step in a pipeline, with clear inputs and outputs.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Vague instructions | Inconsistent outputs | Be specific about format, length, content |
| No output format | Unparseable responses | Specify JSON, markdown, or exact format |
| Asking for too much | Degraded quality on each sub-task | Break into smaller, focused prompts |
| Contradictory instructions | Model picks one randomly | Review and resolve conflicts |
| No examples for ambiguous tasks | Model guesses wrong | Add 2-3 few-shot examples |
| Prompt too long | Important instructions get lost | Put critical instructions first and last |
