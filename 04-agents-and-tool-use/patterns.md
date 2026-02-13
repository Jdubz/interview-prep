# Agent Patterns

## ReAct (Reasoning + Acting)

The most well-known agent pattern. The model explicitly alternates between reasoning and acting.

### Format

```
Thought: I need to find the user's order status. I'll look up their order.
Action: get_order({ orderId: "ORD-123" })
Observation: { status: "shipped", tracking: "1Z999..." }

Thought: The order is shipped. I should get the tracking details.
Action: get_tracking({ trackingNumber: "1Z999..." })
Observation: { location: "Memphis, TN", estimatedDelivery: "2024-03-15" }

Thought: I now have all the information to answer the user.
Answer: Your order ORD-123 has shipped! It's currently in Memphis, TN
with an estimated delivery of March 15th.
```

### Why It Works

- **Explicit reasoning** before each action improves tool selection accuracy
- **Observations** ground the model in actual results (not hallucinated data)
- **Traceability** — you can log and debug each thought/action pair

### When to Use

- Complex, multi-step tasks where reasoning improves accuracy
- Tasks where you want an audit trail of the model's decision-making
- When the model needs to adapt its strategy based on intermediate results

---

## Multi-Step Orchestration

### Sequential Orchestration

Steps execute one after another, each dependent on the previous:

```
Classify intent → Route to handler → Execute action → Format response
```

**Example:** Customer support
1. Classify the customer's intent (billing, technical, account)
2. Retrieve relevant knowledge base articles
3. If action needed, call the appropriate API
4. Format and return the response

### Parallel Orchestration

Independent sub-tasks run concurrently, then results are combined:

```
User: "Compare the specs of Product A and Product B"
    ↓
┌─ get_product_specs("A") ─┐
│                           ├─→ LLM: "Here's the comparison..."
└─ get_product_specs("B") ─┘
```

Many providers support parallel tool calls — the model requests multiple tools in a single response.

### Hierarchical Orchestration

A "manager" agent delegates to specialized "worker" agents:

```
Manager Agent
├── Research Agent (has search tools)
├── Analysis Agent (has data tools)
└── Writing Agent (has formatting tools)
```

**When to use:** Complex tasks that benefit from specialized sub-agents with different tools/system prompts.

---

## Error Handling Patterns

### Pattern 1: Let the Model Recover

Return errors to the model and let it try an alternative approach:

```python
# Instead of crashing, return the error as a tool result
tool_result = {
    "error": "User not found with email john@example.com",
    "suggestion": "Try searching by name or user ID instead",
}
# The model sees this and adapts: "Let me try searching by name..."
```

### Pattern 2: Retry with Backoff

For transient failures (rate limits, timeouts):

```python
async def execute_with_retry(tool_call: ToolCall, max_retries: int = 3) -> Any:
    for i in range(max_retries):
        try:
            return await execute_tool(tool_call)
        except Exception:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(2 ** i)  # Exponential backoff
```

### Pattern 3: Graceful Degradation

When a tool is unavailable, tell the model and let it work with what it has:

```python
if not is_tool_available(tool_call.name):
    return {
        "error": f"Tool {tool_call.name} is currently unavailable.",
        "available_alternatives": get_alternative_tools(tool_call.name),
    }
```

### Pattern 4: Circuit Breaker

Prevent runaway agent loops:

```python
MAX_CONSECUTIVE_ERRORS = 3
consecutive_errors = 0

for step in agent_loop:
    result = await execute_tool(step)
    if result.get("error"):
        consecutive_errors += 1
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return "I'm having trouble completing this task. Let me summarize what I've found so far..."
    else:
        consecutive_errors = 0
```

---

## Guardrails

### Input Validation

Validate tool arguments before execution:

```python
def validate_tool_call(call: ToolCall, schema: ToolDefinition) -> ValidationResult:
    # Check required parameters
    for param in schema.parameters.get("required", []):
        if param not in call.arguments:
            return ValidationResult(valid=False, errors=[f"Missing required parameter: {param}"])

    # Check enum values
    properties = schema.parameters.get("properties", {})
    for key, value in call.arguments.items():
        param_schema = properties.get(key, {})
        if "enum" in param_schema and value not in param_schema["enum"]:
            return ValidationResult(valid=False, errors=[f"Invalid value for {key}: {value}"])

    return ValidationResult(valid=True, errors=[])
```

### Permission Levels

Different tools may require different authorization levels:

```python
PermissionLevel = Literal["read", "write", "admin"]

TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    "search_products": "read",
    "get_order_status": "read",
    "update_order": "write",
    "issue_refund": "admin",
    "delete_account": "admin",
}

def can_execute(tool: str, user_level: PermissionLevel) -> bool:
    levels: list[PermissionLevel] = ["read", "write", "admin"]
    required = levels.index(TOOL_PERMISSIONS.get(tool, "admin"))
    actual = levels.index(user_level)
    return actual >= required
```

### Human-in-the-Loop

For high-stakes actions, require confirmation:

```python
REQUIRES_CONFIRMATION = {"issue_refund", "delete_account", "send_email", "transfer_funds"}

async def maybe_confirm(tool_call: ToolCall) -> bool:
    if tool_call.name not in REQUIRES_CONFIRMATION:
        return True

    # Present the action to the user for confirmation
    confirmed = await request_user_confirmation(
        action=tool_call.name,
        details=tool_call.arguments,
        message=f"The assistant wants to {tool_call.name}. Approve?",
    )
    return confirmed
```

---

## Conversation Memory Patterns

### Full History

Pass the entire conversation to the model each time.

```
Pro: Maximum context
Con: Hits context window limits, increasing cost
```

### Sliding Window

Keep only the last N messages:

```python
def sliding_window(messages: list[Message], window_size: int) -> list[Message]:
    system_messages = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    return system_messages + non_system[-window_size:]
```

### Summarization

Periodically summarize older messages:

```
Messages 1-20 → Summary: "User asked about refund policy.
Agent found order #123 was eligible. User confirmed refund."

Messages 21+ → Keep full messages
```

### Retrieval-Based Memory

Store all messages, retrieve only relevant ones for each turn (essentially RAG over conversation history).

---

## Tool Design Best Practices

1. **Single responsibility** — each tool does one thing well
2. **Descriptive names** — `search_knowledge_base`, not `search` or `kb`
3. **Helpful descriptions** — the model reads them to decide when to use tools
4. **Structured errors** — return `{ error: "message", suggestion: "try this" }` not just strings
5. **Idempotent when possible** — safe to retry without side effects
6. **Minimal parameters** — only require what's truly needed; use sensible defaults
7. **Return useful data** — include enough context for the model to reason about results

---

## Anti-Patterns

| Anti-Pattern | Problem | Better Approach |
|---|---|---|
| Too many tools | Model gets confused, makes poor selections | Group related actions, limit to 10-20 tools |
| Vague descriptions | Model doesn't know when to use the tool | Be explicit about when and why to use each tool |
| No error handling | Agent crashes on first failure | Return errors as results, let model adapt |
| No iteration limit | Infinite loops | Cap at N iterations, gracefully exit |
| Tools with side effects + no confirmation | Destructive actions without oversight | Human-in-the-loop for writes/deletes |
| Overly broad tools | "do_anything(action: string)" | Specific tools with typed parameters |
