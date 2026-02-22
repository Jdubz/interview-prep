# Module 04: Agents & Tool Use — Core Interview Knowledge

## Overview

Agents are LLMs in a loop: they receive input, reason about it, invoke tools, observe results, and repeat until the task is complete. This module covers everything you need to discuss agent architectures, tool integration, and agentic patterns in an interview setting.

**Key insight for interviews:** The model never executes anything. It outputs structured requests. Your code does the execution. The entire agent paradigm rests on this separation.

---

## 1. Function Calling / Tool Use

### What It Is

Tool use (function calling) lets an LLM request structured actions from external systems. The model outputs a JSON object describing which function to call and with what arguments. Your application code executes the function and returns the result.

```
User message
    |
    v
+-------------------+
|   LLM Inference   |  -- Model sees tool schemas in its context
+-------------------+
    |
    v
Decision: text response OR tool call
    |
    +---> Text: return to user
    |
    +---> Tool call: { "name": "get_weather", "arguments": {"city": "Tokyo"} }
              |
              v
          Your code executes get_weather("Tokyo")
              |
              v
          Result: {"temp": 22, "condition": "sunny"}
              |
              v
          Result appended to conversation, sent back to LLM
              |
              v
          LLM generates final text response
```

### Tool Schemas

Tools are defined using JSON Schema. The schema includes the tool name, a description (critical for model selection), and parameter definitions.

```python
# OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": (
                "Search for customer orders by order ID, email, or date range. "
                "Returns order status, items, and shipping information. "
                "Use this when the customer asks about an existing order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID in format ORD-XXXXX"
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer email address"
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "format": "date"},
                            "end": {"type": "string", "format": "date"}
                        }
                    }
                },
                "required": []  # At least one should be provided
            }
        }
    }
]

# Anthropic format — same schema, slightly different wrapper
tools_anthropic = [
    {
        "name": "search_orders",
        "description": "Search for customer orders by order ID, email, or date range...",
        "input_schema": {
            "type": "object",
            "properties": { ... },  # Same JSON Schema as above
            "required": []
        }
    }
]
```

### Parameter Types and Descriptions

Good parameter design directly affects how well the model uses your tools.

| Type | Use Case | Example |
|------|----------|---------|
| `string` | Free-form text, IDs | `"query": {"type": "string"}` |
| `string` + `enum` | Constrained choices | `"priority": {"type": "string", "enum": ["low", "medium", "high"]}` |
| `integer` / `number` | Counts, amounts | `"limit": {"type": "integer", "minimum": 1, "maximum": 100}` |
| `boolean` | Flags | `"include_deleted": {"type": "boolean"}` |
| `array` | Lists of values | `"tags": {"type": "array", "items": {"type": "string"}}` |
| `object` | Nested structures | `"address": {"type": "object", "properties": {...}}` |

**Interview tip:** The description field on each parameter is arguably more important than the type. The model reads descriptions to understand what values to provide. "The search query -- be specific and include relevant keywords" is far better than no description at all.

### Provider Differences in Tool Calling

#### OpenAI

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
)

# Parallel tool calls: OpenAI can return multiple tool calls in one response
for tool_call in response.choices[0].message.tool_calls:
    result = execute(tool_call.function.name, json.loads(tool_call.function.arguments))
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
```

#### Anthropic

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=messages,
    tools=tools,
    max_tokens=4096
)

# Anthropic returns content blocks — can mix text and tool_use in one response
for block in response.content:
    if block.type == "tool_use":
        result = execute(block.name, block.input)
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            }]
        })
```

#### Google (Gemini)

```python
# Google uses a similar pattern with FunctionDeclaration
from google.generativeai.types import FunctionDeclaration, Tool

weather_func = FunctionDeclaration(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)
tool = Tool(function_declarations=[weather_func])
```

#### Key Differences Summary

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Tool result role | `"tool"` | `"user"` with `tool_result` block | `FunctionResponse` |
| Parallel calls | Native support | Supported | Supported |
| Force specific tool | `tool_choice: {function: {name}}` | `tool_choice: {type: "tool", name: "..."}` | `tool_config` |
| Text + tool in same response | No (either text or tools) | Yes (mixed content blocks) | Yes |
| Schema format | `parameters` | `input_schema` | `parameters` |

### The Model Does Not Execute

This point cannot be overstated in interviews. The model outputs a structured request. It has zero capability to run code, make HTTP calls, or access databases. The execution layer is entirely your responsibility. This separation is what makes tool use safe and controllable.

---

## 2. The Agent Loop

### Core Architecture

An agent is an LLM in a loop. The loop continues until the model responds with text (no tool calls) or a maximum iteration count is reached.

```
                    +------------------+
                    |   User Message   |
                    +--------+---------+
                             |
                             v
                 +-----------+-----------+
            +--->|   Send to LLM with    |
            |    |   messages + tools     |
            |    +-----------+-----------+
            |                |
            |                v
            |    +-----------+-----------+
            |    |   LLM Response        |
            |    +-----------+-----------+
            |                |
            |        +-------+-------+
            |        |               |
            |        v               v
            |   Tool Call(s)    Text Response
            |        |               |
            |        v               v
            |   Execute tool    Return to user
            |        |           (LOOP ENDS)
            |        v
            |   Append result
            |   to messages
            |        |
            +--------+
         (back to top, unless
          max iterations hit)
```

### Exit Conditions

Every agent loop needs clear termination conditions:

1. **Model returns text only** -- the natural exit. The model decided it has enough information.
2. **Max iterations reached** -- safety valve. Prevents infinite loops and runaway costs.
3. **Stop tool invoked** -- some designs include an explicit "done" tool the model calls.
4. **Error threshold** -- too many consecutive errors triggers graceful exit.
5. **Token/cost budget exhausted** -- hard limit on spend per agent invocation.

### Implementation Skeleton

```python
async def agent_loop(
    messages: list[dict],
    tools: list[dict],
    max_iterations: int = 10,
) -> str:
    """Core agent loop. Returns the final text response."""
    for i in range(max_iterations):
        response = await call_llm(messages, tools)

        # Check if model wants to use tools
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # Model responded with text -- we're done
            return extract_text(response)

        # Execute each tool call and append results
        for tool_call in tool_calls:
            result = await execute_tool(tool_call)
            messages.append(format_tool_result(tool_call, result))

        # Append the assistant's response (with tool calls) to history
        messages.append(format_assistant_message(response))

    return "Max iterations reached. Here's what I found so far..."
```

---

## 3. ReAct Pattern (Reasoning + Acting)

### What It Is

ReAct (Yao et al., 2022) interleaves reasoning traces with tool actions. Instead of the model silently deciding to call a tool, it explicitly states its reasoning before each action.

```
Thought: The user wants to know their order status. I need their order ID.
         They mentioned order #ORD-4521. Let me look it up.
Action:  get_order(order_id="ORD-4521")
Observation: {"status": "shipped", "carrier": "UPS", "tracking": "1Z999..."}

Thought: The order is shipped via UPS. Let me get detailed tracking info.
Action:  get_tracking(tracking_number="1Z999...")
Observation: {"current_location": "Memphis, TN", "eta": "2025-03-15"}

Thought: I have all the information needed to answer the user's question.
Answer:  Your order ORD-4521 has shipped via UPS. It's currently in
         Memphis, TN with an estimated delivery of March 15th.
```

### Why It Works Better Than Acting Alone

1. **Explicit reasoning improves tool selection.** The model is less likely to call the wrong tool when it first articulates why it needs information.
2. **Observations ground the model.** Real results replace potential hallucinations.
3. **Debuggability.** You can read the thought trace to understand why the agent made each decision.
4. **Error recovery.** When a tool returns an error, the model can reason about alternatives.

### Implementation Approaches

**Approach 1: Prompt-based ReAct** -- instruct the model to output Thought/Action/Observation format in its text responses. You parse the text to extract tool calls. This was the original approach.

**Approach 2: Native tool use as ReAct** -- modern tool-use APIs naturally implement the ReAct loop. The model's internal reasoning (sometimes exposed via chain-of-thought or extended thinking) serves as the "Thought," the tool call is the "Action," and the tool result is the "Observation." No special parsing needed.

**Interview insight:** Most production systems today use Approach 2. The native tool-use APIs make explicit ReAct prompting unnecessary in many cases, though the conceptual framework remains valuable for understanding what the model is doing internally.

---

## 4. Planning Patterns

### Implicit Planning

The model decides what to do next based on conversation context, with no explicit planning step. This is the default behavior of any agent loop.

**When to use:** Simple tasks with 1-3 tool calls. Customer service lookups, single-step data retrieval.

**Limitation:** The model can lose track of multi-step plans, repeat steps, or miss steps entirely on complex tasks.

### Explicit Plan-Then-Execute

Force the model to output a plan before taking any action.

```
System prompt:
    Before using any tools, first output your plan:

    PLAN:
    1. [First step and why]
    2. [Second step and why]
    ...

    Then execute each step using the available tools. After each step,
    check if your plan needs adjustment.
```

**When to use:** Tasks requiring 4+ steps, tasks where order matters, tasks where the user would benefit from seeing the plan upfront.

**Tradeoff:** More tokens consumed, but significantly more reliable on complex tasks.

### Plan-and-Revise

Generate an initial plan, execute steps, and revise the plan as new information arrives.

```
Step 1: Generate plan
    "I'll search for the user's account, check their subscription,
     then process the upgrade."

Step 2: Execute step 1 -> get result
    Account found, but user has two accounts.

Step 3: Revise plan
    "There are two accounts. I need to ask which one, then proceed
     with the upgrade on the correct account."

Step 4: Continue with revised plan
```

**When to use:** Research tasks, data analysis, any task where early results change the strategy.

---

## 5. Tool Design Best Practices

### Naming and Descriptions

The tool name and description are how the model decides *when* to use each tool. Think of the description as a prompt.

```python
# Bad: vague, model won't know when to use it
{"name": "search", "description": "Search for things"}

# Good: specific trigger conditions and return value description
{
    "name": "search_knowledge_base",
    "description": (
        "Search the company knowledge base for product information, "
        "return policies, warranty details, and troubleshooting guides. "
        "Use when the customer asks a question about our products or policies. "
        "Returns relevant articles ranked by relevance with titles and snippets."
    )
}
```

### Error Messages That Help Recovery

When a tool call fails, the error message goes back to the model. Design errors that help the model try again successfully.

```python
# Bad: the model has no idea what to do next
{"error": "Invalid input"}

# Good: the model can reason about alternatives
{
    "error": "No customer found with email 'john@example.com'",
    "suggestion": "Try searching by name or phone number instead",
    "available_search_fields": ["name", "phone", "customer_id"]
}
```

### Idempotency

Tools that modify state should be idempotent when possible. If the agent loop retries (due to errors, timeouts, or the model calling the same tool again), idempotent tools prevent duplicate side effects.

```python
# Non-idempotent: calling twice creates two tickets
def create_ticket(title, description): ...

# Idempotent: uses a client-generated key to prevent duplicates
def create_ticket(title, description, idempotency_key): ...
```

### Parameter Design Rules

1. **Require only what is necessary.** Optional parameters with defaults reduce friction.
2. **Use enums for constrained values.** The model selects from valid options rather than guessing.
3. **Describe each parameter.** "The customer's email address" is better than nothing.
4. **Validate on your side.** Never trust model-generated arguments blindly.
5. **Limit tool count.** 5-15 tools is a sweet spot. Beyond 20, selection accuracy degrades.

---

## 6. Model Context Protocol (MCP)

### What It Is

MCP is an open standard (introduced by Anthropic in late 2024) for connecting LLMs to external data sources and tools. It standardizes how tools are discovered, described, and invoked, replacing one-off integrations with a common protocol.

### Architecture

```
+-------------------+          +-------------------+
|   MCP Client      |  <---->  |   MCP Server      |
| (LLM application) |  JSON-   | (tool provider)   |
|                   |  RPC     |                   |
| - Discovers tools |  over    | - Exposes tools   |
| - Sends requests  |  stdio   | - Executes calls  |
| - Receives results|  or SSE  | - Returns results |
+-------------------+          +-------------------+
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Server** | Process that exposes tools, resources, and prompts via MCP |
| **Client** | Application that connects to MCP servers and uses their tools |
| **Tools** | Functions the server exposes (same as function calling tools) |
| **Resources** | Data the server can provide (files, DB records, API data) |
| **Prompts** | Template prompts the server suggests for common tasks |
| **Transport** | Communication layer: stdio (local) or SSE (remote) |

### Why It Matters

1. **Standardization.** One protocol for all tool integrations instead of custom code per tool.
2. **Ecosystem.** A growing library of pre-built MCP servers (databases, APIs, file systems).
3. **Composability.** An MCP client can connect to multiple servers simultaneously.
4. **Separation of concerns.** Tool implementation is decoupled from the LLM application.

### Connection Flow

```
1. Client starts/connects to MCP server
2. Client calls `initialize` to negotiate capabilities
3. Client calls `tools/list` to discover available tools
4. Client includes discovered tools in LLM API calls
5. When LLM requests a tool call:
   a. Client calls `tools/call` on the appropriate MCP server
   b. Server executes the tool and returns the result
   c. Client passes result back to the LLM
```

### Interview Framing

MCP solves the "N x M integration problem." Without it, N applications each need custom integrations for M tool providers (N x M total). With MCP, each application implements one client, and each provider implements one server (N + M total). This is analogous to how USB standardized peripheral connectivity or how LSP standardized editor-language integration.

---

## 7. Memory Systems

Agents need memory beyond the immediate conversation. Different memory types serve different purposes.

### Memory Taxonomy

```
+-------------------------------------------------------------------+
|                        AGENT MEMORY                                |
|                                                                    |
|  SHORT-TERM (within session)          LONG-TERM (across sessions)  |
|  +-----------------------------+     +---------------------------+ |
|  | Conversation History        |     | Episodic Memory           | |
|  | - Current messages          |     | - Past interaction logs   | |
|  | - Tool call results         |     | - Successful strategies   | |
|  | - Working context           |     | - User preferences        | |
|  +-----------------------------+     +---------------------------+ |
|  +-----------------------------+     +---------------------------+ |
|  | Working Memory              |     | Semantic Memory           | |
|  | - Current plan/state        |     | - Knowledge base (RAG)    | |
|  | - Intermediate results      |     | - Domain facts            | |
|  | - Scratchpad                |     | - Documentation           | |
|  +-----------------------------+     +---------------------------+ |
|                                      +---------------------------+ |
|                                      | Procedural Memory         | |
|                                      | - Learned tool sequences  | |
|                                      | - Task-specific prompts   | |
|                                      | - Refined strategies      | |
|                                      +---------------------------+ |
+-------------------------------------------------------------------+
```

### Conversation Memory Strategies

#### Full History
Pass every message to the model each time. Simple but hits context window limits.

#### Sliding Window
Keep only the last N messages plus the system prompt.

```python
def sliding_window(messages: list[dict], window_size: int = 20) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    return system + others[-window_size:]
```

**Problem:** Loses early context. If the user mentioned their name in message 3, the model forgets it by message 25.

#### Summarization

Periodically compress older messages into a summary.

```python
async def summarize_and_trim(messages: list[dict], threshold: int = 30) -> list[dict]:
    if len(messages) < threshold:
        return messages

    system = [m for m in messages if m["role"] == "system"]
    old_messages = messages[len(system):threshold - 10]
    recent_messages = messages[threshold - 10:]

    summary = await call_llm([
        {"role": "system", "content": "Summarize this conversation concisely."},
        *old_messages
    ])

    return system + [
        {"role": "system", "content": f"Previous conversation summary: {summary}"}
    ] + recent_messages
```

#### Retrieval-Based (RAG over conversation)

Store all messages in a vector database. For each new turn, retrieve the most relevant past messages. Best for long-running agents with extensive histories.

### Memory Comparison

| Strategy | Context Usage | Information Loss | Complexity | Best For |
|----------|--------------|-----------------|------------|----------|
| Full history | High (grows linearly) | None | Low | Short conversations |
| Sliding window | Fixed | Loses old context | Low | Casual chat |
| Summarization | Moderate | Lossy compression | Medium | Multi-turn agents |
| Retrieval-based | Moderate | Selective recall | High | Long-running assistants |

---

## 8. Guardrails and Safety

### Tool Call Validation

Never execute a tool call without validating its arguments against the schema.

```python
import jsonschema

def validate_tool_call(tool_name: str, arguments: dict, tool_schemas: dict) -> tuple[bool, str]:
    schema = tool_schemas.get(tool_name)
    if not schema:
        return False, f"Unknown tool: {tool_name}"

    try:
        jsonschema.validate(arguments, schema["parameters"])
        return True, ""
    except jsonschema.ValidationError as e:
        return False, str(e.message)
```

### Confirmation for Destructive Actions

High-stakes operations should require explicit human approval.

```python
DESTRUCTIVE_TOOLS = {"delete_account", "issue_refund", "send_email", "transfer_funds"}

async def execute_with_approval(tool_call: dict) -> dict:
    if tool_call["name"] in DESTRUCTIVE_TOOLS:
        approved = await request_human_approval(
            action=tool_call["name"],
            arguments=tool_call["arguments"],
        )
        if not approved:
            return {"error": "Action not approved by user", "status": "rejected"}

    return await execute_tool(tool_call)
```

### Sandboxing

Code execution tools must run in isolated environments.

```
Production sandboxing options:
  - Docker containers (resource-limited, network-restricted)
  - E2B sandboxes (cloud-hosted, purpose-built for AI code execution)
  - gVisor / Firecracker (lightweight VM isolation)
  - WASM runtimes (language-level sandboxing)
```

### Principle of Least Privilege

Each tool should have the minimum permissions required.

```python
# Define permission tiers
TOOL_PERMISSIONS = {
    "search_products": "read",      # No auth needed
    "get_order_status": "read",     # Needs user context
    "update_shipping": "write",     # Needs write access
    "issue_refund": "admin",        # Needs admin + approval
}

def check_permission(tool_name: str, user_role: str) -> bool:
    required = TOOL_PERMISSIONS.get(tool_name, "admin")  # Default to highest
    hierarchy = ["read", "write", "admin"]
    return hierarchy.index(user_role) >= hierarchy.index(required)
```

### Human-in-the-Loop Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Always approve** | Every tool call requires approval | Development, high-risk domains |
| **Approve destructive only** | Read operations auto-execute, writes need approval | Most production systems |
| **Confidence threshold** | Auto-execute when model confidence is high | Research, low-risk operations |
| **Audit log only** | Auto-execute everything, log for review | Low-stakes, high-volume |
| **Escalation** | Auto-execute up to N failures, then escalate to human | Customer service |

---

## 9. Structured Output vs. Tool Use

Both produce structured data from the model, but they serve fundamentally different purposes.

### When to Use Each

| Scenario | Use Structured Output | Use Tool Use |
|----------|----------------------|--------------|
| Extract entities from text | Yes | No |
| Query a database | No | Yes |
| Classify customer intent | Yes | No |
| Send an email | No | Yes |
| Parse a document into JSON | Yes | No |
| Search a knowledge base | No | Yes |
| Generate a report format | Yes | No |
| Execute code | No | Yes |

### How They Complement Each Other

A common pattern: an agent uses tools to gather data, then returns structured output as its final response.

```python
# Agent uses tools to gather data
# Step 1: search_orders(customer_id="C-123")  -> tool use
# Step 2: get_return_policy(product_type="electronics")  -> tool use
# Step 3: Return structured response  -> structured output

final_response_schema = {
    "type": "object",
    "properties": {
        "eligible_for_return": {"type": "boolean"},
        "reason": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "reference_number": {"type": "string"}
    }
}
```

### Provider-Specific Structured Output

```python
# OpenAI: response_format parameter
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_schema", "json_schema": {"name": "analysis", "schema": schema}}
)

# Anthropic: tool use as structured output (define a tool, force its use)
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=messages,
    tools=[{"name": "structured_response", "description": "...", "input_schema": schema}],
    tool_choice={"type": "tool", "name": "structured_response"}
)
```

---

## 10. Provider Differences — Deep Comparison

### OpenAI

- **Parallel tool calls:** The model can request multiple tool calls in a single response. Each has a unique `tool_call_id` that must be referenced in the corresponding tool result.
- **Tool choice:** `"auto"` (model decides), `"required"` (must use a tool), `"none"` (text only), or force a specific function.
- **Strict mode:** Can enforce that the model's output exactly matches the JSON Schema (no extra fields, correct types).
- **Assistants API:** Higher-level abstraction with built-in file search, code interpreter, and persistent threads.

### Anthropic (Claude)

- **Mixed content blocks:** A single response can contain both text and tool_use blocks. The model might explain its reasoning in text while simultaneously making a tool call.
- **Tool result format:** Tool results are sent as `tool_result` content blocks within a `user` role message, referencing the `tool_use_id`.
- **Extended thinking:** Claude can show its reasoning process, which maps naturally to the "Thought" step in ReAct.
- **Computer use:** Claude supports a special `computer_20241022` tool type for interacting with desktop environments via screenshots and mouse/keyboard actions.

### Open-Source Models

- **Variable support:** Function calling quality varies significantly. Llama 3.1+ and Mistral have decent support. Smaller models struggle.
- **Special tokens vs. prompt-based:** Some models use special tokens to delimit tool calls; others rely on prompt engineering to produce parseable output.
- **Frameworks help:** Libraries like Outlines, LangChain, and vLLM provide structured generation to force valid tool call output from open-source models.

---

## Interview Questions You Should Be Ready For

### Conceptual

1. **"Explain how function calling works. Does the model execute the function?"**
   No. The model outputs a structured request (function name + arguments). Your code executes it and returns the result. The model has no execution capability.

2. **"What is the ReAct pattern and why does it improve agent performance?"**
   ReAct interleaves explicit reasoning (Thought) with tool invocation (Action) and result processing (Observation). Explicit reasoning before action improves tool selection accuracy and enables error recovery through deliberate analysis of results.

3. **"How would you handle an agent that gets stuck in a loop?"**
   Max iteration limits, consecutive error tracking (circuit breaker), cost budgets, and loop detection (same tool called with same arguments repeatedly). Return a graceful summary of progress when any limit is hit.

4. **"What is MCP and why does it matter?"**
   Model Context Protocol standardizes tool interfaces. Instead of N applications each building M custom tool integrations (N*M work), MCP lets each application implement one client and each tool provider implement one server (N+M work). It is the USB or LSP of agent tooling.

### System Design

5. **"Design an agent for a customer service chatbot."**
   Tools: search_kb, get_order, create_ticket, transfer_to_human. Agent loop with max 10 iterations. Sliding window + summarization memory. Human approval for refunds. Permission tiers based on customer authentication. Graceful escalation to human after 3 failed resolution attempts.

6. **"How would you add memory to an agent that handles long conversations?"**
   Start with sliding window (simple, predictable cost). Add summarization when conversations exceed 30 messages. For cross-session memory, store summaries and key facts in a database. For truly long-term memory, use RAG over past conversations.

### Implementation

7. **"Walk me through implementing a basic agent loop."**
   See the agent_loop function in Section 2. Key decisions: max iterations, how to format tool results for the provider, error handling strategy, exit conditions.

8. **"How do you validate tool calls before execution?"**
   Validate arguments against the JSON Schema (required fields, types, enum values). Check permissions. For destructive operations, require human approval. Return validation errors as tool results so the model can self-correct.

---

## Key Takeaways

1. **The model requests, your code executes.** This separation is the foundation of safe tool use.
2. **Agent = LLM + tools + loop.** The loop is where the intelligence emerges from iteration.
3. **Tool descriptions are prompts.** Invest in good descriptions; they determine selection accuracy.
4. **Plan for failure.** Max iterations, error recovery, circuit breakers, graceful degradation.
5. **Memory is a spectrum.** Choose the simplest strategy that meets your needs.
6. **MCP standardizes the ecosystem.** One protocol, many integrations.
7. **Guardrails are not optional.** Validate inputs, confirm destructive actions, sandbox execution.
8. **ReAct = reasoning before acting.** Explicit reasoning traces improve accuracy and debuggability.
