# Module 04: Agents & Tool Use — Cheat Sheet

Quick reference for agent patterns, tool schemas, and debugging strategies.

---

## Agent Pattern Decision Tree

```
What kind of task is this?
|
+-- Simple (1-3 tool calls, linear flow)
|   --> Single agent, implicit planning, basic loop
|
+-- Medium (4-10 tool calls, some branching)
|   --> Single agent, explicit plan-then-execute
|   --> Consider ReAct for traceability
|
+-- Complex (10+ tool calls, multiple domains)
|   +-- Single domain, many steps?
|   |   --> Single agent, plan-and-revise, summarization memory
|   |
|   +-- Multiple domains, specialized knowledge?
|   |   --> Orchestrator + specialist agents
|   |
|   +-- Requires iteration/refinement?
|       --> Peer-to-peer (writer/critic) or supervisor/worker
|
+-- Unbounded (open-ended research, coding)
    --> Long-running agent with checkpointing
    --> Context management (summarization)
    --> Idempotent tools, cost budgets
```

---

## Tool Schema Template

Copy-paste starting point for defining tools:

```python
tool_template = {
    "type": "function",  # OpenAI format
    "function": {
        "name": "verb_noun",  # e.g., search_orders, create_ticket, get_balance
        "description": (
            "What this tool does. "
            "When to use it (trigger conditions). "
            "What it returns (so the model knows what to expect)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "What this parameter represents"
                },
                "optional_param": {
                    "type": "string",
                    "description": "What this parameter represents",
                    "enum": ["option_a", "option_b", "option_c"]  # Use enums when possible
                },
                "numeric_param": {
                    "type": "integer",
                    "description": "What this number represents",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["required_param"]
        }
    }
}

# Anthropic format: same schema, different wrapper
tool_template_anthropic = {
    "name": "verb_noun",
    "description": "...",
    "input_schema": {
        "type": "object",
        "properties": { ... },
        "required": [...]
    }
}
```

---

## MCP Overview

### What It Is

Model Context Protocol -- open standard for connecting LLMs to tools and data sources.

### Core Components

```
MCP Client (your app)  <-->  MCP Server (tool provider)
    |                            |
    |-- initialize()             |-- Exposes tools
    |-- tools/list()             |-- Exposes resources
    |-- tools/call()             |-- Exposes prompts
    |-- resources/read()         |
```

### Connection Flow

```
1. Client starts server process (stdio) or connects (SSE/HTTP)
2. Client sends `initialize` with capabilities
3. Server responds with supported capabilities
4. Client calls `tools/list` to discover tools
5. Client includes tools in LLM API calls
6. When LLM requests tool call:
   a. Client calls `tools/call` on MCP server
   b. Server executes and returns result
   c. Client feeds result back to LLM
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| Transport | stdio (local process) or SSE/HTTP (remote) |
| Tools | Functions the server exposes for LLM use |
| Resources | Data the server provides (files, records, etc.) |
| Prompts | Template prompts the server suggests |
| Sampling | Server can request LLM completions from client |

### Why It Matters

Solves N x M problem: N apps x M tool providers = N*M integrations.
With MCP: N clients + M servers = N+M integrations.
Analogous to USB (peripherals) or LSP (editor-language integration).

---

## Memory System Comparison

| Type | Scope | Implementation | Token Cost | Info Loss | Best For |
|------|-------|---------------|-----------|-----------|----------|
| **Full history** | Single session | Keep all messages | High (grows) | None | Short conversations (<20 msgs) |
| **Sliding window** | Single session | Keep last N messages | Fixed | Loses early context | Casual chat, stateless tasks |
| **Summarization** | Single session | Compress old messages | Moderate | Lossy compression | Multi-turn agents (20-100 msgs) |
| **Retrieval (RAG)** | Cross-session | Vector DB over messages | Moderate | Selective recall | Long-running assistants |
| **Episodic** | Cross-session | Store interaction logs | Low (on-demand) | Query-dependent | Personalization |
| **Semantic** | Permanent | Knowledge base | Low (on-demand) | None (curated) | Domain expertise |
| **Procedural** | Permanent | Refined prompts/strategies | None (pre-set) | None | Task optimization |

### Quick Decision Guide

```
How long is the conversation?
  < 20 messages  --> Full history
  20-100 messages --> Sliding window + summarization
  100+ messages   --> Retrieval-based

Do you need cross-session memory?
  No  --> In-memory is fine
  Yes --> Database + retrieval
```

---

## Common Failure Modes and Debugging

### Failure Mode Table

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Model never uses tools | Bad tool descriptions, too many tools | Rewrite descriptions with clear trigger conditions |
| Model always uses tools (even when unnecessary) | `tool_choice: "required"` or description too broad | Set `tool_choice: "auto"`, narrow descriptions |
| Model calls wrong tool | Similar tool descriptions, ambiguous names | Differentiate descriptions, add "do NOT use for..." |
| Model passes wrong arguments | Poor parameter descriptions | Add examples in parameter descriptions |
| Agent loops forever | No max iterations, model repeats same call | Add max iterations + loop detection |
| Agent gives up too early | Max iterations too low, error handling too strict | Increase limit, return errors as results |
| Context window exceeded | Too many tool results, long conversations | Summarize, truncate large tool results |
| Inconsistent tool call format | Open-source model, prompt mismatch | Use structured generation (Outlines) or switch to a model with native function calling |

### Debugging Checklist

```
1. Log every LLM request and response (messages sent, response received)
2. Log every tool call: name, arguments, result, duration
3. Track iteration count per agent invocation
4. Monitor token usage per turn (input + output)
5. Record the full message history at the point of failure
6. Check: is the tool description clear about WHEN to use it?
7. Check: are error messages informative enough for the model to recover?
8. Check: is the system prompt conflicting with tool descriptions?
```

### Performance Optimization

| Technique | Impact | Tradeoff |
|-----------|--------|----------|
| Parallel tool calls | Reduces latency | More complex result handling |
| Cache tool results | Reduces cost + latency | Stale data risk |
| Smaller model for routing | Reduces cost | Lower accuracy on routing |
| Summarize large tool results | Reduces tokens | Information loss |
| Limit tool count per agent | Improves selection accuracy | Less capability per agent |

---

## Framework Comparison Matrix

| Feature | Roll Your Own | LangChain/LangGraph | CrewAI | AutoGen | OpenAI Assistants |
|---------|--------------|-------------------|--------|---------|-------------------|
| Setup complexity | Low | Medium | Low | High | Low |
| Multi-agent | DIY | Built-in (LangGraph) | Built-in | Built-in | No |
| Persistence | DIY | Built-in | Limited | Built-in | Built-in |
| Streaming | DIY | Built-in | Limited | Limited | Built-in |
| Observability | DIY | LangSmith | Limited | Limited | Dashboard |
| Provider lock-in | None | None | None | None | OpenAI only |
| Learning curve | Low | High | Medium | High | Low |
| Customization | Full | High | Medium | Medium | Low |
| Production readiness | Depends on you | Mature | Maturing | Maturing | Mature |

### The 80/20 Rule

- **80% of agent use cases:** A hand-written loop (50-100 LOC) + good tool definitions + proper error handling.
- **20% needing frameworks:** Multi-agent with complex routing, persistence, visual debugging, and streaming.

---

## Provider API Quick Reference

### OpenAI

```python
# Tool call response structure
response.choices[0].message.tool_calls[i].id           # "call_abc123"
response.choices[0].message.tool_calls[i].function.name # "search_orders"
response.choices[0].message.tool_calls[i].function.arguments  # '{"query": "..."}'

# Tool result message
{"role": "tool", "tool_call_id": "call_abc123", "content": "..."}

# Tool choice options
tool_choice="auto"          # Model decides
tool_choice="required"      # Must use a tool
tool_choice="none"          # Text only
tool_choice={"type": "function", "function": {"name": "search_orders"}}  # Force specific
```

### Anthropic

```python
# Tool call in response content blocks
response.content[i].type    # "tool_use"
response.content[i].id      # "toolu_abc123"
response.content[i].name    # "search_orders"
response.content[i].input   # {"query": "..."}

# Tool result message
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "..."}]}

# Tool choice options
tool_choice={"type": "auto"}
tool_choice={"type": "any"}           # Must use a tool
tool_choice={"type": "tool", "name": "search_orders"}  # Force specific
```

### Google Gemini

```python
# Tool call in response
response.candidates[0].content.parts[i].function_call.name  # "search_orders"
response.candidates[0].content.parts[i].function_call.args  # {"query": "..."}

# Tool result
genai.protos.Part(function_response=genai.protos.FunctionResponse(
    name="search_orders", response={"result": "..."}
))
```

---

## Key Numbers to Know

| Metric | Typical Range | Notes |
|--------|--------------|-------|
| Tools per agent | 5-15 | >20 degrades selection accuracy |
| Max iterations | 5-25 | Depends on task complexity |
| Agent loop latency | 2-30s per iteration | Dominated by LLM API call |
| Tool call accuracy | 85-95% | With good descriptions and capable model |
| Context for agent | 8K-200K tokens | Depends on provider and model |
| Cost per agent task | $0.01-$1.00 | Varies wildly by complexity and model |
| MCP server startup | <1s (stdio) | SSE connections are persistent |
