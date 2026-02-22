# Agents & Tool Use — Core Concepts

## What Is Tool Use / Function Calling?

Tool use (also called function calling) allows an LLM to request structured actions from external systems instead of just generating text. The model doesn't execute anything itself — it outputs a structured request, your code executes it, and the result is fed back.

```
User: "What's the weather in Tokyo?"
    ↓
LLM decides to call a tool:
    { "name": "get_weather", "arguments": { "city": "Tokyo" } }
    ↓
Your code executes the tool:
    getWeather("Tokyo") → { temp: 22, condition: "sunny" }
    ↓
Result sent back to LLM:
    LLM: "It's currently 22°C and sunny in Tokyo."
```

### How It Works (Provider-Agnostic)

1. **Define tools:** Provide the model with tool schemas (name, description, parameters)
2. **Model decides:** Based on the conversation, the model either responds with text OR requests a tool call
3. **Execute:** Your code validates and executes the tool call
4. **Return result:** Send the tool result back to the model
5. **Continue:** The model uses the result to continue the conversation

---

## Tool Schemas

Tools are defined with a name, description, and parameter schema. The description is critical — it's how the model decides *when* to use each tool.

```python
@dataclass
class ToolDefinition:
    name: str
    description: str  # This is your "prompt" for tool selection
    parameters: dict  # JSON Schema object, e.g.:
    # {
    #     "type": "object",
    #     "properties": {
    #         "query": {"type": "string", "description": "..."},
    #     },
    #     "required": ["query"],
    # }
```

### Example Tool Definitions

```python
tools = [
    ToolDefinition(
        name="search_knowledge_base",
        description=(
            "Search the company knowledge base for information about products, "
            "policies, or procedures. Use this when the user asks a question "
            "that requires specific company information."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific and include relevant keywords",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to narrow results",
                    "enum": ["products", "policies", "procedures", "faq"],
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="create_ticket",
        description=(
            "Create a support ticket in the ticketing system. Use this when "
            "the user has an issue that cannot be resolved in conversation "
            "and needs to be escalated."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Brief title describing the issue"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Issue priority level",
                },
                "description": {"type": "string", "description": "Detailed description of the issue"},
            },
            "required": ["title", "priority", "description"],
        },
    ),
]
```

### Tips for Tool Descriptions

- **Be specific about when to use it** — the model uses the description to decide
- **Describe what it returns** — helps the model anticipate the result
- **Include constraints** — e.g., "Only use this for confirmed customers"
- **Parameter descriptions matter** — they guide the model in constructing arguments

---

## The Agent Loop

An agent is an LLM that can use tools in a loop — calling tools, processing results, and deciding what to do next until the task is complete.

```
┌──────────────────────────────────────────┐
│              AGENT LOOP                   │
│                                           │
│  1. Send messages + tools to LLM          │
│         ↓                                 │
│  2. LLM responds with:                    │
│     ├─ Text → Return to user (DONE)       │
│     └─ Tool call → Continue to step 3     │
│         ↓                                 │
│  3. Execute tool, get result              │
│         ↓                                 │
│  4. Append tool result to messages        │
│         ↓                                 │
│  5. Go to step 1                          │
│         (unless max iterations reached)   │
└──────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Options | Guidance |
|---|---|---|
| Max iterations | 3–20 | Depends on task complexity; start at 10 |
| Parallel tool calls | Sequential / Parallel | Some providers support parallel; simpler to start sequential |
| Error strategy | Retry / Skip / Abort | Return error to model so it can reason about alternatives |
| Memory | Full history / Summarized / Sliding window | Full history until context limit, then summarize |

---

## Planning Patterns

How does an agent decide what to do? Several patterns exist:

### Implicit Planning (Default)

The model plans implicitly through its text generation. No explicit planning step — the model just decides what tool to call next based on the conversation.

**Pros:** Simpler, works for straightforward tasks
**Cons:** Can lose track of complex multi-step plans

### Explicit Planning

Ask the model to output a plan before acting:

```
System: Before using any tools, first output a plan in this format:
Plan:
1. [step 1]
2. [step 2]
...

Then execute each step using the available tools.
```

**Pros:** More reliable for complex tasks, easier to debug
**Cons:** Uses more tokens, plan may need revision as new information arrives

### Plan-and-Revise

Generate a plan, execute steps, revise the plan when new information changes things:

```
1. Generate initial plan
2. Execute step 1 → get result
3. Evaluate: does the plan still make sense? Revise if needed
4. Execute next step → get result
5. Repeat until done
```

---

## Multi-Turn Tool Use

In real applications, a single user request often requires multiple tool calls:

```
User: "Transfer $500 from my checking to savings"

Turn 1 (LLM): Call get_account_balance({ account: "checking" })
Result: { balance: 1200 }

Turn 2 (LLM): Call get_account_balance({ account: "savings" })
Result: { balance: 3000 }

Turn 3 (LLM): Call transfer({ from: "checking", to: "savings", amount: 500 })
Result: { success: true, confirmation: "TXN-12345" }

Turn 4 (LLM): "Done! I've transferred $500 from your checking to savings.
Your new balances: Checking: $700, Savings: $3,500.
Confirmation: TXN-12345"
```

The model naturally chains tool calls — each result informs the next action.

---

## Structured Output vs. Tool Use

Both produce structured data, but they're different:

| | Structured Output | Tool Use |
|---|---|---|
| **Purpose** | Format the model's response | Trigger external actions |
| **Who acts** | The model generates data | Your code executes the action |
| **Format** | JSON schema for the response | Function call with arguments |
| **Example** | "Return analysis as JSON" | "Call the database query function" |

**They combine well:** An agent might use tools to gather data, then return a structured JSON response.

---

## Provider Differences

### OpenAI
- `tools` parameter in API request
- Supports parallel tool calls
- `tool_choice: "auto" | "required" | { function: { name } }`
- Tool results via `role: "tool"` messages

### Anthropic (Claude)
- `tools` parameter, similar schema
- Tool results via `tool_result` content blocks
- Supports forcing specific tool use
- Can return text + tool calls in the same response

### Open Source (Llama, Mistral, etc.)
- Function calling support varies by model and fine-tune
- Some use special tokens, others use prompt-based approaches
- Libraries like LangChain / LlamaIndex provide a uniform interface

### Key Commonality

Despite API differences, the pattern is universal:
1. Define tools as schemas
2. Model outputs structured tool calls
3. Your code executes and returns results
4. Model continues with the result
