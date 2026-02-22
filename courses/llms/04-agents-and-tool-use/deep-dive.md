# Module 04: Agents & Tool Use — Deep Dive

Extended content covering multi-agent architectures, long-running agents, evaluation, frameworks, and advanced patterns. This material goes beyond core interview knowledge into system design territory.

---

## 1. Multi-Agent Architectures

### Why Multiple Agents?

A single agent with 50 tools and a complex system prompt becomes unreliable. Splitting responsibilities across specialized agents improves accuracy, makes systems easier to debug, and allows independent scaling.

### Architecture Patterns

#### Orchestrator Pattern

A central "orchestrator" agent receives the user request, decides which specialist to invoke, and synthesizes results.

```
                    +------------------+
                    |   User Request   |
                    +--------+---------+
                             |
                             v
                  +----------+----------+
                  |   Orchestrator      |
                  |   Agent             |
                  | - Routes tasks      |
                  | - Synthesizes       |
                  |   results           |
                  +----+-----+----+----+
                       |     |    |
            +----------+  +--+--+  +----------+
            v             v      v             v
     +------+------+ +---+----+ +------+------+
     | Research    | | Data   | | Writing     |
     | Agent       | | Agent  | | Agent       |
     | - Web search| | - SQL  | | - Formatting|
     | - Summarize | | - Viz  | | - Drafting  |
     +-------------+ +--------+ +-------------+
```

**Strengths:** Clear control flow, easy to add/remove specialists, orchestrator can prioritize.
**Weaknesses:** Orchestrator is a single point of failure, can become a bottleneck.

**Example: Coding assistant**
- Orchestrator: understands the request, decides what needs to happen
- Planner agent: breaks the task into implementation steps
- Coder agent: writes the code (has file read/write tools)
- Reviewer agent: reviews the code for bugs and style issues
- Tester agent: runs tests and reports failures

#### Supervisor / Worker

Similar to orchestrator but more dynamic. The supervisor monitors worker progress and can reassign or retry tasks.

```python
class Supervisor:
    async def run(self, task: str) -> str:
        plan = await self.plan_agent.create_plan(task)

        for step in plan.steps:
            worker = self.select_worker(step)
            result = await worker.execute(step)

            if not result.success:
                # Supervisor decides: retry, reassign, or adapt plan
                recovery = await self.decide_recovery(step, result)
                if recovery.action == "retry":
                    result = await worker.execute(step)
                elif recovery.action == "reassign":
                    alt_worker = self.select_alternative_worker(step)
                    result = await alt_worker.execute(step)
                elif recovery.action == "revise_plan":
                    plan = await self.plan_agent.revise(plan, result.error)

        return await self.synthesize(plan.results)
```

**When to use:** Tasks with uncertain outcomes, where adaptive replanning is needed.

#### Peer-to-Peer (Debate / Collaboration)

Agents communicate directly with each other without a central coordinator. Useful for adversarial setups (debate, red-team/blue-team) or collaborative refinement.

```
+--------+       +--------+
| Agent A| <---> | Agent B|
| Writer |       | Critic |
+--------+       +--------+
     |                |
     v                v
  Draft 1  --->  Feedback 1
  Draft 2  <---  Feedback 2
  Draft 3  --->  "Approved"
     |
     v
  Final Output
```

**Example: Research paper review**
- Writer agent drafts sections
- Critic agent reviews for accuracy, clarity, and completeness
- Writer revises based on feedback
- Loop continues until critic approves or max iterations hit

#### Hierarchical (Tree Structure)

For complex tasks that decompose recursively. A manager delegates to sub-managers who delegate to workers.

```
                  CEO Agent
                 /    |    \
           VP-Eng  VP-Data  VP-Design
           /   \      |       |
      FE    BE   Analytics  UX-Agent
      Agent Agent  Agent
```

**When to use:** Very large tasks (full application generation, comprehensive research reports). Rare in practice due to coordination overhead and error amplification.

### Choosing an Architecture

| Factor | Single Agent | Orchestrator | Supervisor/Worker | Peer-to-Peer |
|--------|-------------|-------------|-------------------|--------------|
| Task complexity | Low-medium | Medium-high | High | Medium |
| Tools needed | <15 | 15+ (split across specialists) | 15+ | Few per peer |
| Error recovery | Basic | Moderate | Strong | Moderate |
| Latency | Low | Medium (sequential routing) | Higher | Variable |
| Implementation cost | Low | Medium | High | Medium |
| Debuggability | Easy | Moderate | Hard | Hard |

---

## 2. Long-Running Agents

### The Problem

Context windows have finite capacity. An agent working on a multi-hour task will eventually exhaust its context. Token costs grow quadratically with context length (for attention computation). State can be lost on crashes.

### State Persistence

```python
@dataclass
class AgentState:
    task_id: str
    messages: list[dict]
    plan: list[str]
    completed_steps: list[str]
    tool_results: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

class PersistentAgent:
    def __init__(self, state_store: StateStore):
        self.store = state_store

    async def run(self, task_id: str) -> str:
        # Resume from saved state or create new
        state = await self.store.load(task_id) or AgentState(task_id=task_id, ...)

        try:
            result = await self._execute(state)
            state.status = "completed"
            await self.store.save(state)
            return result
        except Exception as e:
            state.status = "failed"
            state.metadata["error"] = str(e)
            await self.store.save(state)
            raise
```

### Checkpointing

Save state after each tool execution so work is not lost on failure.

```python
async def agent_loop_with_checkpoints(state: AgentState) -> str:
    for i in range(state.max_iterations):
        response = await call_llm(state.messages, state.tools)
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            return extract_text(response)

        for tool_call in tool_calls:
            result = await execute_tool(tool_call)
            state.messages.append(format_tool_result(tool_call, result))
            state.completed_steps.append(tool_call["name"])

            # Checkpoint after each tool execution
            await state_store.save(state)

    return "Max iterations reached."
```

### Handling Context Window Limits

When conversation history grows too large, compress it:

```python
async def manage_context(state: AgentState, max_tokens: int = 100_000) -> None:
    current_tokens = count_tokens(state.messages)

    if current_tokens > max_tokens * 0.8:  # 80% threshold
        # Summarize older messages
        split_point = len(state.messages) // 2
        old_messages = state.messages[:split_point]
        recent_messages = state.messages[split_point:]

        summary = await summarize_messages(old_messages)

        state.messages = [
            {"role": "system", "content": f"Previous work summary:\n{summary}"},
            *recent_messages
        ]
```

### Idempotent Tool Calls for Resumability

When resuming after a crash, the agent may re-execute tool calls. Idempotent tools prevent duplicate side effects.

```python
class IdempotentToolExecutor:
    def __init__(self, result_cache: dict[str, Any]):
        self.cache = result_cache

    async def execute(self, tool_call: dict) -> Any:
        # Create a deterministic key from the call
        cache_key = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = await self._execute_raw(tool_call)
        self.cache[cache_key] = result
        return result
```

---

## 3. Human-in-the-Loop (HITL)

### Approval Workflows

```
Agent decides to call tool
         |
         v
  Is tool in REQUIRES_APPROVAL set?
         |
    +----+----+
    |         |
    No        Yes
    |         |
    v         v
  Execute   Present to human:
  directly    "Agent wants to issue_refund($150).
              Approve? [Yes/No]"
              |
         +----+----+
         |         |
       Approve   Reject
         |         |
         v         v
       Execute   Return rejection
       tool      to agent as
                 tool result
```

### Confidence-Based Routing

Some systems let the model express confidence, routing low-confidence decisions to humans.

```python
async def confidence_routing(agent_response: dict) -> dict:
    confidence = agent_response.get("confidence", 0.5)

    if confidence >= 0.9:
        return await auto_execute(agent_response)
    elif confidence >= 0.6:
        return await execute_with_logging(agent_response)  # Auto but flagged for review
    else:
        return await request_human_review(agent_response)
```

**Caveat:** LLM-reported confidence is not well-calibrated. Use this as a heuristic, not a guarantee.

### Escalation Patterns

```python
class EscalationPolicy:
    def __init__(self):
        self.failure_count = 0
        self.max_failures = 3

    async def handle_result(self, result: dict) -> str:
        if result.get("error"):
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                return await self.escalate_to_human(result)
            return "retry"

        self.failure_count = 0  # Reset on success
        return "continue"

    async def escalate_to_human(self, context: dict) -> str:
        # Create a support ticket, send a Slack message, etc.
        ticket = await create_support_ticket(
            title="Agent escalation — unable to resolve",
            context=context,
            priority="high",
        )
        return f"I've escalated this to a human agent. Ticket: {ticket.id}"
```

---

## 4. Agent Evaluation

### What to Measure

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Task completion rate** | Did the agent successfully complete the task? | Binary per task, aggregate as percentage |
| **Tool call efficiency** | How many tool calls to complete vs. minimum needed? | Ratio: actual calls / optimal calls |
| **Error recovery rate** | When a tool call fails, does the agent recover? | Track recovery after errors |
| **Cost per task** | Total token cost (input + output) per completed task | Sum API costs across all iterations |
| **Latency** | Wall-clock time from request to final response | End-to-end timer |
| **Safety violations** | Did the agent attempt unauthorized actions? | Count blocked/rejected tool calls |
| **User satisfaction** | Did the end user rate the interaction positively? | Feedback scores, thumbs up/down |

### Evaluation Approaches

#### Trajectory Evaluation

Evaluate the entire sequence of tool calls, not just the final answer.

```python
def evaluate_trajectory(
    trajectory: list[dict],      # Actual tool calls made
    reference: list[dict],       # Expected optimal tool calls
) -> dict:
    return {
        "correct_tools_used": tool_overlap(trajectory, reference),
        "unnecessary_calls": len(trajectory) - len(reference),
        "correct_order": is_order_preserved(trajectory, reference),
        "final_answer_correct": check_final_answer(trajectory[-1], reference[-1]),
    }
```

#### Benchmark Suites

- **SWE-bench:** Coding agents resolve real GitHub issues.
- **WebArena:** Browser agents complete web tasks.
- **GAIA:** General AI assistant tasks requiring multi-step tool use.
- **ToolBench:** Tests tool selection across hundreds of APIs.
- **AgentBench:** Multi-domain benchmark covering code, web, games, and databases.

#### A/B Testing in Production

Run two agent configurations side by side on real traffic. Compare completion rates, costs, and user satisfaction. This is the gold standard for evaluation but requires production traffic.

### Cost Analysis

```python
def analyze_agent_cost(runs: list[AgentRun]) -> dict:
    return {
        "avg_iterations": mean(r.iterations for r in runs),
        "avg_input_tokens": mean(r.input_tokens for r in runs),
        "avg_output_tokens": mean(r.output_tokens for r in runs),
        "avg_cost_usd": mean(r.total_cost for r in runs),
        "p95_cost_usd": percentile(95, [r.total_cost for r in runs]),
        "cost_per_successful_task": (
            sum(r.total_cost for r in runs) /
            sum(1 for r in runs if r.success)
        ),
    }
```

---

## 5. Code Execution Agents

### Sandboxing Options

| Option | Isolation Level | Startup Time | Use Case |
|--------|----------------|-------------|----------|
| Docker | Process-level | ~1s | General code execution |
| E2B | VM-level | ~300ms | AI-specific sandboxes |
| gVisor | Kernel-level | <100ms | High-security environments |
| Firecracker | MicroVM | ~125ms | AWS Lambda-style |
| WASM | Language-level | <10ms | Lightweight, browser-compatible |
| subprocess | None (dangerous) | Instant | Development only |

### REPL Pattern

The most effective code execution agents use an iterative REPL (Read-Eval-Print Loop):

```
1. Agent writes code
2. Code executes in sandbox
3. Output (stdout, stderr, return value) goes back to agent
4. Agent analyzes output:
   - Success? Use the result and continue
   - Error? Read the traceback, fix the code, retry
   - Unexpected output? Adjust approach
5. Repeat until task is complete
```

```python
class CodeExecutionAgent:
    async def solve(self, task: str) -> str:
        messages = [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for _ in range(self.max_iterations):
            response = await call_llm(messages, tools=[self.execute_code_tool])
            tool_calls = extract_tool_calls(response)

            if not tool_calls:
                return extract_text(response)

            for tc in tool_calls:
                code = tc["arguments"]["code"]
                result = await self.sandbox.execute(code, timeout=30)

                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_value": result.return_value,
                        "execution_time": result.duration_ms,
                    })
                })

        return "Max iterations reached."
```

### Security Considerations

- **Network access:** Disable by default. Enable only for specific tasks.
- **File system:** Mount only necessary directories, read-only when possible.
- **Resource limits:** CPU time, memory, disk space.
- **Secrets:** Never expose API keys or credentials to the sandbox.
- **Persistent state:** Clean sandbox between executions unless explicitly needed.

---

## 6. Retrieval-Augmented Agents

### Agent-Controlled Retrieval

Unlike static RAG (always retrieve, then generate), retrieval-augmented agents decide *when* and *what* to retrieve.

```
User: "What's the return policy for electronics over $500?"
         |
         v
Agent thinks: I need to check the return policy docs.
              This is specifically about high-value electronics.
         |
         v
Tool call: search_knowledge_base(
    query="return policy electronics over $500",
    filters={"category": "policies"}
)
         |
         v
Results: [policy_doc_1, policy_doc_2]
         |
         v
Agent thinks: The policy says 30 days for electronics over $500
              with original packaging. I have enough to answer.
         |
         v
Response: "For electronics over $500, our return policy is..."
```

### Multi-Step Retrieval

The agent might need multiple searches to answer a question:

```python
# Agent recognizes it needs more context
# Step 1: Search for the general policy
search_kb(query="electronics return policy")  # General results

# Step 2: Refine search based on initial results
search_kb(query="extended warranty exceptions high-value items")  # Specific refinement

# Step 3: Check for recent policy updates
search_kb(query="return policy changes 2025", filters={"date_after": "2025-01-01"})
```

### Retrieval Tool Design

```python
search_tool = {
    "name": "search_knowledge_base",
    "description": (
        "Search the knowledge base for relevant documents. Returns the top-k "
        "most relevant results with titles, snippets, and relevance scores. "
        "Use specific, descriptive queries for best results. You can search "
        "multiple times with different queries to find comprehensive information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query. Be specific."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 20)",
                "default": 5,
                "maximum": 20
            },
            "filters": {
                "type": "object",
                "description": "Optional filters to narrow results",
                "properties": {
                    "category": {"type": "string", "enum": ["policies", "products", "faq", "technical"]},
                    "date_after": {"type": "string", "format": "date"}
                }
            }
        },
        "required": ["query"]
    }
}
```

---

## 7. Agent Frameworks Comparison

### Overview

| Framework | Approach | Strengths | Weaknesses |
|-----------|----------|-----------|------------|
| **LangChain / LangGraph** | Graph-based agent orchestration | Flexible, large ecosystem, good for complex flows | Steep learning curve, heavy abstraction |
| **CrewAI** | Role-based multi-agent | Easy multi-agent setup, good metaphors | Less flexible than LangGraph |
| **AutoGen (Microsoft)** | Conversational multi-agent | Strong for agent-to-agent communication | Complex configuration |
| **Claude Code (Anthropic)** | Single-agent with tool use | Excellent coding agent, minimal framework | Claude-only |
| **OpenAI Assistants API** | Managed agent service | Built-in file search, code interpreter, threads | OpenAI-only, less control |
| **Smolagents (HuggingFace)** | Lightweight code agents | Simple, open-source, code-first | Fewer features |
| **Pydantic AI** | Type-safe agent framework | Strong typing, dependency injection | Newer, smaller ecosystem |

### Build vs. Buy Decision

```
Do you need multi-agent orchestration?
  |
  +-- No --> Is your use case simple (< 5 tools, linear flow)?
  |            |
  |            +-- Yes --> Build your own loop (50-100 lines of code)
  |            +-- No  --> LangGraph or Pydantic AI for complex single-agent
  |
  +-- Yes --> Do agents need to communicate with each other?
               |
               +-- No  --> Orchestrator pattern with your own code
               +-- Yes --> CrewAI (simple), AutoGen (complex), LangGraph (flexible)
```

### The "Just Write a Loop" Argument

For many production systems, a hand-written agent loop of 50-100 lines is preferable to a framework.

**Advantages of no framework:**
- Full control over execution flow
- No dependency on framework release cycles
- Easy to debug (it is just your code)
- No abstraction leaks
- Simpler testing

**When a framework helps:**
- Multi-agent coordination with complex message passing
- You need persistence, streaming, and observability out of the box
- Your team is building many different agents and wants shared patterns
- You want a visual graph editor (LangGraph Studio)

### LangGraph Example Structure

```python
# LangGraph defines agents as nodes in a directed graph
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Define nodes (each is a function that processes state)
graph.add_node("classifier", classify_intent)
graph.add_node("researcher", research_agent)
graph.add_node("responder", generate_response)

# Define edges (control flow)
graph.add_edge("classifier", "researcher")
graph.add_edge("researcher", "responder")
graph.add_edge("responder", END)

# Conditional routing
graph.add_conditional_edges(
    "classifier",
    route_by_intent,  # Function that returns next node name
    {"research": "researcher", "simple": "responder"}
)

app = graph.compile()
result = await app.ainvoke({"messages": [user_message]})
```

---

## 8. Error Handling and Recovery

### Error Taxonomy

| Error Type | Source | Strategy |
|-----------|--------|----------|
| **Malformed tool call** | Model outputs invalid JSON or missing params | Return validation error, model retries |
| **Tool execution failure** | API timeout, service down | Retry with backoff, fallback tool |
| **Wrong tool selected** | Model chose inappropriate tool | Error result guides model to correct tool |
| **Infinite loop** | Model repeats same action | Loop detection, max iterations |
| **Context overflow** | Too many messages | Summarize, sliding window |
| **Rate limiting** | Too many API calls | Exponential backoff, queue |
| **Permission denied** | Tool requires higher auth | Return error, suggest alternative |

### Retry Strategies

```python
async def execute_with_retry(
    tool_call: dict,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> dict:
    last_error = None

    for attempt in range(max_retries):
        try:
            return await execute_tool(tool_call)
        except RateLimitError:
            wait = backoff_base * (2 ** attempt)
            await asyncio.sleep(wait)
        except TimeoutError:
            last_error = "Tool execution timed out"
        except ToolNotFoundError as e:
            # Don't retry -- tool genuinely doesn't exist
            return {"error": str(e), "available_tools": list_tools()}
        except Exception as e:
            last_error = str(e)

    return {"error": f"Tool failed after {max_retries} attempts: {last_error}"}
```

### Loop Detection

```python
class LoopDetector:
    def __init__(self, max_repeats: int = 3):
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats

    def check(self, tool_call: dict) -> bool:
        """Returns True if a loop is detected."""
        call_signature = f"{tool_call['name']}:{json.dumps(tool_call['arguments'], sort_keys=True)}"
        self.recent_calls.append(call_signature)

        # Check if the same call appears max_repeats times in the last N calls
        if self.recent_calls.count(call_signature) >= self.max_repeats:
            return True

        return False
```

### Error Messages as Context

The error message returned to the model is a prompt. Make it informative.

```python
# Tier 1: What went wrong
# Tier 2: Why it went wrong
# Tier 3: What to try instead

def format_tool_error(error: Exception, tool_call: dict) -> dict:
    if isinstance(error, NotFoundError):
        return {
            "error": f"No results found for {tool_call['arguments']}",
            "reason": "The search returned zero matches",
            "suggestions": [
                "Try broader search terms",
                "Check for typos in the query",
                "Try searching by a different field (e.g., name instead of email)"
            ]
        }
    elif isinstance(error, PermissionError):
        return {
            "error": f"Permission denied for {tool_call['name']}",
            "reason": "Current user does not have the required access level",
            "suggestions": [
                "Use a read-only alternative tool",
                "Ask the user to confirm they want to proceed (escalation needed)"
            ]
        }
    else:
        return {
            "error": str(error),
            "reason": "Unexpected error during tool execution",
            "suggestions": ["Try a different approach to accomplish the task"]
        }
```

---

## 9. Computer Use / Browser Agents

### How It Works

Vision-capable models can interpret screenshots and generate mouse/keyboard actions. This enables agents to interact with arbitrary user interfaces without APIs.

```
1. Take screenshot of the screen/browser
2. Send screenshot to vision-capable LLM
3. Model analyzes the UI and decides what to do
4. Model outputs an action: click(x, y), type("text"), scroll, etc.
5. Execute the action in the environment
6. Take new screenshot
7. Repeat
```

### Tool Definition for Computer Use

```python
computer_use_tools = [
    {
        "name": "screenshot",
        "description": "Take a screenshot of the current screen state"
    },
    {
        "name": "click",
        "description": "Click at specific coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "type_text",
        "description": "Type text at the current cursor position",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "key_press",
        "description": "Press a keyboard shortcut",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {"type": "string", "description": "e.g., 'ctrl+c', 'enter', 'tab'"}
            },
            "required": ["keys"]
        }
    }
]
```

### Current State and Limitations

**What works today:**
- Form filling and data entry
- Navigation through known workflows
- Simple web research (search, read, extract)
- Testing UI flows

**Current limitations:**
- **Latency:** Each step requires a screenshot + LLM call (~2-5 seconds per action).
- **Accuracy:** Coordinate-based clicking is fragile. Small UI changes break flows.
- **Cost:** Vision models are expensive. A 30-step browser task can cost several dollars.
- **Security:** Giving an agent full computer access is a significant trust boundary.
- **Dynamic content:** Animations, loading spinners, and pop-ups confuse the model.

### Anthropic's Computer Use

Anthropic provides a first-party computer use capability with Claude:

```python
# Anthropic computer use tools
tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
    },
    {
        "type": "text_editor_20241022",
        "name": "str_replace_editor",
    },
    {
        "type": "bash_20241022",
        "name": "bash",
    }
]
```

The model outputs structured actions (`screenshot`, `click`, `type`, `key`, `scroll`) rather than raw coordinates, which improves reliability.

### Interview Perspective

Computer use represents the frontier of agent capability. The key insight: it converts any GUI application into a tool, without requiring an API. But it is currently slow, expensive, and fragile compared to API-based tool use. Use it when no API exists and the value justifies the cost.

---

## Interview Questions — Deep Dive Level

### Multi-Agent

1. **"When would you use multi-agent vs. single-agent architecture?"**
   Single agent for tasks with <15 tools and straightforward control flow. Multi-agent when you need specialized expertise (different system prompts, different tool sets), when a single context window cannot hold all the required tools and instructions, or when you want independent scaling and iteration on each agent.

2. **"How do agents communicate in a multi-agent system?"**
   Message passing (each agent's output becomes input for the next), shared state (a common data store all agents read/write), or orchestrator mediation (a coordinator routes messages between agents). The orchestrator pattern is most common because it is simplest to debug.

### Long-Running Agents

3. **"How do you handle a task that takes 100+ tool calls?"**
   Checkpoint state after each tool call. Implement context window management (summarization of older messages). Use idempotent tools for crash recovery. Set cost budgets. Consider breaking the task into sub-tasks that each fit within reasonable limits.

### Evaluation

4. **"How do you evaluate an agent in production?"**
   Task completion rate (primary metric), tool call efficiency (actual vs. optimal), cost per task, latency, error recovery rate, and safety violations. Use trajectory evaluation during development and A/B testing in production.

### Frameworks

5. **"Should I use LangChain or build my own agent loop?"**
   For simple agents (1 agent, <10 tools), build your own -- it is 50-100 lines of code with full control. For multi-agent systems with complex routing, persistence, and streaming, a framework like LangGraph saves significant effort. The key is avoiding premature abstraction.
