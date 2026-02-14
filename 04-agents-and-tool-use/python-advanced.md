# Python Advanced: Agents & Tool Use

> **Advanced Python patterns for building reliable agent systems.** This guide covers async orchestration, error handling, retry logic, type safety, and state management.

---

## Advanced Async Patterns

### Concurrent Tool Execution

```python
import asyncio
from typing import Any

async def execute_tool(name: str, args: dict) -> dict[str, Any]:
    """Simulate async tool execution."""
    await asyncio.sleep(0.5)  # Simulate work
    return {"tool": name, "result": f"Result for {args}"}

async def execute_tools_parallel(
    tool_calls: list[tuple[str, dict]]
) -> list[dict[str, Any]]:
    """
    Execute multiple tools concurrently.

    PYTHON: Create tasks, then gather results
    Much faster than sequential execution!
    """
    # Create all tasks
    tasks = [
        execute_tool(name, args)
        for name, args in tool_calls
    ]

    # Run concurrently, wait for all
    results = await asyncio.gather(*tasks)

    return results

# Usage
tool_calls = [
    ("calculator", {"expr": "2+2"}),
    ("search", {"query": "Python"}),
    ("weather", {"city": "SF"})
]

results = await execute_tools_parallel(tool_calls)
# All 3 tools execute in parallel - takes ~0.5s instead of ~1.5s
```

### Timeout Handling

```python
async def call_with_timeout(
    coro,  # Coroutine to execute
    timeout_seconds: float
) -> Any:
    """
    Execute async function with timeout.

    PYTHON: asyncio.wait_for() raises TimeoutError if too slow
    Essential for preventing hung agent loops!
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

# Usage
async def slow_llm_call():
    await asyncio.sleep(10)  # Simulate slow API
    return "response"

try:
    # Will timeout after 5 seconds
    result = await call_with_timeout(slow_llm_call(), timeout_seconds=5.0)
except TimeoutError as e:
    print(f"LLM call failed: {e}")
    # Fall back to cached response or error handling
```

### Racing Multiple Providers

```python
async def call_first_to_respond(
    *coroutines
) -> tuple[Any, int]:
    """
    Race multiple async calls, return first to complete.

    PYTHON: asyncio.wait() with FIRST_COMPLETED
    Useful for: trying multiple LLM providers, redundant calls for reliability
    """
    tasks = [asyncio.create_task(coro) for coro in coroutines]

    # Wait for first to complete
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel remaining tasks
    for task in pending:
        task.cancel()

    # Get result from first completed task
    result = list(done)[0].result()
    winner_index = tasks.index(list(done)[0])

    return result, winner_index

# Usage
async def call_openai():
    await asyncio.sleep(0.8)
    return "OpenAI response"

async def call_anthropic():
    await asyncio.sleep(0.5)  # Faster!
    return "Anthropic response"

result, winner = await call_first_to_respond(call_openai(), call_anthropic())
# Returns: ("Anthropic response", 1) after 0.5s
# OpenAI call is cancelled
```

---

## Error Handling & Retries

### Retry Decorator with Exponential Backoff

```python
import asyncio
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar('T')  # Generic type

def async_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for async functions with retry logic.

    PYTHON CONCEPTS:
    - Decorators that take parameters (function that returns decorator)
    - TypeVar for generic return types
    - *args/**kwargs to forward all arguments
    - Exponential backoff: wait 1s, 2s, 4s, ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)  # Preserves function name and docstring
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        # Calculate backoff delay
                        delay = backoff_base ** attempt
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed")

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator

# Usage
@async_retry(max_attempts=3, backoff_base=2.0, exceptions=(TimeoutError, ConnectionError))
async def flaky_api_call(prompt: str) -> str:
    """This function will auto-retry on failures."""
    # Simulate occasional failures
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("API unavailable")

    return f"Response to: {prompt}"

# Call it normally - retries happen automatically
response = await flaky_api_call("Hello")
```

### Error Context and Logging

```python
from dataclasses import dataclass
from typing import Any
from enum import Enum

class ErrorSeverity(Enum):
    """Enum for error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AgentError:
    """Structured error for agent systems."""
    message: str
    severity: ErrorSeverity
    context: dict[str, Any]
    recoverable: bool
    suggested_action: str | None = None

def handle_agent_error(error: Exception, context: dict[str, Any]) -> AgentError:
    """
    Convert exceptions to structured errors with context.

    PYTHON: Match statement (3.10+) for pattern matching
    Similar to switch/case but more powerful
    """
    match error:
        case TimeoutError():
            return AgentError(
                message=f"Operation timed out: {error}",
                severity=ErrorSeverity.WARNING,
                context=context,
                recoverable=True,
                suggested_action="Retry with longer timeout or simpler task"
            )

        case ValueError() | TypeError():
            return AgentError(
                message=f"Invalid input: {error}",
                severity=ErrorSeverity.ERROR,
                context=context,
                recoverable=False,
                suggested_action="Check input validation"
            )

        case ConnectionError() | asyncio.TimeoutError():
            return AgentError(
                message=f"Network error: {error}",
                severity=ErrorSeverity.ERROR,
                context=context,
                recoverable=True,
                suggested_action="Retry or use fallback provider"
            )

        case _:  # Default case
            return AgentError(
                message=f"Unexpected error: {error}",
                severity=ErrorSeverity.CRITICAL,
                context=context,
                recoverable=False,
                suggested_action="Log and alert engineering team"
            )

# Usage
try:
    result = await risky_operation()
except Exception as e:
    agent_error = handle_agent_error(e, {"step": "tool_execution", "tool": "calculator"})

    if agent_error.recoverable:
        # Attempt recovery
        result = await fallback_operation()
    else:
        # Cannot recover, propagate error
        raise
```

---

## Type Safety with Generics

### Generic Agent Response

```python
from typing import TypeVar, Generic

T = TypeVar('T')  # Type variable

@dataclass
class AgentResult(Generic[T]):
    """
    Generic agent result that preserves the type of the output.

    PYTHON: Generic[T] makes this a generic class
    Can be AgentResult[str], AgentResult[dict], etc.
    """
    success: bool
    output: T | None
    error: AgentError | None
    steps_taken: int
    tokens_used: int

    @property
    def is_success(self) -> bool:
        """Type-safe success check."""
        return self.success and self.output is not None

# Usage with type hints
async def run_agent_text() -> AgentResult[str]:
    """Returns AgentResult[str] - output is guaranteed to be str if success"""
    try:
        output = await generate_text()
        return AgentResult(
            success=True,
            output=output,  # Type checker knows this is str
            error=None,
            steps_taken=3,
            tokens_used=150
        )
    except Exception as e:
        return AgentResult(
            success=False,
            output=None,
            error=handle_agent_error(e, {}),
            steps_taken=0,
            tokens_used=0
        )

async def run_agent_structured() -> AgentResult[dict[str, Any]]:
    """Returns AgentResult[dict] - output is guaranteed to be dict if success"""
    # Implementation...
    pass

# Type-safe usage
result: AgentResult[str] = await run_agent_text()
if result.is_success:
    text: str = result.output  # Type checker knows this is str
    print(text.upper())  # Can call string methods
```

### Protocol-based Tool Interface

```python
from typing import Protocol, Any

class Tool(Protocol):
    """
    Protocol defining tool interface.

    PYTHON: Any class with these methods can be used as a Tool
    No inheritance needed - structural typing!
    """
    name: str
    description: str

    async def execute(self, **kwargs) -> Any:
        """Execute tool with given arguments."""
        ...

    def get_schema(self) -> dict[str, Any]:
        """Return JSON schema for tool parameters."""
        ...

# Implementations don't need to inherit from Tool
class CalculatorTool:
    """Automatically satisfies Tool protocol."""
    name = "calculator"
    description = "Perform arithmetic calculations"

    async def execute(self, expression: str) -> float:
        # Safe eval would go here
        return eval(expression)  # Simplified!

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }

# Type-safe tool execution
async def execute_tool_safely(tool: Tool, args: dict) -> Any:
    """
    This function accepts any object matching the Tool protocol.
    Type checker verifies tool has required methods.
    """
    schema = tool.get_schema()
    # Validate args against schema...
    result = await tool.execute(**args)
    return result

# Works with any Tool implementation
calc = CalculatorTool()
result = await execute_tool_safely(calc, {"expression": "2+2"})
```

---

## State Management

### Agent State with History

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentStep:
    """Single step in agent execution."""
    step_number: int
    action: str
    tool_name: str | None
    tool_input: dict[str, Any] | None
    tool_output: Any | None
    reasoning: str
    timestamp: float

@dataclass
class AgentState:
    """
    Tracks agent state across execution.

    PYTHON: field(default_factory=list) for mutable defaults
    NEVER use mutable defaults like [] or {} directly!
    """
    status: AgentStatus = AgentStatus.IDLE
    steps: list[AgentStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    error: AgentError | None = None

    def add_step(self, step: AgentStep) -> None:
        """Add execution step to history."""
        self.steps.append(step)
        self.status = AgentStatus.EXECUTING

    def mark_completed(self) -> None:
        """Mark agent as successfully completed."""
        self.status = AgentStatus.COMPLETED

    def mark_failed(self, error: AgentError) -> None:
        """Mark agent as failed with error."""
        self.status = AgentStatus.FAILED
        self.error = error

    @property
    def step_count(self) -> int:
        """Number of steps executed."""
        return len(self.steps)

    def get_recent_steps(self, n: int = 5) -> list[AgentStep]:
        """Get last n steps (for context window management)."""
        return self.steps[-n:]

# Usage
state = AgentState()

# Add steps as agent executes
step1 = AgentStep(
    step_number=1,
    action="search",
    tool_name="web_search",
    tool_input={"query": "Python tutorials"},
    tool_output={"results": [...]},
    reasoning="Need to find learning resources",
    timestamp=time.time()
)
state.add_step(step1)

# Check state
if state.status == AgentStatus.EXECUTING:
    print(f"Agent has executed {state.step_count} steps")
```

### Context Manager for State

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator

@asynccontextmanager
async def agent_execution(task: str) -> AsyncIterator[AgentState]:
    """
    Context manager for agent execution with automatic cleanup.

    PYTHON CONCEPTS:
    - @asynccontextmanager decorator
    - try/finally for guaranteed cleanup
    - yield to provide resource
    - AsyncIterator type hint
    """
    state = AgentState()
    state.status = AgentStatus.THINKING
    state.context["task"] = task

    print(f"Starting agent execution: {task}")

    try:
        yield state  # Provide state to code block

    except Exception as e:
        # Handle errors automatically
        error = handle_agent_error(e, state.context)
        state.mark_failed(error)
        raise

    finally:
        # Always runs, even if exception occurred
        if state.status != AgentStatus.FAILED:
            state.mark_completed()

        print(f"Agent finished. Status: {state.status}, Steps: {state.step_count}")

# Usage
async with agent_execution("Analyze data") as state:
    # State is automatically initialized
    # Errors are automatically handled
    # Cleanup happens automatically

    step = AgentStep(...)
    state.add_step(step)

    # Even if exception occurs, finally block runs
    await do_work()

# State is finalized here
```

---

## Advanced Decorators

### Tool Registration Decorator

```python
from typing import Callable, Any

# Global tool registry
_TOOL_REGISTRY: dict[str, Callable] = {}

def tool(
    name: str,
    description: str,
    parameters: dict[str, Any]
):
    """
    Decorator to register functions as tools.

    PYTHON: Decorator factory pattern
    Captures metadata and stores function in registry
    """
    def decorator(func: Callable) -> Callable:
        # Store function with metadata
        _TOOL_REGISTRY[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }

        # Add metadata to function object
        func._tool_name = name
        func._tool_description = description
        func._tool_parameters = parameters

        return func

    return decorator

# Usage - declare tools with decorator
@tool(
    name="calculator",
    description="Perform arithmetic calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    }
)
async def calculator(expression: str) -> float:
    """Tool function with automatic registration."""
    return eval(expression)  # Simplified

@tool(
    name="search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
)
async def search(query: str) -> list[str]:
    """Another tool."""
    return [f"Result for {query}"]

# Get all registered tools
def get_available_tools() -> list[dict]:
    """Return list of all registered tools with metadata."""
    return [
        {
            "name": name,
            "description": tool_data["description"],
            "parameters": tool_data["parameters"]
        }
        for name, tool_data in _TOOL_REGISTRY.items()
    ]

# Execute tool by name
async def execute_tool_by_name(name: str, **kwargs) -> Any:
    """Dynamically execute tool by name."""
    if name not in _TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    tool_func = _TOOL_REGISTRY[name]["function"]
    return await tool_func(**kwargs)

# Usage
result = await execute_tool_by_name("calculator", expression="10 + 5")
# Returns: 15.0
```

---

## Async Iterators for Streaming

### Streaming Agent Thoughts

```python
from typing import AsyncIterator
from enum import Enum

class ThoughtType(Enum):
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"

@dataclass
class AgentThought:
    """Single thought in agent's reasoning chain."""
    type: ThoughtType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

async def stream_agent_execution(task: str) -> AsyncIterator[AgentThought]:
    """
    Stream agent's thoughts as they happen.

    PYTHON: async def + yield = async generator
    Allows real-time streaming to UI
    """
    # Initial reasoning
    yield AgentThought(
        type=ThoughtType.REASONING,
        content=f"I need to solve: {task}",
        metadata={"step": 1}
    )

    await asyncio.sleep(0.5)  # Simulate thinking

    # Tool selection
    yield AgentThought(
        type=ThoughtType.REASONING,
        content="I'll use the calculator tool",
        metadata={"step": 2}
    )

    # Tool execution
    yield AgentThought(
        type=ThoughtType.TOOL_CALL,
        content="calculator(expression='10+5')",
        metadata={"step": 3, "tool": "calculator"}
    )

    await asyncio.sleep(0.3)  # Simulate tool execution

    # Tool result
    yield AgentThought(
        type=ThoughtType.TOOL_RESULT,
        content="Result: 15",
        metadata={"step": 4, "tool": "calculator"}
    )

    # Final answer
    yield AgentThought(
        type=ThoughtType.ANSWER,
        content="The answer is 15",
        metadata={"step": 5, "final": True}
    )

# Usage - stream thoughts to console/UI
async for thought in stream_agent_execution("What is 10+5?"):
    print(f"[{thought.type.value}] {thought.content}")

# Output:
# [reasoning] I need to solve: What is 10+5?
# [reasoning] I'll use the calculator tool
# [tool_call] calculator(expression='10+5')
# [tool_result] Result: 15
# [answer] The answer is 15
```

---

## Next Steps

You now have advanced Python skills for building agent systems! Continue to:

1. Study [concepts.md](concepts.md) and [patterns.md](patterns.md) for agent architecture
2. Read [examples.py](examples.py) to see these patterns in production code
3. Move on to `05-production-concerns/` for deployment best practices

**Key takeaways:**
- Use async for concurrent tool execution
- Implement retries with exponential backoff
- Use generics for type-safe results
- Track state with dataclasses
- Stream results with async generators
- Register tools with decorators

---

## Quick Reference

| Task | Python Pattern |
|---|---|
| Concurrent execution | `await asyncio.gather(*tasks)` |
| Timeout | `await asyncio.wait_for(coro, timeout=5.0)` |
| Race condition | `asyncio.wait(..., return_when=FIRST_COMPLETED)` |
| Retry decorator | Custom decorator with exponential backoff |
| Error patterns | `match error: case TimeoutError(): ...` |
| Generic types | `class Result(Generic[T]):` |
| Protocol | `class Tool(Protocol): ...` |
| Mutable defaults | `field(default_factory=list)` |
| Context manager | `@asynccontextmanager async def f():` |
| Tool registration | Decorator with global registry |
| Async generator | `async def f(): yield x` |

**Next:** [examples.py](examples.py) and [patterns.md](patterns.md)
