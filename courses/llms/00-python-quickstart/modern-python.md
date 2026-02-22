# Modern Python (3.10+)

> **This guide covers modern Python features (3.10+) used throughout the LLM examples.** These patterns will feel natural coming from TypeScript.

---

## Type Hints

Python 3.10+ supports clean, TypeScript-like type syntax.

### Basic Types

```python
# Pre-3.10 (still valid)
from typing import List, Dict, Optional, Union

def process(items: List[str]) -> Dict[str, int]:
    pass

# Modern Python 3.10+ (preferred)
def process(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Built-in types can be used directly
name: str = "Claude"
age: int = 2
temperature: float = 0.7
is_ready: bool = True
```

### Optional and Union Types

```python
# Optional (can be None)
from typing import Optional

# Old way
def get_config(key: str) -> Optional[str]:
    return config.get(key)

# Modern way (Python 3.10+)
def get_config(key: str) -> str | None:
    return config.get(key)

# Union types
def format_output(value: str | int | float) -> str:
    return f"Value: {value}"

result = format_output("text")   # OK
result = format_output(42)       # OK
result = format_output(3.14)     # OK
```

### Collection Types

```python
# Lists and dicts with type hints
messages: list[str] = ["Hello", "World"]
config: dict[str, int] = {"max_tokens": 1024, "timeout": 30}

# Nested types
model_configs: dict[str, dict[str, float]] = {
    "claude": {"temperature": 0.7, "top_p": 0.9},
    "gpt4": {"temperature": 0.8, "top_p": 1.0}
}

# List of dicts (common for messages)
messages: list[dict[str, str]] = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
```

### Literal Types (Constrained Values)

```python
from typing import Literal

# Only allow specific string values (like TS union types)
Role = Literal["system", "user", "assistant"]

def create_message(role: Role, content: str) -> dict[str, str]:
    return {"role": role, "content": content}

create_message("user", "Hi")      # OK
create_message("admin", "Hi")     # Type error!

# Real LLM example
ModelName = Literal["gpt-4", "claude-sonnet-4", "claude-opus-4"]

def call_llm(model: ModelName, prompt: str) -> str:
    pass
```

### Type Aliases

```python
# Create reusable type aliases
Message = dict[str, str]
MessageList = list[Message]

def format_conversation(messages: MessageList) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

# More complex alias
LLMConfig = dict[str, str | int | float]

config: LLMConfig = {
    "model": "claude",
    "max_tokens": 1024,
    "temperature": 0.7
}
```

---

## Dataclasses

**Dataclasses eliminate boilerplate for simple classes.** Think of them as TypeScript interfaces + implementation.

### Basic Dataclass

```python
from dataclasses import dataclass

# Without dataclass (verbose)
class MessageOld:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

# With dataclass (concise!)
@dataclass
class Message:
    role: str
    content: str

# Auto-generates __init__, __repr__, __eq__
msg = Message(role="user", content="Hello")
print(msg)  # Message(role='user', content='Hello')
print(msg.role)  # "user"
```

### Default Values

```python
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

# Use defaults
config = LLMConfig(model="claude")
# LLMConfig(model='claude', temperature=0.7, max_tokens=1024, stream=False)

# Override defaults
config = LLMConfig(model="gpt-4", temperature=0.9, stream=True)
```

### Nested Dataclasses

```python
@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatRequest:
    messages: list[Message]
    model: str
    temperature: float = 0.7

# Usage
request = ChatRequest(
    messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ],
    model="claude"
)
```

### Dataclass Methods

```python
@dataclass
class Message:
    role: str
    content: str

    def is_user(self) -> bool:
        return self.role == "user"

    def token_count(self) -> int:
        # Rough estimate: 1 token ≈ 4 chars
        return len(self.content) // 4

msg = Message(role="user", content="Hello world!")
print(msg.is_user())      # True
print(msg.token_count())  # 3
```

### Converting to Dict

```python
from dataclasses import dataclass, asdict

@dataclass
class LLMConfig:
    model: str
    temperature: float

config = LLMConfig(model="claude", temperature=0.7)

# Convert to dict (for JSON serialization)
config_dict = asdict(config)
# {"model": "claude", "temperature": 0.7}

# Use in API calls
import json
json.dumps(asdict(config))
```

---

## Protocols (Structural Subtyping)

**Protocols = TypeScript interfaces.** They define structure without requiring inheritance.

### Basic Protocol

```python
from typing import Protocol

# Define the interface
class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...

# Any class matching this structure is compatible
class OpenAIProvider:
    def generate(self, prompt: str) -> str:
        return f"OpenAI response to: {prompt}"

class AnthropicProvider:
    def generate(self, prompt: str) -> str:
        return f"Anthropic response to: {prompt}"

# Generic function accepting any provider
def call_llm(provider: LLMProvider, prompt: str) -> str:
    return provider.generate(prompt)

# Both work without explicit inheritance!
result = call_llm(OpenAIProvider(), "Hello")
result = call_llm(AnthropicProvider(), "Hello")
```

### Protocol with Properties

```python
from typing import Protocol

class EmbeddingModel(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...

class OpenAIEmbedding:
    dimensions = 1536

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimensions

def process_embedding(model: EmbeddingModel, text: str) -> int:
    embedding = model.embed(text)
    return len(embedding)

model = OpenAIEmbedding()
size = process_embedding(model, "test")  # 1536
```

**Why use Protocols?** Your code examples work with any LLM provider without tight coupling.

---

## Async/Await

**Essential for LLM applications.** API calls should never block.

### Basic Async Function

```python
import asyncio

# Async function (like TS async)
async def fetch_data(url: str) -> str:
    # Simulate API call
    await asyncio.sleep(1)
    return f"Data from {url}"

# Await inside async function
async def main():
    result = await fetch_data("https://api.example.com")
    print(result)

# Run async code
asyncio.run(main())
```

### Multiple Concurrent Calls

```python
async def call_llm(prompt: str) -> str:
    await asyncio.sleep(1)  # Simulate API delay
    return f"Response to: {prompt}"

async def main():
    # Sequential (slow: 3 seconds)
    r1 = await call_llm("Prompt 1")
    r2 = await call_llm("Prompt 2")
    r3 = await call_llm("Prompt 3")

    # Concurrent (fast: 1 second!)
    results = await asyncio.gather(
        call_llm("Prompt 1"),
        call_llm("Prompt 2"),
        call_llm("Prompt 3")
    )

asyncio.run(main())
```

### Async Iterators (Streaming)

```python
from typing import AsyncIterator

# Async generator for streaming responses
async def stream_response(prompt: str) -> AsyncIterator[str]:
    words = prompt.split()
    for word in words:
        await asyncio.sleep(0.1)  # Simulate streaming delay
        yield word

# Consume stream
async def main():
    async for chunk in stream_response("Hello world from LLM"):
        print(chunk, end=" ", flush=True)

asyncio.run(main())
# Output: Hello world from LLM (one word at a time)
```

### Real LLM Example

```python
from typing import AsyncIterator

async def call_llm_streaming(prompt: str) -> AsyncIterator[str]:
    """Stream LLM response word by word."""
    # In reality, this would call an API
    response = "This is a streaming response from the LLM"
    for word in response.split():
        await asyncio.sleep(0.05)
        yield word + " "

async def main():
    print("Streaming response:")
    async for chunk in call_llm_streaming("Tell me a story"):
        print(chunk, end="", flush=True)
    print("\nDone!")

asyncio.run(main())
```

---

## Generic Functions

**Preserve types through function calls.**

```python
from typing import TypeVar

T = TypeVar('T')

# Generic retry wrapper
async def with_retry(func: callable, *args, max_attempts: int = 3) -> T:
    for attempt in range(max_attempts):
        try:
            return await func(*args)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Usage preserves return type
async def fetch_string() -> str:
    return "data"

result: str = await with_retry(fetch_string)  # Type is str!
```

---

## Context Managers (with statement)

**Automatic resource cleanup.**

```python
# File handling (auto-closes)
with open("config.json") as f:
    data = f.read()
# File automatically closed here

# Custom context manager
from contextlib import contextmanager

@contextmanager
def llm_session(api_key: str):
    client = connect(api_key)
    print("Session started")
    try:
        yield client
    finally:
        client.close()
        print("Session closed")

# Usage
with llm_session("api-key") as client:
    response = client.generate("Hello")
# Auto-cleanup happens here
```

---

## Pattern Matching (Python 3.10+)

**Like switch/case but more powerful.**

```python
def handle_response(response: dict) -> str:
    match response:
        case {"type": "text", "content": content}:
            return f"Text: {content}"
        case {"type": "error", "message": msg}:
            return f"Error: {msg}"
        case {"type": "stream", "delta": delta}:
            return f"Stream chunk: {delta}"
        case _:
            return "Unknown response type"

# Usage
print(handle_response({"type": "text", "content": "Hello"}))
# Output: Text: Hello
```

---

## List/Dict Unpacking

```python
# List unpacking
first, *rest = [1, 2, 3, 4]
# first = 1, rest = [2, 3, 4]

first, *middle, last = [1, 2, 3, 4, 5]
# first = 1, middle = [2, 3, 4], last = 5

# Dict unpacking (merge dicts)
base_config = {"model": "claude", "temp": 0.7}
overrides = {"temp": 0.9, "stream": True}
final_config = {**base_config, **overrides}
# {"model": "claude", "temp": 0.9, "stream": True}

# Function call unpacking
def call_llm(model: str, temperature: float, max_tokens: int):
    pass

config = {"model": "claude", "temperature": 0.7, "max_tokens": 1024}
call_llm(**config)  # Unpacks dict as keyword arguments
```

---

## Decorators (Quick Intro)

**Decorators wrap functions to add behavior.**

```python
# Simple timer decorator
import time
from functools import wraps

def timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# Usage
@timer
async def call_llm(prompt: str) -> str:
    await asyncio.sleep(1)
    return "Response"

# Equivalent to: call_llm = timer(call_llm)

await call_llm("Hello")
# Output: call_llm took 1.00s
```

---

## JSON Handling

**Critical for LLM work — all API I/O is JSON.**

```python
import json

# Parse JSON string
data = json.loads('{"model": "claude", "temp": 0.7}')
# {"model": "claude", "temp": 0.7}

# Convert to JSON string
config = {"model": "claude", "temp": 0.7}
json_str = json.dumps(config)
# '{"model": "claude", "temp": 0.7}'

# Pretty print
json_str = json.dumps(config, indent=2)
# {
#   "model": "claude",
#   "temp": 0.7
# }

# Read from file
with open("config.json") as f:
    config = json.load(f)

# Write to file
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# With dataclasses
from dataclasses import dataclass, asdict

@dataclass
class Config:
    model: str
    temperature: float

config = Config("claude", 0.7)
json.dumps(asdict(config))  # '{"model": "claude", "temperature": 0.7}'
```

---

## Next Steps

You now know modern Python patterns used in professional LLM code! You're ready for:

1. [exercises.py](exercises.py) — Practice these concepts
2. `01-foundations/` — Start building LLM applications

---

## Quick Reference: Modern Python Patterns

| Pattern | Syntax | Used For |
|---|---|---|
| Type hints | `def f(x: int) -> str` | Function signatures |
| Union types | `str \| None` | Optional/multiple types |
| Literal types | `Literal["a", "b"]` | Constrained values |
| Dataclasses | `@dataclass class X` | Structured data |
| Protocols | `class X(Protocol)` | Interfaces |
| Async functions | `async def f()` | Non-blocking I/O |
| Await | `await f()` | Wait for async result |
| Async iteration | `async for x in stream` | Streaming responses |
| Context managers | `with x as y:` | Resource cleanup |
| Pattern matching | `match x: case y:` | Conditional logic |
| Decorators | `@decorator` | Function wrappers |
| Dict unpacking | `{**a, **b}` | Merge dicts |
| Comprehensions | `[x for x in items]` | Transform collections |

**Next:** [exercises.py](exercises.py)
