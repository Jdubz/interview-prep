# Python Basics

> **For TypeScript developers:** This guide highlights Python fundamentals with comparisons to TypeScript where helpful. Focus on the differences.

---

## Variables and Basic Types

Python uses dynamic typing but supports optional type hints (covered in [modern-python.md](modern-python.md)).

```python
# Variables (no const/let/var needed)
name = "Claude"              # str
age = 2                      # int
temperature = 0.7            # float
is_streaming = True          # bool (capital T/F!)

# Python uses snake_case, not camelCase
max_tokens = 1024
model_name = "claude-sonnet-4"
```

**Key differences from TypeScript:**
- No `const`, `let`, or `var` — just assign
- `True`/`False` (capitalized), not `true`/`false`
- `None` instead of `null` or `undefined`
- snake_case convention instead of camelCase

---

## Strings

```python
# Three ways to create strings
single = 'Hello'
double = "Hello"
multiline = """
This string
spans multiple lines
"""

# F-strings (like template literals)
name = "Claude"
greeting = f"Hello, {name}!"                    # "Hello, Claude!"
prompt = f"You are {name}, a helpful AI."

# String methods
text = "  hello world  "
text.strip()           # "hello world" (removes whitespace)
text.upper()           # "  HELLO WORLD  "
text.replace("o", "0") # "  hell0 w0rld  "
text.split()           # ["hello", "world"]

# Concatenation
full_prompt = "You are helpful. " + "Be concise."
# Or with f-strings (preferred)
full_prompt = f"{system_prompt} {user_message}"
```

**For LLM work:** You'll use f-strings constantly for prompt templates.

---

## Collections

### Lists (like arrays)

```python
# Create lists
models = ["gpt-4", "claude-3", "llama-2"]
numbers = [1, 2, 3, 4, 5]
mixed = ["text", 42, True, None]  # Can mix types

# Access elements
first = models[0]        # "gpt-4"
last = models[-1]        # "llama-2" (negative indexing!)

# Slicing
first_two = models[0:2]  # ["gpt-4", "claude-3"]
last_two = models[-2:]   # ["claude-3", "llama-2"]

# Modify
models.append("gemini")           # Add to end
models.insert(0, "gpt-3.5")       # Insert at position
models.remove("llama-2")          # Remove by value
popped = models.pop()             # Remove and return last

# Common operations
len(models)              # Length
"gpt-4" in models        # Check membership
models.sort()            # Sort in place
sorted(models)           # Return sorted copy
```

### Dictionaries (like objects)

```python
# Create dicts (like TS objects/Records)
config = {
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "temperature": 0.7
}

# Access values
model = config["model"]              # "claude-sonnet-4"
model = config.get("model")          # Same, but returns None if missing
model = config.get("model", "default")  # With default value

# Modify
config["temperature"] = 0.9          # Update
config["stream"] = True              # Add new key

# Common operations
config.keys()            # dict_keys(['model', 'max_tokens', ...])
config.values()          # dict_values(['claude-sonnet-4', 1024, ...])
config.items()           # Key-value pairs

# Check membership
"model" in config        # True
"api_key" in config      # False
```

**For LLM work:** Dicts are used everywhere — API responses, configs, tool schemas.

### Tuples (immutable lists)

```python
# Tuples use parentheses (immutable)
coordinates = (1.5, 2.3)
model_and_temp = ("claude", 0.7)

# Unpacking
x, y = coordinates
model, temp = model_and_temp
```

### Sets (unique values)

```python
# Sets (like TS Set)
unique_tags = {"python", "llm", "rag"}
unique_tags.add("agents")
unique_tags.remove("rag")

# Set operations
a = {1, 2, 3}
b = {2, 3, 4}
a & b  # Intersection: {2, 3}
a | b  # Union: {1, 2, 3, 4}
a - b  # Difference: {1}
```

---

## Control Flow

### If statements

```python
# If/elif/else (note the colons and indentation!)
temperature = 0.7

if temperature < 0.5:
    mode = "deterministic"
elif temperature < 1.0:
    mode = "balanced"
else:
    mode = "creative"

# Inline if (ternary)
mode = "high" if temperature > 0.8 else "low"

# Truthiness
# False: None, False, 0, "", [], {}, ()
# True: Everything else

if api_key:  # Checks if api_key is truthy
    client = create_client(api_key)
```

**Important:** Python uses indentation (4 spaces) for blocks, not braces `{}`.

### Loops

```python
# For loop (over items)
models = ["gpt-4", "claude", "llama"]
for model in models:
    print(f"Testing {model}")

# For loop with index
for i, model in enumerate(models):
    print(f"{i}: {model}")

# For loop over dict
config = {"model": "claude", "temp": 0.7}
for key, value in config.items():
    print(f"{key} = {value}")

# While loop
attempts = 0
while attempts < 3:
    print(f"Attempt {attempts}")
    attempts += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip to next iteration
    if i == 7:
        break     # Exit loop
    print(i)
```

### Comprehensions (powerful!)

```python
# List comprehension (like TS map)
numbers = [1, 2, 3, 4]
doubled = [n * 2 for n in numbers]  # [2, 4, 6, 8]

# With filter
evens = [n for n in numbers if n % 2 == 0]  # [2, 4]

# Dict comprehension
squared = {n: n**2 for n in numbers}  # {1: 1, 2: 4, 3: 9, 4: 16}

# Real example: extract models from responses
responses = [
    {"model": "gpt-4", "tokens": 100},
    {"model": "claude", "tokens": 200}
]
models = [r["model"] for r in responses]  # ["gpt-4", "claude"]
```

---

## Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

result = greet("Claude")  # "Hello, Claude!"

# With type hints (optional but recommended)
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Default arguments
def create_prompt(message: str, system: str = "You are helpful") -> str:
    return f"{system}\n\n{message}"

# Keyword arguments
prompt = create_prompt(message="Hi", system="Be concise")
prompt = create_prompt("Hi", system="Be concise")  # Positional + keyword

# Multiple return values (returns tuple)
def analyze_response(text: str) -> tuple[int, float]:
    tokens = len(text.split())
    avg_length = len(text) / tokens if tokens > 0 else 0
    return tokens, avg_length

token_count, avg = analyze_response("Hello world")

# *args and **kwargs (variable arguments)
def call_llm(prompt: str, **options):
    # options is a dict of all keyword arguments
    model = options.get("model", "claude")
    temp = options.get("temperature", 0.7)
    return f"Calling {model} with temp {temp}"

result = call_llm("Hi", model="gpt-4", temperature=0.9)
```

---

## Classes

```python
# Basic class
class LLMConfig:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature
        }

# Usage
config = LLMConfig("claude", 0.7)
print(config.model)        # "claude"
print(config.to_dict())    # {"model": "claude", "temperature": 0.7}

# Class with properties
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    @property
    def length(self) -> int:
        return len(self.content)

    def is_user(self) -> bool:
        return self.role == "user"

msg = Message("user", "Hello!")
print(msg.length)    # 6 (property, no parentheses!)
print(msg.is_user()) # True (method, needs parentheses)
```

**For LLM work:** You'll mostly use dataclasses (next file) instead of manual classes.

---

## Imports

```python
# Import entire module
import json
data = json.loads('{"key": "value"}')

# Import specific items
from json import loads, dumps
data = loads('{"key": "value"}')

# Import with alias
import json as j
data = j.loads('{"key": "value"}')

# Common imports for LLM work
import json              # JSON parsing
from typing import List  # Type hints (pre-3.9)
from dataclasses import dataclass  # Data classes
```

---

## Error Handling

```python
# Try/except (like try/catch)
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Always runs")

# Common pattern for LLM work
def call_api(prompt: str) -> str:
    try:
        response = api.call(prompt)
        return response.text
    except TimeoutError:
        return "Request timed out"
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise  # Re-raise the exception
```

---

## Common Built-ins

```python
# Type checking
isinstance("hello", str)        # True
isinstance([1, 2], list)        # True
type("hello")                   # <class 'str'>

# Conversions
str(42)                         # "42"
int("42")                       # 42
float("3.14")                   # 3.14
list("abc")                     # ['a', 'b', 'c']
dict([("a", 1), ("b", 2)])      # {"a": 1, "b": 2}

# Useful functions
len([1, 2, 3])                  # 3
max([1, 5, 3])                  # 5
min([1, 5, 3])                  # 1
sum([1, 2, 3])                  # 6
sorted([3, 1, 2])               # [1, 2, 3]
reversed([1, 2, 3])             # iterator [3, 2, 1]

# Range (for loops)
range(5)                        # 0, 1, 2, 3, 4
range(1, 5)                     # 1, 2, 3, 4
range(0, 10, 2)                 # 0, 2, 4, 6, 8

# Zip (combine lists)
names = ["alice", "bob"]
scores = [95, 87]
list(zip(names, scores))        # [("alice", 95), ("bob", 87)]
```

---

## Next Steps

You now know enough Python to start learning LLM concepts! Move on to:

1. [modern-python.md](modern-python.md) — Type hints, dataclasses, protocols
2. [exercises.py](exercises.py) — Practice problems
3. `01-foundations/` — Start learning LLM concepts with Python

---

## Quick Reference: Python vs TypeScript

| TypeScript | Python | Notes |
|---|---|---|
| `const x = 5` | `x = 5` | No const/let/var |
| `let arr: string[]` | `arr: list[str]` | Type hints optional |
| `camelCase` | `snake_case` | Naming convention |
| `true`/`false` | `True`/`False` | Capitalized |
| `null`, `undefined` | `None` | Single null value |
| `[1, 2, 3]` | `[1, 2, 3]` | Lists (arrays) |
| `{a: 1}` | `{"a": 1}` | Dicts (objects) |
| `` `Hello ${name}` `` | `f"Hello {name}"` | Template strings |
| `arr.map(x => x*2)` | `[x*2 for x in arr]` | Comprehensions |
| `function` | `def` | Function keyword |
| `class { constructor() }` | `class: def __init__()` | Constructor |
| `this` | `self` | Instance reference |
| `import { x } from 'y'` | `from y import x` | Imports |

---

**Next:** [modern-python.md](modern-python.md) for type hints, dataclasses, and async/await
