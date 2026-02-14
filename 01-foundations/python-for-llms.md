# Python for LLM Work

> **Essential Python patterns for working with LLMs.** This guide covers the specific Python skills you'll use daily when building LLM applications.

---

## Working with JSON

**LLM APIs use JSON for everything** — requests, responses, configs, tool schemas. Mastering JSON is critical.

### Parsing JSON Responses

```python
import json

# API response as string
response_text = '{"content": "Hello!", "model": "claude", "tokens": 42}'

# Parse to dict
response = json.loads(response_text)
print(response["content"])   # "Hello!"
print(response["tokens"])    # 42

# Handle missing keys safely
model = response.get("model", "unknown")
stop_reason = response.get("stop_reason", None)  # None if missing

# Check if key exists
if "content" in response:
    print(response["content"])
```

### Creating JSON Requests

```python
# Build request dict
request = {
    "model": "claude-sonnet-4",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
}

# Convert to JSON string
request_json = json.dumps(request)

# Pretty print (for debugging)
print(json.dumps(request, indent=2))
# {
#   "model": "claude-sonnet-4",
#   "messages": [
#     {"role": "user", "content": "Hello"}
#   ],
#   ...
# }
```

### Nested JSON Structures

```python
# Typical LLM API response structure
response = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Here's the answer:"
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20
    }
}

# Navigate nested structure
text_content = response["content"][0]["text"]
input_tokens = response["usage"]["input_tokens"]

# Safe navigation with .get()
stop_reason = response.get("stop_reason")
if stop_reason:
    print(f"Stopped because: {stop_reason}")
```

### JSON Files

```python
# Read config from JSON file
with open("llm_config.json") as f:
    config = json.load(f)

# Write results to JSON file
results = {
    "model": "claude",
    "prompts_tested": 100,
    "avg_tokens": 250
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## String Manipulation for Prompts

### F-strings (Template Literals)

```python
# Basic f-string
user_input = "Python"
prompt = f"Explain {user_input} to a beginner."
# "Explain Python to a beginner."

# Multi-line prompts
system_message = "You are a helpful coding assistant."
user_query = "How do async functions work?"

full_prompt = f"""System: {system_message}

User: {user_query}"""
# Multi-line string with proper formatting

# Expression evaluation in f-strings
tokens = 150
cost_per_million = 3.0
cost = f"Cost: ${tokens / 1_000_000 * cost_per_million:.4f}"
# "Cost: $0.0005"

# Format numbers
temperature = 0.7
prompt = f"Temperature: {temperature:.2f}"  # "Temperature: 0.70"
```

### String Methods for Prompt Processing

```python
# Clean user input
user_input = "  Hello World!  "
cleaned = user_input.strip()           # "Hello World!"
cleaned = user_input.lower().strip()   # "hello world!"

# Check content
if "python" in prompt.lower():
    category = "programming"

# Split and join
conversation = "user: Hi\nassistant: Hello\nuser: Bye"
messages = conversation.split("\n")    # ["user: Hi", "assistant: Hello", ...]

# Join list into string
lines = ["System: Be helpful", "User: Hello"]
prompt = "\n\n".join(lines)

# Replace text
template = "Explain {{topic}} in simple terms"
prompt = template.replace("{{topic}}", "recursion")

# Find and extract
response = "Here's the answer: 42. That's it!"
answer_start = response.find("answer: ") + len("answer: ")
answer_end = response.find(".", answer_start)
answer = response[answer_start:answer_end]  # "42"
```

### Working with Multi-line Strings

```python
# Triple quotes for multi-line
prompt_template = """
You are a {role}.

Your task:
{task}

Context:
{context}
"""

# Format with .format() or f-strings
prompt = prompt_template.format(
    role="Python expert",
    task="Explain decorators",
    context="User is a beginner"
)

# Strip extra whitespace
prompt = prompt.strip()
```

---

## Working with Lists of Messages

### Message List Patterns

```python
# Typical conversation structure
messages: list[dict[str, str]] = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Explain Python"}
]

# Add message
messages.append({"role": "assistant", "content": "Python is..."})

# Get last message
last_message = messages[-1]

# Get last user message
user_messages = [m for m in messages if m["role"] == "user"]
last_user_msg = user_messages[-1] if user_messages else None

# Count tokens across all messages
total_tokens = sum(len(m["content"]) // 4 for m in messages)

# Build conversation string
conversation_text = "\n\n".join(
    f"{m['role'].upper()}: {m['content']}"
    for m in messages
)
```

### Filtering and Transforming Messages

```python
# Extract user messages only
user_messages = [m["content"] for m in messages if m["role"] == "user"]

# Remove system messages
chat_messages = [m for m in messages if m["role"] != "system"]

# Add metadata to messages
enriched_messages = [
    {**msg, "tokens": len(msg["content"]) // 4}
    for msg in messages
]

# Truncate long messages
MAX_LENGTH = 500
truncated_messages = [
    {
        "role": m["role"],
        "content": m["content"][:MAX_LENGTH] + "..."
                   if len(m["content"]) > MAX_LENGTH
                   else m["content"]
    }
    for m in messages
]
```

---

## Dictionary Operations for Configs

### Building Configuration Dicts

```python
# Base configuration
base_config = {
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "temperature": 0.7,
}

# Merge with overrides
user_overrides = {"temperature": 0.9, "top_p": 0.95}
final_config = {**base_config, **user_overrides}
# {"model": "claude-sonnet-4", "max_tokens": 1024, "temperature": 0.9, "top_p": 0.95}

# Conditional config building
def build_llm_config(model: str, creative: bool = False) -> dict:
    config = {"model": model}

    if creative:
        config["temperature"] = 1.0
        config["top_p"] = 0.95
    else:
        config["temperature"] = 0.3

    return config

# Extract specific keys
full_config = {
    "model": "claude",
    "temperature": 0.7,
    "max_tokens": 1024,
    "internal_id": "abc123"  # Don't send to API
}

# Filter for API request (exclude internal fields)
api_config = {
    k: v for k, v in full_config.items()
    if k != "internal_id"
}
```

### Validating Configs

```python
def validate_config(config: dict) -> bool:
    """Validate LLM config has required fields."""
    required_keys = ["model", "max_tokens"]

    # Check all required keys present
    if not all(key in config for key in required_keys):
        return False

    # Validate value ranges
    if config["max_tokens"] <= 0:
        return False

    if "temperature" in config:
        if not (0 <= config["temperature"] <= 1):
            return False

    return True

# Usage
config = {"model": "claude", "max_tokens": 1024, "temperature": 0.7}
if validate_config(config):
    response = call_llm(**config)
```

---

## Error Handling for API Calls

### Try/Except Patterns

```python
import json
from typing import Any

def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Parse JSON, return None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# Usage
response_text = '{"result": "success"}'
data = safe_json_parse(response_text)
if data:
    print(data["result"])
else:
    print("Invalid JSON")
```

### Handling API Errors

```python
class LLMError(Exception):
    """Custom exception for LLM errors."""
    pass

def call_llm_safe(prompt: str, retries: int = 3) -> str:
    """Call LLM with retry logic and error handling."""
    for attempt in range(retries):
        try:
            response = call_llm_api(prompt)
            return response["content"]

        except KeyError as e:
            # Response missing expected field
            raise LLMError(f"Invalid response format: {e}")

        except TimeoutError:
            if attempt == retries - 1:
                raise LLMError("Request timed out after retries")
            continue  # Retry

        except Exception as e:
            # Unexpected error
            raise LLMError(f"Unexpected error: {e}")

    raise LLMError("Max retries exceeded")

# Usage with error handling
try:
    result = call_llm_safe("Explain Python")
    print(result)
except LLMError as e:
    print(f"LLM call failed: {e}")
```

---

## List Comprehensions for Data Processing

### Common LLM Data Processing Patterns

```python
# Extract specific fields from list of dicts
responses = [
    {"id": 1, "model": "claude", "content": "Hello", "tokens": 10},
    {"id": 2, "model": "gpt4", "content": "Hi", "tokens": 8},
    {"id": 3, "model": "claude", "content": "Hey", "tokens": 9}
]

# Get all content
all_content = [r["content"] for r in responses]
# ["Hello", "Hi", "Hey"]

# Filter by model
claude_responses = [r for r in responses if r["model"] == "claude"]

# Calculate total tokens
total_tokens = sum(r["tokens"] for r in responses)  # 27

# Build new structure
formatted = [
    f"{r['model']}: {r['content']} ({r['tokens']} tokens)"
    for r in responses
]
# ["claude: Hello (10 tokens)", ...]

# Dict comprehension: group by model
from collections import defaultdict
by_model = defaultdict(list)
for r in responses:
    by_model[r["model"]].append(r)

# Or with comprehension (for unique keys)
unique_models = {r["model"]: r for r in responses}
# {" claude": {last claude response}, "gpt4": {gpt4 response}}
```

### Chaining Operations

```python
# Get content from user messages, lowercase, filter short ones
user_content = [
    m["content"].lower()
    for m in messages
    if m["role"] == "user" and len(m["content"]) > 10
]

# Process in steps (more readable)
user_messages = [m for m in messages if m["role"] == "user"]
long_messages = [m for m in user_messages if len(m["content"]) > 10]
content_lower = [m["content"].lower() for m in long_messages]
```

---

## Practical Examples

### Complete Message Processing Pipeline

```python
def process_conversation(
    messages: list[dict[str, str]],
    max_context_tokens: int = 4000
) -> list[dict[str, str]]:
    """Process conversation to fit within context window."""

    # 1. Remove system messages (will add back at end)
    system_msgs = [m for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]

    # 2. Estimate tokens
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    # 3. Truncate from beginning if needed
    total_tokens = sum(estimate_tokens(m["content"]) for m in chat_msgs)

    while total_tokens > max_context_tokens and len(chat_msgs) > 1:
        removed = chat_msgs.pop(0)
        total_tokens -= estimate_tokens(removed["content"])

    # 4. Add system message back at start
    return system_msgs + chat_msgs

# Usage
long_conversation = [
    {"role": "system", "content": "Be helpful"},
    {"role": "user", "content": "Hi" * 1000},  # Very long
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "Question?"}
]

processed = process_conversation(long_conversation, max_context_tokens=100)
```

### JSON Response Parser with Error Recovery

```python
def extract_json_from_llm(response: str) -> dict | None:
    """
    Extract JSON from LLM response.
    Handles markdown code blocks and extra text.
    """
    import json
    import re

    # Try parsing directly first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ``` blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Look for {...} anywhere in text
    brace_pattern = r'\{[^{}]*\}'
    matches = re.findall(brace_pattern, response)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None

# Usage
llm_response = """
Here's the data you requested:

```json
{
  "name": "Claude",
  "version": 4
}
```

Let me know if you need anything else!
"""

data = extract_json_from_llm(llm_response)
print(data)  # {"name": "Claude", "version": 4}
```

---

## Next Steps

You now have the Python fundamentals for LLM work! Practice these patterns by:

1. Completing [exercises.py](exercises.py)
2. Reading [concepts.md](concepts.md) to understand how LLMs work
3. Moving on to prompt engineering in `02-prompt-engineering/`

**Key takeaway:** LLM programming is mostly **data transformation** — parsing JSON, manipulating strings, filtering lists, and building dictionaries. Master these patterns and you'll be productive quickly.

---

## Quick Reference

| Task | Python Pattern |
|---|---|
| Parse JSON | `json.loads(text)` |
| Create JSON | `json.dumps(dict)` |
| Format strings | `f"Hello {name}"` |
| Multi-line strings | `"""text"""` or `f"""text {var}"""` |
| Safe dict access | `d.get("key", default)` |
| Check key exists | `"key" in dict` |
| Filter messages | `[m for m in msgs if m["role"] == "user"]` |
| Extract field | `[m["content"] for m in msgs]` |
| Merge dicts | `{**dict1, **dict2}` |
| String cleanup | `text.strip().lower()` |
| Error handling | `try/except` with specific exceptions |
| List total | `sum(item["value"] for item in items)` |

**Next:** [exercises.py](exercises.py) or [concepts.md](concepts.md)
