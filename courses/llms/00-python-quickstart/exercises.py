"""
Python Quickstart Exercises

Practice problems to reinforce Python fundamentals before LLM work.
All exercises relate to tasks you'll do when working with LLMs.

Run individual exercises by uncommenting the test calls at the bottom.
"""

# ============================================================================
# PART 1: STRINGS AND FORMATTING
# ============================================================================

def build_prompt(system_message: str, user_message: str) -> str:
    """
    Combine system and user messages into a formatted prompt.

    Example:
        build_prompt("You are helpful", "Hello")
        -> "SYSTEM: You are helpful\n\nUSER: Hello"
    """
    prompt = f"SYSTEM: {system_message}\n\nUSER: {user_message}"
    return prompt


def count_tokens_estimate(text: str) -> int:
    """
    Rough token estimate: ~1 token per 4 characters.

    Example:
        count_tokens_estimate("Hello world")  -> 2
    """
    return len(text) // 4    


def truncate_prompt(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within max_tokens (assume 4 chars = 1 token).
    Add "..." if truncated.

    Example:
        truncate_prompt("Hello world", max_tokens=2)  -> "Hello wo..."
    """
    # TODO: Implement
    pass


# ============================================================================
# PART 2: LISTS AND DICTS
# ============================================================================

def extract_user_messages(messages: list[dict[str, str]]) -> list[str]:
    """
    Extract content from all user messages.

    Example:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"}
        ]
        extract_user_messages(messages)  -> ["Hi", "Bye"]
    """
    # TODO: Use list comprehension
    pass


def count_tokens_per_message(messages: list[dict[str, str]]) -> dict[str, int]:
    """
    Return dict mapping role to total token count (use 4 chars = 1 token).

    Example:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"}
        ]
        count_tokens_per_message(messages)
        -> {"user": 0, "assistant": 2}
    """
    # TODO: Implement with dict operations
    pass


def merge_configs(base: dict, overrides: dict) -> dict:
    """
    Merge two config dicts. Overrides take precedence.
    Don't modify the original dicts.

    Example:
        base = {"model": "claude", "temp": 0.7}
        overrides = {"temp": 0.9}
        merge_configs(base, overrides)
        -> {"model": "claude", "temp": 0.9}
    """
    # TODO: Use dict unpacking
    pass


# ============================================================================
# PART 3: DATACLASSES
# ============================================================================

from dataclasses import dataclass, asdict

@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str

    # TODO: Add method is_user() -> bool that returns True if role == "user"

    # TODO: Add method token_estimate() -> int using 4 chars = 1 token


@dataclass
class LLMResponse:
    """Represents an LLM API response."""
    content: str
    model: str
    tokens_used: int

    # TODO: Add method to_dict() -> dict that returns dict representation
    # Hint: use asdict()


def create_conversation(messages: list[tuple[str, str]]) -> list[Message]:
    """
    Convert list of (role, content) tuples to Message objects.

    Example:
        create_conversation([("user", "Hi"), ("assistant", "Hello")])
        -> [Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello")]
    """
    # TODO: Implement using list comprehension
    pass


# ============================================================================
# PART 4: TYPE HINTS AND PROTOCOLS
# ============================================================================

from typing import Protocol, Literal

# Define valid roles
Role = Literal["system", "user", "assistant"]

def validate_role(role: str) -> Role:
    """
    Validate that role is one of: system, user, assistant.
    Raise ValueError if invalid.

    Example:
        validate_role("user")  -> "user"
        validate_role("admin")  -> ValueError
    """
    # TODO: Implement with if/elif/else and raise
    pass


class LLMProvider(Protocol):
    """Protocol defining the interface for any LLM provider."""

    def generate(self, prompt: str, temperature: float) -> str:
        """Generate a response to the prompt."""
        ...


class MockLLM:
    """A mock LLM for testing that implements LLMProvider protocol."""

    # TODO: Implement generate() method that returns f"Mock response to: {prompt}"
    pass


def call_with_retries(provider: LLMProvider, prompt: str, attempts: int = 3) -> str:
    """
    Call provider.generate() with retry logic.
    Return result on success, or "Failed after N attempts" on failure.

    Simulate failure: if "error" in prompt.lower(), raise Exception("API Error")
    """
    # TODO: Implement with for loop and try/except
    pass


# ============================================================================
# PART 5: ASYNC (Preview - will use more in later sections)
# ============================================================================

import asyncio
from typing import AsyncIterator

async def simulate_llm_call(prompt: str, delay: float = 0.5) -> str:
    """
    Simulate an async LLM API call with delay.
    """
    await asyncio.sleep(delay)
    return f"Response to: {prompt}"


async def call_multiple_llms(prompts: list[str]) -> list[str]:
    """
    Call simulate_llm_call for each prompt concurrently.
    Return list of responses in same order as prompts.

    Hint: Use asyncio.gather()
    """
    # TODO: Implement
    pass


async def stream_words(text: str) -> AsyncIterator[str]:
    """
    Async generator that yields one word at a time with 0.1s delay.

    Example usage:
        async for word in stream_words("Hello world"):
            print(word)
    """
    # TODO: Implement with async for and yield
    pass


# ============================================================================
# PART 6: PRACTICAL LLM DATA PROCESSING
# ============================================================================

def parse_json_response(response: str) -> dict | None:
    """
    Parse JSON from LLM response. LLMs sometimes include extra text.
    Extract JSON from markdown code blocks if present.
    Return None if parsing fails.

    Example:
        parse_json_response('```json\n{"key": "value"}\n```')
        -> {"key": "value"}
    """
    import json

    # TODO:
    # 1. Check if response contains ```json...```
    # 2. If yes, extract just the JSON part
    # 3. Parse with json.loads()
    # 4. Return None if any step fails (use try/except)
    pass


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract all code blocks from markdown text.
    Code blocks are surrounded by ``` markers.

    Example:
        text = "Here's code:\n```python\nprint('hi')\n```\nAnd more:\n```\ntest\n```"
        extract_code_blocks(text)  -> ["python\nprint('hi')", "test"]
    """
    # TODO: Implement using string methods (split, strip)
    # Hint: Split by "```", take every other segment
    pass


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float
) -> float:
    """
    Calculate cost for LLM API call.
    Prices are per 1 million tokens.

    Example:
        # Claude Sonnet 4: $3 input, $15 output per 1M tokens
        calculate_cost(1000, 500, 3.0, 15.0)  -> 0.0105
    """
    # TODO: Implement
    pass


# ============================================================================
# TESTS (Uncomment to run)
# ============================================================================

def test_part1():
    """Test string and formatting exercises."""
    print("\n=== PART 1: STRINGS ===")
    print(build_prompt("Be concise", "Hello"))
    print(f"Token estimate: {count_tokens_estimate('Hello world')}")
    print(f"Truncated: {truncate_prompt('Hello world how are you', 3)}")


def test_part2():
    """Test list and dict exercises."""
    print("\n=== PART 2: LISTS & DICTS ===")
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "How are you?"}
    ]
    print(f"User messages: {extract_user_messages(messages)}")
    print(f"Token counts: {count_tokens_per_message(messages)}")

    base = {"model": "claude", "temp": 0.7, "max_tokens": 1024}
    overrides = {"temp": 0.9}
    print(f"Merged config: {merge_configs(base, overrides)}")


def test_part3():
    """Test dataclass exercises."""
    print("\n=== PART 3: DATACLASSES ===")
    msg = Message(role="user", content="Hello")
    print(f"Message: {msg}")
    print(f"Is user: {msg.is_user()}")
    print(f"Token estimate: {msg.token_estimate()}")

    response = LLMResponse(content="Hi there", model="claude", tokens_used=10)
    print(f"Response dict: {response.to_dict()}")

    conversation = create_conversation([
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "Bye")
    ])
    print(f"Conversation: {conversation}")


def test_part4():
    """Test type hints and protocols."""
    print("\n=== PART 4: TYPES & PROTOCOLS ===")
    try:
        print(f"Valid role: {validate_role('user')}")
        validate_role("admin")  # Should raise
    except ValueError as e:
        print(f"Caught error: {e}")

    mock = MockLLM()
    result = call_with_retries(mock, "test prompt")
    print(f"Mock result: {result}")

    result = call_with_retries(mock, "this will error")
    print(f"After retries: {result}")


async def test_part5():
    """Test async exercises."""
    print("\n=== PART 5: ASYNC ===")

    # Single call
    result = await simulate_llm_call("Hello")
    print(f"Single result: {result}")

    # Multiple concurrent calls
    prompts = ["First", "Second", "Third"]
    results = await call_multiple_llms(prompts)
    print(f"Multiple results: {results}")

    # Streaming
    print("Streaming: ", end="", flush=True)
    async for word in stream_words("Hello world from LLM"):
        print(word, end=" ", flush=True)
    print()


def test_part6():
    """Test practical LLM data processing."""
    print("\n=== PART 6: PRACTICAL ===")

    # Parse JSON
    response = '```json\n{"model": "claude", "version": 4}\n```'
    parsed = parse_json_response(response)
    print(f"Parsed JSON: {parsed}")

    # Extract code blocks
    text = "Here's Python:\n```python\nprint('hi')\n```\nAnd shell:\n```bash\nls\n```"
    blocks = extract_code_blocks(text)
    print(f"Code blocks: {blocks}")

    # Calculate cost
    cost = calculate_cost(1000, 500, 3.0, 15.0)
    print(f"API cost: ${cost:.4f}")


# Uncomment to run tests:
# test_part1()
# test_part2()
# test_part3()
# test_part4()
# asyncio.run(test_part5())
# test_part6()

"""
SOLUTIONS CHECKLIST
After attempting exercises, verify you covered:

Part 1: Strings
- [ ] Used f-strings for formatting
- [ ] String methods: len(), slicing
- [ ] Conditional string building

Part 2: Lists & Dicts
- [ ] List comprehensions with filtering
- [ ] Dict operations: get(), iteration
- [ ] Dict unpacking with {**a, **b}

Part 3: Dataclasses
- [ ] @dataclass decorator
- [ ] Methods in dataclasses
- [ ] asdict() for serialization
- [ ] Type hints on fields

Part 4: Types & Protocols
- [ ] Literal types for constrained values
- [ ] Protocol definition
- [ ] Implementing protocol without inheritance
- [ ] Exception handling with try/except

Part 5: Async
- [ ] async def and await
- [ ] asyncio.gather() for concurrency
- [ ] Async generators with yield
- [ ] AsyncIterator type hint

Part 6: Practical
- [ ] JSON parsing with error handling
- [ ] String manipulation (split, strip, find)
- [ ] Mathematical calculations
- [ ] Edge case handling

All of these patterns appear in real LLM code!
"""
