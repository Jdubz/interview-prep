"""
Foundations Exercises: Python + LLM Concepts

Practice Python skills while working with LLM-related data structures.
These exercises reinforce both Python fundamentals and LLM concepts.
"""

import json
from typing import Literal

# ============================================================================
# EXERCISE 1: JSON Parsing and API Response Handling
# ============================================================================

def extract_message_content(response_json: str) -> str | None:
    """
    Parse an LLM API response and extract the message content.
    Return None if parsing fails or content is missing.

    Example response:
    {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }

    Should return: "Hello!"
    """
    # TODO: Implement
    # 1. Parse JSON string to dict
    # 2. Navigate to content[0]["text"]
    # 3. Handle errors and missing fields
    pass


def calculate_api_cost(response_json: str) -> float:
    """
    Calculate cost from API response.
    Input tokens: $3 per 1M tokens
    Output tokens: $15 per 1M tokens

    Return 0.0 if parsing fails.
    """
    # TODO: Implement
    # 1. Parse JSON
    # 2. Extract usage.input_tokens and usage.output_tokens
    # 3. Calculate: (input * 3 + output * 15) / 1_000_000
    pass


# ============================================================================
# EXERCISE 2: Message List Processing
# ============================================================================

def build_message_list(conversation: str) -> list[dict[str, str]]:
    """
    Convert conversation string to list of message dicts.

    Input format:
        "user: Hello
        assistant: Hi there!
        user: How are you?"

    Output:
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    """
    # TODO: Implement
    # 1. Split by newlines
    # 2. For each line, split by ": " to get role and content
    # 3. Build list of dicts
    pass


def count_conversation_tokens(messages: list[dict[str, str]]) -> dict[str, int]:
    """
    Count tokens per role (estimate: 1 token per 4 characters).

    Example:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there, how are you?"}
        ]
        Returns: {"user": 1, "assistant": 5}
    """
    # TODO: Implement using dict operations and comprehensions
    pass


def truncate_conversation(
    messages: list[dict[str, str]],
    max_tokens: int
) -> list[dict[str, str]]:
    """
    Truncate conversation from the beginning to fit within max_tokens.
    Always keep the last message.
    Token estimate: 1 token per 4 characters.

    Example:
        messages = [
            {"role": "user", "content": "x" * 100},     # ~25 tokens
            {"role": "assistant", "content": "y" * 100}, # ~25 tokens
            {"role": "user", "content": "z" * 100}      # ~25 tokens
        ]
        truncate_conversation(messages, max_tokens=40)
        # Should remove first message, keep last two
    """
    # TODO: Implement
    pass


# ============================================================================
# EXERCISE 3: Prompt Template Building
# ============================================================================

def build_system_prompt(
    role: str,
    traits: list[str],
    constraints: list[str] | None = None
) -> str:
    """
    Build a system prompt from components.

    Example:
        role = "Python tutor"
        traits = ["patient", "clear explanations"]
        constraints = ["keep answers under 100 words"]

        Returns:
        "You are a Python tutor. You are patient and clear explanations.

        Constraints:
        - keep answers under 100 words"
    """
    # TODO: Implement using f-strings and string methods
    pass


def format_few_shot_examples(
    examples: list[tuple[str, str]],
    input_label: str = "Input",
    output_label: str = "Output"
) -> str:
    """
    Format few-shot examples for prompt.

    Example:
        examples = [
            ("2 + 2", "4"),
            ("5 * 3", "15")
        ]

        Returns:
        "Input: 2 + 2
        Output: 4

        Input: 5 * 3
        Output: 15"
    """
    # TODO: Implement
    pass


# ============================================================================
# EXERCISE 4: Configuration Management
# ============================================================================

def create_llm_config(
    model: Literal["gpt-4", "claude-sonnet-4", "claude-opus-4"],
    mode: Literal["precise", "balanced", "creative"] = "balanced"
) -> dict:
    """
    Create LLM configuration dict based on mode.

    Modes:
    - precise: temperature=0.3, top_p=0.9
    - balanced: temperature=0.7, top_p=0.9
    - creative: temperature=1.0, top_p=0.95

    Always include: model, max_tokens=1024
    """
    # TODO: Implement using dict building and conditionals
    pass


def merge_configs(*configs: dict) -> dict:
    """
    Merge multiple config dicts. Later configs override earlier ones.

    Example:
        base = {"model": "claude", "temp": 0.7}
        override = {"temp": 0.9, "top_p": 0.95}
        merge_configs(base, override)
        # Returns: {"model": "claude", "temp": 0.9, "top_p": 0.95}
    """
    # TODO: Implement using dict unpacking
    pass


def validate_llm_config(config: dict) -> tuple[bool, str]:
    """
    Validate LLM config. Return (is_valid, error_message).

    Rules:
    - Must have "model" and "max_tokens" keys
    - max_tokens must be > 0 and <= 100000
    - If "temperature" present, must be 0.0 to 1.0
    - If "top_p" present, must be 0.0 to 1.0

    Return (True, "") if valid, (False, "error reason") if invalid
    """
    # TODO: Implement validation logic
    pass


# ============================================================================
# EXERCISE 5: Response Parsing
# ============================================================================

def extract_json_from_text(text: str) -> dict | None:
    """
    Extract JSON object from text that may contain other content.
    LLMs often return JSON wrapped in markdown or with explanation text.

    Handle these cases:
    1. Plain JSON: '{"key": "value"}'
    2. Markdown: '```json\n{"key": "value"}\n```'
    3. With text: 'Here is the data: {"key": "value"}'

    Return None if no valid JSON found.
    """
    # TODO: Implement
    # Hint: Try json.loads first, then look for patterns
    pass


def parse_tool_calls(response: str) -> list[dict[str, str]]:
    """
    Parse function/tool calls from LLM response.

    Format:
        "TOOL: calculator
        INPUT: 2 + 2

        TOOL: search
        INPUT: Python tutorials"

    Returns:
        [
            {"tool": "calculator", "input": "2 + 2"},
            {"tool": "search", "input": "Python tutorials"}
        ]
    """
    # TODO: Implement using string methods
    pass


# ============================================================================
# EXERCISE 6: Token Estimation and Context Windows
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count (1 token ≈ 4 characters for English).
    This is a rough estimate - real tokenization is more complex.
    """
    # TODO: Implement simple estimation
    pass


def fits_in_context(
    messages: list[dict[str, str]],
    max_context: int = 200000  # Claude's context window
) -> bool:
    """
    Check if conversation fits in context window.
    Use estimate_tokens() for each message.
    """
    # TODO: Implement
    pass


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,  # tokens
    overlap: int = 100       # tokens
) -> list[str]:
    """
    Split text into overlapping chunks for processing.
    Use character approximation: 1 token ≈ 4 chars.

    Example:
        text = "a" * 10000  # ~2500 tokens
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
        # Should return 3 chunks with overlap
    """
    # TODO: Implement
    # Hint: Convert tokens to characters, use slicing
    pass


# ============================================================================
# TESTS
# ============================================================================

def test_exercise_1():
    print("\n=== EXERCISE 1: JSON Parsing ===")

    response = '''
    {
        "id": "msg_123",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    '''

    content = extract_message_content(response)
    print(f"Content: {content}")

    cost = calculate_api_cost(response)
    print(f"Cost: ${cost:.6f}")


def test_exercise_2():
    print("\n=== EXERCISE 2: Message Lists ===")

    conv = """user: Hello
assistant: Hi there!
user: How are you?"""

    messages = build_message_list(conv)
    print(f"Messages: {messages}")

    tokens = count_conversation_tokens(messages)
    print(f"Tokens by role: {tokens}")

    long_messages = [
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 400},
        {"role": "user", "content": "z" * 400}
    ]
    truncated = truncate_conversation(long_messages, max_tokens=150)
    print(f"Truncated from {len(long_messages)} to {len(truncated)} messages")


def test_exercise_3():
    print("\n=== EXERCISE 3: Prompt Templates ===")

    system = build_system_prompt(
        role="Python tutor",
        traits=["patient", "clear"],
        constraints=["under 100 words"]
    )
    print(f"System prompt:\n{system}\n")

    examples = [("2+2", "4"), ("3*5", "15")]
    formatted = format_few_shot_examples(examples)
    print(f"Few-shot:\n{formatted}\n")


def test_exercise_4():
    print("\n=== EXERCISE 4: Configuration ===")

    config = create_llm_config("claude-sonnet-4", mode="creative")
    print(f"Config: {config}")

    base = {"model": "claude", "temp": 0.7}
    override = {"temp": 0.9}
    merged = merge_configs(base, override)
    print(f"Merged: {merged}")

    valid, error = validate_llm_config({"model": "claude", "max_tokens": 1024})
    print(f"Valid: {valid}, Error: {error}")


def test_exercise_5():
    print("\n=== EXERCISE 5: Response Parsing ===")

    text = 'Here is the data: ```json\n{"result": "success"}\n```'
    data = extract_json_from_text(text)
    print(f"Extracted JSON: {data}")

    response = """TOOL: calculator
INPUT: 2 + 2

TOOL: search
INPUT: Python"""

    tools = parse_tool_calls(response)
    print(f"Tool calls: {tools}")


def test_exercise_6():
    print("\n=== EXERCISE 6: Tokens and Context ===")

    text = "Hello world! This is a test."
    tokens = estimate_tokens(text)
    print(f"Estimated tokens: {tokens}")

    messages = [
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 1000}
    ]
    fits = fits_in_context(messages, max_context=1000)
    print(f"Fits in context: {fits}")

    long_text = "word " * 1000
    chunks = split_text_into_chunks(long_text, chunk_size=100, overlap=20)
    print(f"Split into {len(chunks)} chunks")


# Uncomment to run tests:
# test_exercise_1()
# test_exercise_2()
# test_exercise_3()
# test_exercise_4()
# test_exercise_5()
# test_exercise_6()

"""
LEARNING OBJECTIVES CHECKLIST

After completing these exercises, you should be comfortable with:

JSON & API Responses:
- [ ] Parsing JSON strings with json.loads()
- [ ] Navigating nested dict structures
- [ ] Handling missing keys with .get()
- [ ] Error handling with try/except

Message Lists:
- [ ] List comprehensions for filtering
- [ ] Building dicts from structured data
- [ ] Aggregating data (token counts)
- [ ] List slicing and truncation

String Processing:
- [ ] F-strings for templates
- [ ] Multi-line string formatting
- [ ] String methods: split(), strip(), join()
- [ ] Building complex prompts

Configuration:
- [ ] Dict building with conditionals
- [ ] Merging dicts with unpacking
- [ ] Validation logic
- [ ] Literal types for constraints

Practical Skills:
- [ ] Extracting structured data from text
- [ ] Pattern matching in strings
- [ ] Token estimation
- [ ] Text chunking algorithms

These are the core Python skills for LLM applications!
"""
