"""
Prompt Engineering Examples

Annotated Python examples demonstrating core prompt engineering techniques.
Uses a generic LLM interface — swap in any provider (OpenAI, Anthropic, etc.).

PYTHON CONCEPTS DEMONSTRATED:
- dataclasses: Clean data structures without boilerplate (@dataclass decorator)
- Protocols: Structural typing for flexible interfaces (like TypeScript interfaces)
- Type hints: list[str], dict, Literal, async/await
- Async functions: All LLM calls are async (non-blocking I/O)
- F-strings: Dynamic prompt building
- List comprehensions: Data transformation
- Pattern matching with conditionals

This file is meant to be READ, not run. To make it runnable, implement the
CompletionFn protocol with your chosen LLM provider (OpenAI, Anthropic, etc.).
"""

from dataclasses import dataclass  # Eliminates class boilerplate
from typing import Protocol, Literal  # Type system features
import json  # For parsing/generating JSON


# ---------------------------------------------------------------------------
# Generic LLM Interface
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMConfig:
    """Minimal configuration for an LLM call."""
    model: str
    temperature: float = 1.0
    max_tokens: int | None = None
    response_format: Literal["json_object", "text"] | None = None


class CompletionFn(Protocol):
    """
    Generic completion function — implement with your provider of choice.

    PYTHON NOTE: Protocol = structural typing (like TypeScript interfaces).
    Any object with a __call__ method matching this signature will work.
    No inheritance required! This makes the code provider-agnostic.

    Example implementations:
    - class OpenAICompletion: async def __call__(...) -> str: ...
    - class AnthropicCompletion: async def __call__(...) -> str: ...
    Both work as CompletionFn without explicitly inheriting from it.
    """
    async def __call__(
        self, messages: list[Message], config: LLMConfig
    ) -> str: ...


# ---------------------------------------------------------------------------
# Example 1: Zero-Shot Classification
# ---------------------------------------------------------------------------

async def zero_shot_classify(
    complete: CompletionFn, ticket_text: str
) -> str:
    """
    Classify a support ticket with no examples — just clear instructions.
    Works well for common categories the model has seen in training.

    PYTHON PATTERN: async def + await
    - All LLM calls should be async (they're network I/O)
    - Call with: result = await zero_shot_classify(...)
    - Or run with: asyncio.run(zero_shot_classify(...))
    """
    # PYTHON: Build list of Message objects (dataclass instances)
    # Using multi-line string in parentheses for clean formatting
    messages = [
        Message(
            role="system",  # Literal type enforces valid values
            content=(  # Parentheses allow multi-line string without \
                "You are a support ticket classifier. Classify each ticket "
                "into exactly one category:\n"
                "- billing\n- technical\n- account\n- general\n\n"
                "Respond with ONLY the category name, nothing else."
            ),
        ),
        Message(role="user", content=ticket_text),
    ]

    # PYTHON: await = wait for async function to complete
    # Returns a string (the classification category)
    return await complete(messages, LLMConfig(model="gpt-4o", temperature=0))


# ---------------------------------------------------------------------------
# Example 2: Few-Shot Classification
# ---------------------------------------------------------------------------

async def few_shot_classify(
    complete: CompletionFn, ticket_text: str
) -> str:
    """
    When zero-shot isn't reliable enough, few-shot examples anchor the model's
    understanding of each category — especially useful for ambiguous cases.
    """
    messages = [
        Message(
            role="system",
            content="Classify support tickets. Respond with only the category.",
        ),
        # Few-shot examples as user/assistant pairs
        Message(role="user", content="I was charged twice for my subscription this month"),
        Message(role="assistant", content="billing"),
        Message(role="user", content="The app crashes whenever I try to upload a file larger than 5MB"),
        Message(role="assistant", content="technical"),
        Message(role="user", content="I need to change the email address on my account"),
        Message(role="assistant", content="account"),
        Message(role="user", content="What are your business hours?"),
        Message(role="assistant", content="general"),
        # Actual input
        Message(role="user", content=ticket_text),
    ]

    return await complete(messages, LLMConfig(model="gpt-4o", temperature=0))


# ---------------------------------------------------------------------------
# Example 3: Chain-of-Thought Reasoning
# ---------------------------------------------------------------------------

@dataclass
class ReasonedAnswer:
    reasoning: str
    answer: str
    confidence: Literal["high", "medium", "low"]


async def chain_of_thought(
    complete: CompletionFn, question: str
) -> ReasonedAnswer:
    """
    Force the model to reason step-by-step before answering.
    Dramatically improves accuracy on logic/math/multi-step problems.
    """
    messages = [
        Message(
            role="system",
            content=(
                "You are an analytical assistant. For every question:\n"
                "1. Break down the problem step by step\n"
                "2. Show your reasoning explicitly\n"
                "3. State your final answer\n"
                "4. Rate your confidence\n\n"
                "Respond in JSON:\n"
                '{\n  "reasoning": "step-by-step explanation",\n'
                '  "answer": "final answer",\n'
                '  "confidence": "high" | "medium" | "low"\n}'
            ),
        ),
        Message(role="user", content=question),
    ]

    # Call LLM with JSON output format enforced
    raw = await complete(
        messages,
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
    )

    # PYTHON: Parse JSON string to dict
    data = json.loads(raw)  # '{"reasoning": "...", ...}' -> {"reasoning": "...", ...}

    # PYTHON: **data unpacks dict as keyword arguments
    # Equivalent to: ReasonedAnswer(reasoning=data["reasoning"], answer=data["answer"], ...)
    return ReasonedAnswer(**data)


# ---------------------------------------------------------------------------
# Example 4: Structured Data Extraction with Delimiters
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntities:
    people: list[str]
    organizations: list[str]
    dates: list[str]
    locations: list[str]
    topics: list[str]


async def extract_entities(
    complete: CompletionFn, document: str
) -> ExtractedEntities:
    """
    Extract structured data from unstructured text.
    Uses XML-style delimiters to clearly separate instructions from data,
    and enforces JSON output for programmatic consumption.
    """
    messages = [
        Message(
            role="system",
            content=(
                "Extract named entities from the provided document.\n"
                "Return a JSON object with these keys:\n"
                "- people: array of person names\n"
                "- organizations: array of org names\n"
                "- dates: array of dates (ISO 8601 format when possible)\n"
                "- locations: array of place names\n"
                "- topics: array of key topics/subjects\n\n"
                "If a category has no entities, return an empty array.\n"
                "Return ONLY valid JSON."
            ),
        ),
        Message(
            role="user",
            # Delimiters prevent document content from being confused with instructions
            content=f"<document>\n{document}\n</document>\n\nExtract all entities from the document above.",
        ),
    ]

    raw = await complete(
        messages,
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
    )
    data = json.loads(raw)
    return ExtractedEntities(**data)


# ---------------------------------------------------------------------------
# Example 5: Prompt Chaining — Multi-Step Pipeline
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    key_points: list[str]
    summary: str
    quality_issues: list[str]


async def chained_summarization(
    complete: CompletionFn, document: str
) -> SummaryResult:
    """
    Complex tasks are more reliable when broken into discrete steps.
    Each step has a focused prompt, and outputs feed into the next step.

    This pipeline: Document → Key Points → Quality Check → Final Summary
    """
    # Step 1: Extract key points
    key_points_raw = await complete(
        [
            Message(
                role="system",
                content=(
                    "Extract the 5-7 most important points from the document. "
                    "Return as JSON: { \"points\": [\"...\", ...] }"
                ),
            ),
            Message(role="user", content=f"<document>\n{document}\n</document>"),
        ],
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
    )
    key_points: list[str] = json.loads(key_points_raw)["points"]

    # Step 2: Quality check — are these points well-supported?
    quality_raw = await complete(
        [
            Message(
                role="system",
                content=(
                    "Review these key points against the source document.\n"
                    "Flag any points that are:\n"
                    "- Not directly supported by the text\n"
                    "- Misleading or missing important nuance\n"
                    "- Redundant with other points\n\n"
                    'Return JSON: { "issues": ["..."], "approved_points": ["..."] }'
                ),
            ),
            Message(
                role="user",
                content=(
                    f"<document>\n{document}\n</document>\n\n"
                    f"<key_points>\n{json.dumps(key_points, indent=2)}\n</key_points>"
                ),
            ),
        ],
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
    )
    quality = json.loads(quality_raw)

    # Step 3: Generate final summary from approved points only
    bullet_points = "\n".join(f"- {p}" for p in quality["approved_points"])
    summary = await complete(
        [
            Message(
                role="system",
                content=(
                    "Write a concise 3-4 sentence summary based ONLY on the "
                    "provided key points. Do not add information not in the points."
                ),
            ),
            Message(role="user", content=f"Key points:\n{bullet_points}"),
        ],
        LLMConfig(model="gpt-4o", temperature=0.3),  # Slight creativity for natural prose
    )

    return SummaryResult(
        key_points=quality["approved_points"],
        summary=summary,
        quality_issues=quality["issues"],
    )


# ---------------------------------------------------------------------------
# Example 6: Role Prompting + Output Structuring
# ---------------------------------------------------------------------------

@dataclass
class CodeIssue:
    severity: Literal["critical", "warning", "info"]
    line: int | None
    description: str
    suggestion: str


@dataclass
class CodeReviewResult:
    issues: list[CodeIssue]
    overall_assessment: str


async def review_code(
    complete: CompletionFn, code: str, language: str
) -> CodeReviewResult:
    """
    Combine role prompting with structured output for a code review assistant.
    The role constrains the domain expertise; the structure ensures parseability.
    """
    messages = [
        Message(
            role="system",
            content=(
                f"You are a senior {language} engineer performing a security-focused code review.\n\n"
                "Focus on:\n"
                "1. Security vulnerabilities (injection, XSS, auth issues)\n"
                "2. Bug-prone patterns (race conditions, null dereferences, off-by-one)\n"
                "3. Performance issues (N+1 queries, memory leaks, unnecessary allocations)\n\n"
                "Do NOT comment on:\n"
                "- Code style or formatting\n"
                "- Naming conventions (unless misleading)\n"
                "- Missing comments or documentation\n\n"
                "Return JSON:\n"
                "{\n"
                '  "issues": [\n'
                '    { "severity": "critical"|"warning"|"info", "line": <int|null>,\n'
                '      "description": "what\'s wrong", "suggestion": "how to fix it" }\n'
                "  ],\n"
                '  "overall_assessment": "1-2 sentence summary"\n'
                "}\n\n"
                "If there are no issues, return an empty issues array."
            ),
        ),
        Message(role="user", content=f"```{language}\n{code}\n```"),
    ]

    raw = await complete(
        messages,
        LLMConfig(model="gpt-4o", temperature=0, response_format="json_object"),
    )
    data = json.loads(raw)

    # PYTHON: List comprehension to convert list of dicts to list of dataclass objects
    # [CodeIssue(**issue) for issue in data["issues"]]
    # Equivalent to:
    #   issues = []
    #   for issue in data["issues"]:
    #       issues.append(CodeIssue(**issue))
    return CodeReviewResult(
        issues=[CodeIssue(**issue) for issue in data["issues"]],
        overall_assessment=data["overall_assessment"],
    )


# ---------------------------------------------------------------------------
# Example 7: Iterative Refinement
# ---------------------------------------------------------------------------

@dataclass
class RefinementResult:
    final_output: str
    iterations: int


async def iterative_refinement(
    complete: CompletionFn, task: str, max_iterations: int = 2
) -> RefinementResult:
    """
    Use the model to critique and improve its own output.
    This pattern is the foundation of self-consistency and constitutional AI.
    """
    # Step 1: Initial generation
    current_output = await complete(
        [Message(role="user", content=task)],
        LLMConfig(model="gpt-4o", temperature=0.7),
    )

    for i in range(max_iterations):
        # Step 2: Self-critique
        critique = await complete(
            [
                Message(
                    role="system",
                    content=(
                        "You are a critical reviewer. Evaluate this output for:\n"
                        "- Accuracy and correctness\n"
                        "- Completeness (anything missing?)\n"
                        "- Clarity and conciseness\n"
                        "- Any errors or misleading statements\n\n"
                        'If the output is good enough, respond with exactly "APPROVED".\n'
                        "Otherwise, list specific improvements needed."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Task: {task}\n\nOutput to review:\n{current_output}",
                ),
            ],
            LLMConfig(model="gpt-4o", temperature=0),
        )

        if critique.strip() == "APPROVED":
            return RefinementResult(final_output=current_output, iterations=i + 1)

        # Step 3: Refine based on critique
        current_output = await complete(
            [
                Message(
                    role="user",
                    content=(
                        f"Original task: {task}\n\n"
                        f"Your previous output:\n{current_output}\n\n"
                        f"Feedback:\n{critique}\n\n"
                        "Please provide an improved version addressing all feedback."
                    ),
                ),
            ],
            LLMConfig(model="gpt-4o", temperature=0.5),
        )

    return RefinementResult(final_output=current_output, iterations=max_iterations)


# ---------------------------------------------------------------------------
# Example 8: Dynamic Prompt Assembly
# ---------------------------------------------------------------------------

@dataclass
class PromptContext:
    user_role: Literal["admin", "user", "guest"]
    features: list[str]
    previous_errors: list[str] | None = None
    output_format: Literal["json", "markdown", "text"] = "text"


def build_dynamic_prompt(
    base_instruction: str, context: PromptContext
) -> list[Message]:
    """
    In production, prompts are rarely static strings. This shows a pattern for
    building prompts dynamically based on context — a common real-world need.
    """
    system_parts: list[str] = []

    # Base instruction
    system_parts.append(base_instruction)

    # Role-based constraints
    role_constraints = {
        "admin": "The user has full access. You may discuss system internals.",
        "user": "The user has standard access. Do not expose system internals or admin features.",
        "guest": "The user has limited access. Only discuss publicly available features.",
    }
    system_parts.append(role_constraints[context.user_role])

    # Feature-specific instructions
    if context.features:
        features_str = ", ".join(context.features)
        system_parts.append(
            f"Available features: {features_str}. Only reference these features."
        )

    # Learn from previous errors
    if context.previous_errors:
        error_list = "\n".join(f"- {e}" for e in context.previous_errors)
        system_parts.append(
            f"IMPORTANT: Previous responses had these issues. Avoid repeating them:\n{error_list}"
        )

    # Output format
    format_instructions = {
        "json": "Respond with valid JSON only. No markdown, no explanation.",
        "markdown": "Respond in well-formatted markdown.",
        "text": "Respond in plain text.",
    }
    system_parts.append(format_instructions[context.output_format])

    return [Message(role="system", content="\n\n".join(system_parts))]


# ---------------------------------------------------------------------------
# Usage Examples (illustrative, not runnable without a provider)
# ---------------------------------------------------------------------------

async def demo(complete: CompletionFn) -> None:
    # Zero-shot
    category = await zero_shot_classify(
        complete, "My payment failed but the order still went through"
    )
    print("Category:", category)  # "billing"

    # Chain-of-thought
    answer = await chain_of_thought(
        complete,
        "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, "
        "what's the average speed for the whole trip?",
    )
    print("Answer:", answer)
    # ReasonedAnswer(reasoning="Total distance = 200km, ...", answer="80 km/h", confidence="high")

    # Entity extraction
    entities = await extract_entities(
        complete,
        "Tim Cook announced at Apple's Cupertino headquarters on March 15, 2024 "
        "that the company would invest $2B in AI research.",
    )
    print("Entities:", entities)

    # Code review
    review = await review_code(
        complete,
        'app.get("/user/:id", (req, res) => {\n'
        '  const query = "SELECT * FROM users WHERE id = " + req.params.id;\n'
        "  db.query(query).then(user => res.json(user));\n"
        "});",
        "typescript",
    )
    print("Review:", review)
    # issues: [CodeIssue(severity="critical", description="SQL injection via string concatenation", ...)]
