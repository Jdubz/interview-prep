"""
Module 02: Prompt Engineering — Production-Ready Patterns

Complete, runnable examples demonstrating core prompt engineering techniques.
Each pattern is designed for real production use cases.

Requirements:
    pip install openai anthropic pydantic

These examples use OpenAI's API by default. Swap the client and model
for Anthropic or any OpenAI-compatible endpoint.
"""

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

client = AsyncOpenAI()
DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# 1. Classification with Few-Shot Examples and Structured Output
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Pydantic model enforcing the output schema for classification."""
    category: str
    confidence: str  # "high", "medium", "low"
    reasoning: str


CLASSIFICATION_SYSTEM_PROMPT = """\
You are a support ticket classifier. Classify each ticket into exactly one category.

Categories:
- billing: payment issues, charges, invoices, refunds, subscription management
- technical: bugs, errors, crashes, performance problems, feature malfunctions
- account: login, password, profile, settings, permissions
- general: questions, feedback, feature requests, anything else

Few-shot examples:

Ticket: "I was charged $49.99 but my plan is $29.99"
Category: billing
Confidence: high
Reasoning: Explicit mention of charges and pricing discrepancy.

Ticket: "App freezes for 10 seconds when I open the dashboard"
Category: technical
Confidence: high
Reasoning: Performance issue with specific reproducible behavior.

Ticket: "I can't reset my password — the email never arrives"
Category: account
Confidence: high
Reasoning: Password reset flow failure, core account access issue.

Ticket: "Can you add dark mode?"
Category: general
Confidence: high
Reasoning: Feature request, not an issue with existing functionality.

Ticket: "I want to cancel my account and get a refund for this month"
Category: billing
Confidence: medium
Reasoning: Involves both account cancellation and refund. Billing takes priority because the refund is the actionable item.

Return your response as JSON with keys: category, confidence, reasoning.
"""


async def classify_ticket(ticket_text: str) -> ClassificationResult:
    """
    Classify a support ticket using few-shot prompting with structured output.

    Uses OpenAI's json_schema response format for guaranteed valid JSON.
    The few-shot examples in the system prompt establish the pattern for
    edge cases (like the cancel+refund example showing how to handle ambiguity).
    """
    response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,  # Deterministic for classification consistency
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": ClassificationResult.model_json_schema(),
            },
        },
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Ticket: {ticket_text}"},
        ],
    )

    return ClassificationResult.model_validate_json(
        response.choices[0].message.content
    )


# ---------------------------------------------------------------------------
# 2. Entity Extraction with JSON Schema Enforcement
# ---------------------------------------------------------------------------

class Person(BaseModel):
    name: str
    role: str | None = None


class ExtractionResult(BaseModel):
    """Schema for entity extraction. Enforced at the API level."""
    people: list[Person]
    organizations: list[str]
    dates: list[str]
    monetary_values: list[str]
    locations: list[str]


EXTRACTION_SYSTEM_PROMPT = """\
Extract all entities from the provided document. Follow these rules:
- Only extract explicitly stated information. Do not infer.
- Dates must be in ISO 8601 format (YYYY-MM-DD).
- Monetary values in original currency with symbol (e.g., $50M, EUR 1,200).
- If a field has no entities, use an empty array.
- For people, include their role/title if mentioned.

The document will be wrapped in <document> tags.
"""


async def extract_entities(text: str) -> ExtractionResult:
    """
    Extract structured entities from unstructured text.

    Uses XML delimiter tags to clearly separate the document from instructions.
    The json_schema response format guarantees the output matches our Pydantic model.
    """
    response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entity_extraction",
                "strict": True,
                "schema": ExtractionResult.model_json_schema(),
            },
        },
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<document>\n{text}\n</document>",
            },
        ],
    )

    return ExtractionResult.model_validate_json(
        response.choices[0].message.content
    )


# ---------------------------------------------------------------------------
# 3. Multi-Step Prompt Chain: Analyze -> Plan -> Execute
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    analysis: str
    plan: str
    execution: str


async def analyze_and_improve_code(code: str) -> ChainResult:
    """
    Three-step prompt chain for code improvement.

    Step 1 (Analyze): Identify issues — uses a critical, analytical persona.
    Step 2 (Plan): Create improvement plan — uses the analysis as input.
    Step 3 (Execute): Apply the plan — produces the final improved code.

    Each step is a focused prompt that does one thing well. Different steps
    could use different models (e.g., cheaper model for analysis, stronger
    for code generation).
    """

    # Step 1: Analyze — identify issues in the code
    analysis_response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer. Identify bugs, security "
                    "issues, performance problems, and maintainability concerns. "
                    "List each issue with severity (critical/high/medium/low) "
                    "and a brief explanation. Be thorough but concise."
                ),
            },
            {
                "role": "user",
                "content": f"Review this code:\n\n```\n{code}\n```",
            },
        ],
    )
    analysis = analysis_response.choices[0].message.content

    # Step 2: Plan — create an improvement plan based on the analysis
    # The analysis output becomes input to this step
    plan_response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a software architect. Given a code review, create "
                    "a prioritized improvement plan. For each change: state what "
                    "to change, why, and the expected impact. Order by priority "
                    "(critical fixes first)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original code:\n```\n{code}\n```\n\n"
                    f"Code review findings:\n{analysis}\n\n"
                    "Create an improvement plan."
                ),
            },
        ],
    )
    plan = plan_response.choices[0].message.content

    # Step 3: Execute — apply the plan to produce improved code
    execution_response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.1,  # Slight variation for code generation
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior engineer implementing code improvements. "
                    "Apply the improvement plan to the original code. Return "
                    "only the improved code with brief inline comments explaining "
                    "each change. Do not include explanatory text outside the code."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original code:\n```\n{code}\n```\n\n"
                    f"Improvement plan:\n{plan}\n\n"
                    "Implement all improvements."
                ),
            },
        ],
    )
    execution = execution_response.choices[0].message.content

    return ChainResult(analysis=analysis, plan=plan, execution=execution)


# ---------------------------------------------------------------------------
# 4. Summarization with Configurable Detail Level
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATE = """\
Summarize the following text at the "{detail_level}" detail level.

Detail levels:
- brief: 1-2 sentences capturing the single most important point
- standard: 3-5 bullet points covering key information
- detailed: comprehensive summary with sections for key findings,
  implications, and action items (8-12 bullet points)

Audience: {audience}

<text>
{text}
</text>
"""


async def summarize(
    text: str,
    detail_level: str = "standard",
    audience: str = "technical team",
) -> str:
    """
    Summarization with configurable detail and audience.

    The template uses named placeholders for detail level and audience,
    demonstrating prompt templating. The detail level descriptions in the
    prompt itself ensure the model understands the expected output length
    and structure for each level.
    """
    prompt = SUMMARY_TEMPLATE.format(
        text=text,
        detail_level=detail_level,
        audience=audience,
    )

    response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.3,  # Low creativity, but not fully deterministic
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. Prompt Template System with Variable Injection
# ---------------------------------------------------------------------------

class PromptTemplate:
    """
    A minimal prompt template system for production use.

    Separates the template (version-controlled, reviewed) from the variables
    (runtime data). Supports validation of required variables and optional
    system prompts.
    """

    def __init__(
        self,
        template: str,
        system_prompt: str | None = None,
        required_vars: list[str] | None = None,
    ):
        self.template = template
        self.system_prompt = system_prompt
        # Auto-detect required variables from {placeholders} in the template
        self.required_vars = required_vars or re.findall(
            r"\{(\w+)\}", template
        )

    def render(self, **kwargs: Any) -> list[dict[str, str]]:
        """Render the template with variables. Returns messages list."""
        missing = set(self.required_vars) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        messages = []
        if self.system_prompt:
            rendered_system = self.system_prompt.format(**kwargs)
            messages.append({"role": "system", "content": rendered_system})

        rendered_user = self.template.format(**kwargs)
        messages.append({"role": "user", "content": rendered_user})

        return messages

    async def execute(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str:
        """Render and execute the prompt in one call."""
        messages = self.render(**kwargs)
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        return response.choices[0].message.content


# Pre-built templates for common tasks
TEMPLATES = {
    "classify": PromptTemplate(
        system_prompt=(
            "Classify the input into exactly one of these categories: "
            "{categories}. Respond with ONLY the category name."
        ),
        template="{input}",
        required_vars=["categories", "input"],
    ),
    "extract_json": PromptTemplate(
        system_prompt=(
            "Extract information from the input. Return valid JSON "
            "matching this schema:\n{schema}\n\n"
            "Only extract explicitly stated information."
        ),
        template="<document>\n{input}\n</document>",
        required_vars=["schema", "input"],
    ),
    "qa_with_context": PromptTemplate(
        system_prompt=(
            "Answer the question using ONLY the provided context. "
            "If the answer is not in the context, say so. "
            "Cite sources using [Source N] notation."
        ),
        template=(
            "<context>\n{context}\n</context>\n\n"
            "Question: {question}"
        ),
        required_vars=["context", "question"],
    ),
}


# ---------------------------------------------------------------------------
# 6. Self-Consistency: Multiple Samples + Majority Voting
# ---------------------------------------------------------------------------

COT_PROMPT_TEMPLATE = """\
{question}

Think through this step by step, showing your reasoning.
After your analysis, state your final answer on the last line as:
ANSWER: <your answer>
"""


def extract_answer(response_text: str) -> str | None:
    """Extract the final answer from a CoT response."""
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return the last non-empty line
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else None


async def self_consistency(
    question: str,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Self-consistency implementation: sample multiple CoT paths, majority vote.

    Key design decisions:
    - temperature > 0 is required to get diverse reasoning paths
    - Samples are sent in parallel to minimize latency
    - The voting mechanism uses simple majority (ties broken by first occurrence)
    - Returns both the answer and the full distribution for transparency

    In production, you would also want:
    - Confidence threshold (reject if no answer gets >50% of votes)
    - Logging of all samples for debugging
    - Cost monitoring (N samples = N * cost)
    """
    prompt = COT_PROMPT_TEMPLATE.format(question=question)

    # Send all samples in parallel for minimum latency
    tasks = [
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for _ in range(n_samples)
    ]
    responses = await asyncio.gather(*tasks)

    # Extract answers from each reasoning path
    answers = []
    reasoning_paths = []
    for r in responses:
        text = r.choices[0].message.content
        reasoning_paths.append(text)
        answer = extract_answer(text)
        if answer:
            answers.append(answer)

    # Majority vote
    if not answers:
        return {
            "answer": None,
            "confidence": 0.0,
            "distribution": {},
            "n_samples": n_samples,
            "n_valid": 0,
        }

    vote_counts = Counter(answers)
    winner, winner_count = vote_counts.most_common(1)[0]

    return {
        "answer": winner,
        "confidence": winner_count / len(answers),
        "distribution": dict(vote_counts),
        "n_samples": n_samples,
        "n_valid": len(answers),
        "reasoning_paths": reasoning_paths,  # For debugging
    }


# ---------------------------------------------------------------------------
# 7. Retry-with-Feedback for Structured Output
# ---------------------------------------------------------------------------

async def extract_with_retry(
    text: str,
    schema: type[BaseModel],
    max_retries: int = 3,
) -> BaseModel | None:
    """
    Extract structured data with automatic retry on validation failure.

    When the model produces invalid output, this feeds the validation error
    back to the model so it can self-correct. This is more effective than
    a blind retry because the model sees exactly what went wrong.

    Pattern:
    1. Send extraction prompt
    2. Validate response against Pydantic schema
    3. On failure: send the original prompt + previous response + error message
    4. Repeat up to max_retries

    In production, also consider:
    - Logging each attempt for monitoring prompt quality
    - Alerting if retry rate exceeds a threshold (signals prompt needs improvement)
    - Falling back to a simpler schema or manual review after max retries
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "Extract structured data from the provided text. "
                "Return valid JSON matching this schema:\n\n"
                f"{schema_json}\n\n"
                "Return ONLY the JSON object. No other text."
            ),
        },
        {
            "role": "user",
            "content": f"<document>\n{text}\n</document>",
        },
    ]

    for attempt in range(max_retries):
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            messages=messages,
        )

        raw_output = response.choices[0].message.content

        try:
            # Attempt to parse the JSON and validate against the schema
            # Strip markdown code fences if the model wraps the JSON
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)

            parsed = json.loads(cleaned)
            return schema.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries - 1:
                return None  # Give up after max retries

            # Feed the error back to the model for self-correction
            messages.extend([
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"Your response was not valid. Error:\n{e}\n\n"
                        "Please fix the issue and return valid JSON "
                        "matching the required schema."
                    ),
                },
            ])

    return None


# ---------------------------------------------------------------------------
# Usage Examples
# ---------------------------------------------------------------------------

async def main():
    """Demonstrate each pattern."""

    # 1. Classification
    print("=== Classification ===")
    result = await classify_ticket(
        "The checkout page shows a 500 error when I try to pay with PayPal"
    )
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print()

    # 2. Entity Extraction
    print("=== Entity Extraction ===")
    entities = await extract_entities(
        "Sarah Chen, CTO of Acme Corp, announced a $50M Series C "
        "on March 15, 2024 at their San Francisco headquarters."
    )
    print(f"People: {entities.people}")
    print(f"Organizations: {entities.organizations}")
    print(f"Dates: {entities.dates}")
    print(f"Values: {entities.monetary_values}")
    print()

    # 3. Multi-step chain
    print("=== Code Analysis Chain ===")
    chain_result = await analyze_and_improve_code(
        "def get_user(id):\n"
        "    query = f'SELECT * FROM users WHERE id = {id}'\n"
        "    return db.execute(query)\n"
    )
    print(f"Analysis:\n{chain_result.analysis[:200]}...")
    print(f"\nPlan:\n{chain_result.plan[:200]}...")
    print(f"\nImproved code:\n{chain_result.execution[:200]}...")
    print()

    # 4. Summarization
    print("=== Summarization ===")
    summary = await summarize(
        "Artificial intelligence has transformed the software industry...",
        detail_level="brief",
        audience="executive leadership",
    )
    print(f"Summary: {summary}")
    print()

    # 5. Template system
    print("=== Template System ===")
    result = await TEMPLATES["classify"].execute(
        categories="positive, negative, neutral",
        input="This product exceeded my expectations. Best purchase this year!",
    )
    print(f"Classification: {result}")
    print()

    # 6. Self-consistency
    print("=== Self-Consistency ===")
    sc_result = await self_consistency(
        "A store has 15 apples. They sell 6 in the morning, receive a "
        "shipment of 20 in the afternoon, then sell 8 more. How many "
        "apples do they have at the end of the day?",
        n_samples=5,
    )
    print(f"Answer: {sc_result['answer']}")
    print(f"Confidence: {sc_result['confidence']:.0%}")
    print(f"Vote distribution: {sc_result['distribution']}")
    print()

    # 7. Retry with feedback
    print("=== Retry with Feedback ===")

    class ProductReview(BaseModel):
        product_name: str
        sentiment: str
        key_features: list[str]
        rating_out_of_5: float

    review_result = await extract_with_retry(
        "I love the new AirPods Pro 2! The noise cancellation is incredible "
        "and the battery lasts all day. Sound quality is top-notch. 4.5/5.",
        schema=ProductReview,
    )
    if review_result:
        print(f"Extracted: {review_result.model_dump_json(indent=2)}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
