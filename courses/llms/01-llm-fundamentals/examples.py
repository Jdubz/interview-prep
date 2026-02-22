"""
LLM Fundamentals -- Production-Ready Code Patterns
====================================================
These examples demonstrate core LLM engineering concepts using Python.
The focus is on LLM patterns, not Python syntax.

Requirements:
    pip install openai tiktoken numpy
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# 1. TOKEN COUNTING AND COST ESTIMATION
# ---------------------------------------------------------------------------
# Tokens are the atomic unit of LLM billing. Accurate token counting lets you
# predict costs, stay within context windows, and optimize prompts.

import tiktoken


@dataclass
class ModelPricing:
    """Pricing per 1M tokens for a specific model."""
    name: str
    input_cost_per_million: float
    output_cost_per_million: float
    context_window: int
    max_output_tokens: int
    encoding_name: str = "cl100k_base"  # tokenizer encoding


# Approximate pricing as of early 2025 -- check provider pricing pages
MODEL_CATALOG: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        name="gpt-4o",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    "gpt-4o-mini": ModelPricing(
        name="gpt-4o-mini",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        name="claude-sonnet-4-20250514",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window=200_000,
        max_output_tokens=16_384,
    ),
    "claude-haiku-3-5": ModelPricing(
        name="claude-haiku-3-5",
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
        context_window=200_000,
        max_output_tokens=8_192,
    ),
}


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a string using tiktoken.

    tiktoken is OpenAI's fast BPE tokenizer. The encoding determines the
    vocabulary -- cl100k_base is used by GPT-4/4o, o200k_base by newer models.
    For Claude/Gemini, this gives an approximation (they use different tokenizers).
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def estimate_cost(
    input_text: str,
    estimated_output_tokens: int,
    model_name: str = "gpt-4o",
) -> dict[str, Any]:
    """Estimate the cost of an LLM API call.

    Key insight: output tokens are typically 3-5x more expensive than input tokens.
    Reducing output length (via max_tokens, stop sequences, concise instructions)
    is often the most impactful cost optimization.
    """
    model = MODEL_CATALOG[model_name]
    input_tokens = count_tokens(input_text, model.encoding_name)

    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_million
    output_cost = (estimated_output_tokens / 1_000_000) * model.output_cost_per_million
    total_cost = input_cost + output_cost

    context_usage = (input_tokens + estimated_output_tokens) / model.context_window

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "total_tokens": input_tokens + estimated_output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "context_utilization": f"{context_usage:.1%}",
        "fits_in_context": (input_tokens + estimated_output_tokens) <= model.context_window,
    }


def compare_model_costs(input_text: str, estimated_output_tokens: int) -> None:
    """Compare costs across all models for the same request.

    This is the kind of analysis you'd do when choosing a model tier for a
    production feature -- find the cheapest model that meets quality requirements.
    """
    print(f"Input: {count_tokens(input_text)} tokens | Output: {estimated_output_tokens} tokens\n")
    print(f"{'Model':<30} {'Input $':>10} {'Output $':>10} {'Total $':>10}")
    print("-" * 62)

    for model_name in MODEL_CATALOG:
        result = estimate_cost(input_text, estimated_output_tokens, model_name)
        print(
            f"{model_name:<30} "
            f"${result['input_cost_usd']:>9.6f} "
            f"${result['output_cost_usd']:>9.6f} "
            f"${result['total_cost_usd']:>9.6f}"
        )


# ---------------------------------------------------------------------------
# 2. MODEL SELECTION / ROUTING
# ---------------------------------------------------------------------------
# In production, you route requests to different models based on task complexity.
# A smart router can cut costs 10x while maintaining quality.


class TaskComplexity(Enum):
    """Task complexity tiers that map to model selection.

    The key insight: most production traffic is simple tasks that don't need
    frontier models. Routing 80% of traffic to a cheap model and 20% to a
    capable model is a common pattern.
    """
    SIMPLE = "simple"           # Classification, extraction, yes/no
    MODERATE = "moderate"       # Summarization, general Q&A, formatting
    COMPLEX = "complex"         # Multi-step reasoning, code generation, analysis
    REASONING = "reasoning"     # Math proofs, logic puzzles, complex planning


@dataclass
class RoutingDecision:
    model: str
    complexity: TaskComplexity
    reason: str
    estimated_cost_per_request: float


# Heuristic signals for complexity classification
COMPLEXITY_SIGNALS: dict[str, list[str]] = {
    "reasoning": [
        "step by step", "prove", "derive", "mathematical", "logic puzzle",
        "analyze this code", "debug", "optimize this algorithm",
    ],
    "complex": [
        "compare and contrast", "write a", "generate code", "explain in detail",
        "design a system", "multi-step", "analyze",
    ],
    "simple": [
        "classify", "extract", "yes or no", "true or false", "which category",
        "sentiment", "translate this word", "what is the",
    ],
}


def classify_task_complexity(prompt: str) -> TaskComplexity:
    """Classify a prompt's complexity using keyword heuristics.

    In production, you might use a small classifier model or a set of rules
    based on your domain. The goal is to route expensive reasoning tasks to
    capable models while keeping simple tasks on cheap models.
    """
    prompt_lower = prompt.lower()

    # Check for reasoning signals first (highest priority)
    for signal in COMPLEXITY_SIGNALS["reasoning"]:
        if signal in prompt_lower:
            return TaskComplexity.REASONING

    for signal in COMPLEXITY_SIGNALS["simple"]:
        if signal in prompt_lower:
            return TaskComplexity.SIMPLE

    for signal in COMPLEXITY_SIGNALS["complex"]:
        if signal in prompt_lower:
            return TaskComplexity.COMPLEX

    # Default to moderate for ambiguous cases
    return TaskComplexity.MODERATE


# Model routing table: maps complexity to the appropriate model tier
MODEL_ROUTING: dict[TaskComplexity, str] = {
    TaskComplexity.SIMPLE: "gpt-4o-mini",           # $0.15/$0.60 per 1M
    TaskComplexity.MODERATE: "gpt-4o-mini",          # Still cheap enough
    TaskComplexity.COMPLEX: "gpt-4o",                # $2.50/$10.00 per 1M
    TaskComplexity.REASONING: "claude-sonnet-4-20250514",  # Best reasoning value
}


def route_request(prompt: str) -> RoutingDecision:
    """Route a request to the appropriate model based on complexity.

    This is a simplified version of what you'd build in production. Real
    routers might also consider: user tier, latency requirements, current
    rate limits, A/B test assignments, and cost budgets.
    """
    complexity = classify_task_complexity(prompt)
    model = MODEL_ROUTING[complexity]
    pricing = MODEL_CATALOG[model]

    # Rough cost estimate assuming 500 output tokens
    est_input_tokens = count_tokens(prompt)
    est_output_tokens = 500
    est_cost = (
        (est_input_tokens / 1_000_000) * pricing.input_cost_per_million
        + (est_output_tokens / 1_000_000) * pricing.output_cost_per_million
    )

    return RoutingDecision(
        model=model,
        complexity=complexity,
        reason=f"Classified as {complexity.value} -> routed to {model}",
        estimated_cost_per_request=round(est_cost, 6),
    )


# ---------------------------------------------------------------------------
# 3. CONTEXT WINDOW MANAGEMENT
# ---------------------------------------------------------------------------
# When your content exceeds the context window, you need truncation strategies.
# The approach depends on what information is most important.


class TruncationStrategy(Enum):
    """Different strategies for fitting content into context windows.

    Each strategy makes different tradeoffs:
    - KEEP_START: preserves system prompt and initial instructions
    - KEEP_END: preserves the most recent context (conversation history)
    - KEEP_BOTH: preserves instructions (start) and recent context (end)
    - SUMMARIZE: compresses middle content (requires an LLM call)
    """
    KEEP_START = "keep_start"
    KEEP_END = "keep_end"
    KEEP_BOTH = "keep_both"


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    strategy: TruncationStrategy = TruncationStrategy.KEEP_BOTH,
    encoding_name: str = "cl100k_base",
) -> str:
    """Truncate text to fit within a token budget.

    This operates at the token level, not character level, because context
    windows are measured in tokens. Character-level truncation can cut tokens
    in half, producing invalid inputs.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    if strategy == TruncationStrategy.KEEP_START:
        truncated = tokens[:max_tokens]

    elif strategy == TruncationStrategy.KEEP_END:
        truncated = tokens[-max_tokens:]

    elif strategy == TruncationStrategy.KEEP_BOTH:
        # Keep first 70% and last 30% -- preserves system prompt and recent context
        start_budget = int(max_tokens * 0.7)
        end_budget = max_tokens - start_budget
        truncated = tokens[:start_budget] + tokens[-end_budget:]

    return encoding.decode(truncated)


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


def manage_conversation_context(
    messages: list[Message],
    system_prompt: str,
    max_context_tokens: int,
    output_token_budget: int = 2000,
    encoding_name: str = "cl100k_base",
) -> list[Message]:
    """Manage conversation history to fit within context window.

    Strategy: Always keep the system prompt and the latest user message.
    Trim older conversation history as needed, keeping the most recent turns.
    This preserves the model's instructions and the user's current request.

    This is the "sliding window" approach to conversation management --
    similar to how a chat application keeps only recent messages in view.
    """
    encoding = tiktoken.get_encoding(encoding_name)

    # Reserve space for system prompt, latest message, and output
    system_tokens = len(encoding.encode(system_prompt))
    latest_msg_tokens = len(encoding.encode(messages[-1].content))
    reserved = system_tokens + latest_msg_tokens + output_token_budget

    available_for_history = max_context_tokens - reserved

    if available_for_history <= 0:
        # Only room for system prompt and latest message
        return [
            Message(role="system", content=system_prompt),
            messages[-1],
        ]

    # Fill history from most recent to oldest (recency bias)
    history_messages: list[Message] = []
    tokens_used = 0

    for msg in reversed(messages[:-1]):  # Exclude the latest message
        msg_tokens = len(encoding.encode(msg.content)) + 4  # +4 for role/formatting overhead
        if tokens_used + msg_tokens > available_for_history:
            break
        history_messages.insert(0, msg)
        tokens_used += msg_tokens

    return [
        Message(role="system", content=system_prompt),
        *history_messages,
        messages[-1],
    ]


# ---------------------------------------------------------------------------
# 4. PARAMETER TUNING EXAMPLES
# ---------------------------------------------------------------------------
# Different tasks require different parameter configurations. These presets
# show how parameters affect output characteristics.


@dataclass
class GenerationConfig:
    """Configuration for LLM generation parameters.

    The key insight: temperature and top_p both control randomness but in
    different ways. Temperature scales the entire distribution; top_p truncates
    it. Don't set both aggressively -- pick one to tune.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)

    def to_openai_params(self) -> dict[str, Any]:
        """Convert to OpenAI API parameters."""
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.frequency_penalty != 0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0:
            params["presence_penalty"] = self.presence_penalty
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        return params


# Pre-built configurations for common use cases
GENERATION_PRESETS: dict[str, GenerationConfig] = {
    # Deterministic output: always the same answer for the same input.
    # Use for: classification, data extraction, structured output.
    "deterministic": GenerationConfig(
        temperature=0,
        top_p=1.0,
        max_tokens=1024,
    ),

    # Focused but with slight variation.
    # Use for: code generation, factual Q&A, technical writing.
    "precise": GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        max_tokens=4096,
    ),

    # Balanced creativity and coherence.
    # Use for: general conversation, summaries, documentation.
    "balanced": GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    ),

    # High creativity with diversity encouragement.
    # Use for: brainstorming, creative writing, generating alternatives.
    "creative": GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
        presence_penalty=0.6,  # Encourage topic diversity
    ),

    # Short, constrained output for classification tasks.
    # Use for: sentiment analysis, intent detection, yes/no questions.
    "classifier": GenerationConfig(
        temperature=0,
        top_p=1.0,
        max_tokens=50,
        stop_sequences=["\n"],  # Stop after first line
    ),

    # JSON extraction with deterministic output.
    # Use for: structured data extraction, form filling.
    "json_extraction": GenerationConfig(
        temperature=0,
        top_p=1.0,
        max_tokens=2048,
        stop_sequences=["```"],  # Stop after JSON block
    ),
}


def get_config_for_task(task_description: str) -> GenerationConfig:
    """Select a generation config based on task type.

    In production, you might determine this from the endpoint being called,
    the tool being used, or metadata in the request.
    """
    task_lower = task_description.lower()

    if any(w in task_lower for w in ["classify", "extract", "parse", "label"]):
        return GENERATION_PRESETS["deterministic"]
    elif any(w in task_lower for w in ["code", "sql", "technical", "factual"]):
        return GENERATION_PRESETS["precise"]
    elif any(w in task_lower for w in ["creative", "brainstorm", "story", "poem"]):
        return GENERATION_PRESETS["creative"]
    elif any(w in task_lower for w in ["json", "structured", "schema"]):
        return GENERATION_PRESETS["json_extraction"]
    else:
        return GENERATION_PRESETS["balanced"]


# ---------------------------------------------------------------------------
# 5. EMBEDDING SIMILARITY COMPARISON
# ---------------------------------------------------------------------------
# Embeddings map text to vectors where semantic similarity = geometric proximity.
# This is the foundation of semantic search, RAG, and clustering.


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    cos_sim(A, B) = (A . B) / (||A|| * ||B||)

    Returns a value between -1 and 1 (in practice, 0 to 1 for most embedding models).
    Higher values = more similar meaning.
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def find_most_similar(
    query_embedding: list[float],
    document_embeddings: list[tuple[str, list[float]]],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Find the top-k most similar documents to a query.

    This is the core operation in RAG retrieval: given a user's query embedding,
    find the most relevant document chunks. In production, you'd use a vector
    database (Pinecone, Qdrant, pgvector) instead of brute-force search.

    Brute force is O(n * d) where n = number of documents, d = embedding dimension.
    Vector databases use ANN algorithms (HNSW, IVF) for O(log n) approximate search.
    """
    similarities: list[tuple[str, float]] = []

    for doc_id, doc_embedding in document_embeddings:
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_id, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def demonstrate_embedding_similarity() -> None:
    """Demonstrate embedding concepts with mock vectors.

    In a real application, you'd call an embedding API:
        openai.embeddings.create(model="text-embedding-3-small", input=text)

    The key concept: embeddings capture semantic meaning, not lexical overlap.
    "How do I reset my password?" and "I forgot my login credentials" are
    semantically similar despite sharing zero keywords.
    """
    # Mock embeddings (in reality, these would be 1536-3072 dimensional)
    # Using 8 dimensions for illustration
    mock_embeddings = {
        "password_reset": [0.8, 0.1, 0.3, -0.2, 0.5, 0.1, -0.3, 0.4],
        "forgot_login":   [0.7, 0.2, 0.4, -0.1, 0.4, 0.2, -0.2, 0.3],
        "pizza_nyc":      [-0.3, 0.8, -0.1, 0.5, -0.2, 0.6, 0.1, -0.4],
        "billing_issue":  [0.5, 0.3, 0.6, -0.3, 0.2, 0.1, -0.1, 0.5],
        "account_locked": [0.6, 0.2, 0.5, -0.2, 0.3, 0.1, -0.2, 0.4],
    }

    query = mock_embeddings["password_reset"]
    documents = [
        (name, emb) for name, emb in mock_embeddings.items()
        if name != "password_reset"
    ]

    print("Query: 'How do I reset my password?'")
    print("Finding most similar documents:\n")

    results = find_most_similar(query, documents, top_k=3)
    for doc_id, similarity in results:
        print(f"  {doc_id:<20} similarity: {similarity:.4f}")

    # Expected output: forgot_login and account_locked rank highest
    # because they're semantically related to password/account issues.
    # pizza_nyc ranks lowest because it's completely unrelated.


# ---------------------------------------------------------------------------
# 6. STREAMING RESPONSE HANDLER
# ---------------------------------------------------------------------------
# Streaming returns tokens as they're generated via Server-Sent Events (SSE).
# This reduces perceived latency -- users see output immediately instead of
# waiting for the full response.

# Note: This requires the openai package. Uncomment and install to run.
# from openai import OpenAI


@dataclass
class StreamMetrics:
    """Track streaming performance metrics.

    TTFT (Time to First Token) is the key UX metric for streaming.
    Users perceive the application as fast if the first token appears quickly,
    even if total generation takes seconds.
    """
    time_to_first_token_ms: float = 0.0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    tokens_per_second: float = 0.0


def stream_response_with_metrics(
    prompt: str,
    model: str = "gpt-4o",
    config: GenerationConfig | None = None,
) -> tuple[str, StreamMetrics]:
    """Stream a response and collect performance metrics.

    In production, you'd pipe this into an SSE endpoint (FastAPI StreamingResponse,
    Express res.write, etc.) to stream tokens directly to the client.

    Key architecture insight: streaming doesn't reduce total latency -- it reduces
    *perceived* latency. The model still generates the same number of tokens at
    the same speed. But the user sees output appearing progressively.
    """
    # This is a reference implementation. Uncomment the openai import above to run.
    # Showing the pattern rather than requiring a live API key.

    """
    from openai import OpenAI
    client = OpenAI()

    if config is None:
        config = GENERATION_PRESETS["balanced"]

    start_time = time.monotonic()
    first_token_time = None
    full_response = []
    token_count = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        **config.to_openai_params(),
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token_text = chunk.choices[0].delta.content

            if first_token_time is None:
                first_token_time = time.monotonic()

            full_response.append(token_text)
            token_count += 1

            # In production: yield this to the client via SSE
            print(token_text, end="", flush=True)

    end_time = time.monotonic()
    total_ms = (end_time - start_time) * 1000
    ttft_ms = ((first_token_time or end_time) - start_time) * 1000

    metrics = StreamMetrics(
        time_to_first_token_ms=round(ttft_ms, 1),
        total_duration_ms=round(total_ms, 1),
        total_tokens=token_count,
        tokens_per_second=round(token_count / (total_ms / 1000), 1) if total_ms > 0 else 0,
    )

    return "".join(full_response), metrics
    """

    # Simulated response for demonstration purposes
    print("[Streaming simulation -- install openai package and set API key for live demo]")
    return "", StreamMetrics()


# ---------------------------------------------------------------------------
# 7. TOKEN BUDGET PLANNER
# ---------------------------------------------------------------------------
# Plan how to allocate your context window budget across components.


@dataclass
class TokenBudget:
    """Plan context window allocation for a request.

    The context window is a fixed resource shared between:
    - System prompt (instructions, persona, constraints)
    - Conversation history (previous turns)
    - Retrieved context (RAG documents)
    - User message (current query)
    - Output (model's response)

    This planner helps you stay within limits and allocate efficiently.
    """
    model: str
    system_prompt_tokens: int = 0
    history_tokens: int = 0
    retrieved_context_tokens: int = 0
    user_message_tokens: int = 0
    reserved_output_tokens: int = 2000

    @property
    def total_input_tokens(self) -> int:
        return (
            self.system_prompt_tokens
            + self.history_tokens
            + self.retrieved_context_tokens
            + self.user_message_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.reserved_output_tokens

    @property
    def context_window(self) -> int:
        return MODEL_CATALOG[self.model].context_window

    @property
    def remaining_tokens(self) -> int:
        return self.context_window - self.total_tokens

    def fits(self) -> bool:
        return self.total_tokens <= self.context_window

    def summary(self) -> str:
        lines = [
            f"Token Budget for {self.model} (context: {self.context_window:,})",
            "-" * 50,
            f"  System prompt:      {self.system_prompt_tokens:>8,} tokens",
            f"  Conversation history:{self.history_tokens:>7,} tokens",
            f"  Retrieved context:  {self.retrieved_context_tokens:>8,} tokens",
            f"  User message:       {self.user_message_tokens:>8,} tokens",
            f"  Output budget:      {self.reserved_output_tokens:>8,} tokens",
            "-" * 50,
            f"  Total:              {self.total_tokens:>8,} tokens",
            f"  Remaining:          {self.remaining_tokens:>8,} tokens",
            f"  Utilization:        {self.total_tokens / self.context_window:>8.1%}",
            f"  Fits in context:    {'YES' if self.fits() else 'NO -- EXCEEDS LIMIT'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEMO / MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LLM Fundamentals -- Code Examples")
    print("=" * 60)

    # --- Token counting and cost estimation ---
    print("\n1. TOKEN COUNTING AND COST ESTIMATION")
    print("-" * 40)

    sample_text = (
        "You are a helpful customer support agent for Acme Corp. "
        "Answer questions about our products, pricing, and policies. "
        "Be concise and professional."
    )
    print(f"Text: {sample_text[:60]}...")
    print(f"Token count: {count_tokens(sample_text)}")
    print()
    compare_model_costs(sample_text, estimated_output_tokens=500)

    # --- Model routing ---
    print("\n\n2. MODEL ROUTING")
    print("-" * 40)

    test_prompts = [
        "Classify this email as spam or not spam: 'You won a prize!'",
        "Summarize this article about climate change in 3 bullet points.",
        "Write a Python function to implement a binary search tree with balancing.",
        "Prove that the square root of 2 is irrational. Show your work step by step.",
    ]

    for prompt in test_prompts:
        decision = route_request(prompt)
        print(f"\nPrompt: {prompt[:70]}...")
        print(f"  -> {decision.reason}")
        print(f"  -> Est. cost: ${decision.estimated_cost_per_request:.6f}")

    # --- Context window management ---
    print("\n\n3. CONTEXT WINDOW MANAGEMENT")
    print("-" * 40)

    long_text = "This is a sentence about artificial intelligence. " * 200
    for strategy in TruncationStrategy:
        truncated = truncate_to_token_limit(long_text, max_tokens=100, strategy=strategy)
        print(f"\n  Strategy: {strategy.value}")
        print(f"  Original tokens: {count_tokens(long_text)}")
        print(f"  Truncated tokens: {count_tokens(truncated)}")
        print(f"  Preview: {truncated[:80]}...")

    # --- Parameter configs ---
    print("\n\n4. PARAMETER CONFIGURATIONS")
    print("-" * 40)

    tasks = [
        "Classify this sentiment",
        "Write a creative story",
        "Generate SQL query",
        "Extract JSON from this text",
    ]

    for task in tasks:
        config = get_config_for_task(task)
        print(f"\n  Task: {task}")
        print(f"  -> temperature={config.temperature}, top_p={config.top_p}, "
              f"max_tokens={config.max_tokens}")

    # --- Embedding similarity ---
    print("\n\n5. EMBEDDING SIMILARITY")
    print("-" * 40)
    demonstrate_embedding_similarity()

    # --- Token budget planner ---
    print("\n\n6. TOKEN BUDGET PLANNER")
    print("-" * 40)

    budget = TokenBudget(
        model="gpt-4o",
        system_prompt_tokens=500,
        history_tokens=3000,
        retrieved_context_tokens=8000,
        user_message_tokens=200,
        reserved_output_tokens=2000,
    )
    print(budget.summary())


if __name__ == "__main__":
    main()
