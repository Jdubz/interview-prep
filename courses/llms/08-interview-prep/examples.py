"""
Module 08: Interview Coding Solutions — Golden Examples

Complete, clean implementations of common interview live-coding challenges.
These are the kind of solutions you would write in an interview: well-structured,
clearly named, with type hints and concise comments. No external dependencies
beyond the standard library and typing.

Each example is self-contained and demonstrates production-quality patterns.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Shared types used across examples
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMConfig:
    model: str = "default"
    temperature: float = 0.0
    max_tokens: int = 1024


class CompletionFn(Protocol):
    """Protocol for LLM completion functions. Provider-agnostic."""
    def __call__(
        self, messages: list[Message], config: LLMConfig,
        tools: list[ToolDefinition] | None = None,
    ) -> Message: ...


# ===========================================================================
# Example 1: RAG Pipeline from Scratch (In-Memory)
# ===========================================================================

@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.

    In production you would chunk by tokens, but character-based
    chunking demonstrates the same principle without a tokenizer.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks


class SimpleRAGPipeline:
    """In-memory RAG pipeline. No external dependencies.

    In a real system, embed_fn would call an embedding API and the
    vector store would be a proper database. This demonstrates the
    architecture and data flow.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        complete_fn: CompletionFn,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.embed_fn = embed_fn
        self.complete_fn = complete_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.chunks: list[Chunk] = []

    def ingest(self, documents: list[Document]) -> int:
        """Chunk, embed, and store documents. Returns chunk count."""
        for doc in documents:
            texts = chunk_text(doc.text, self.chunk_size, self.chunk_overlap)
            for i, text in enumerate(texts):
                chunk = Chunk(
                    doc_id=doc.id,
                    chunk_index=i,
                    text=text,
                    embedding=self.embed_fn(text),
                    metadata={**doc.metadata, "chunk_index": i},
                )
                self.chunks.append(chunk)
        return len(self.chunks)

    def retrieve(self, query: str) -> list[Chunk]:
        """Retrieve the top-K most relevant chunks for a query."""
        query_embedding = self.embed_fn(query)
        scored = [
            (chunk, cosine_similarity(query_embedding, chunk.embedding))
            for chunk in self.chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _score in scored[:self.top_k]]

    def query(self, question: str) -> str:
        """Full RAG pipeline: retrieve context, generate answer."""
        chunks = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[Source: {c.doc_id}, chunk {c.chunk_index}]\n{c.text}"
            for c in chunks
        )
        messages = [
            Message(
                role="system",
                content=(
                    "Answer the user's question using ONLY the provided context. "
                    "If the answer is not in the context, say so. Cite your sources."
                ),
            ),
            Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.complete_fn(messages, LLMConfig(temperature=0.0))
        return response.content


# ===========================================================================
# Example 2: Agent Loop with Tool Execution
# ===========================================================================

@dataclass
class AgentResult:
    response: str
    tool_calls_made: list[dict[str, Any]]
    iterations: int
    total_tokens_estimate: int


class AgentLoop:
    """Production-quality agent loop with error handling and guardrails.

    Demonstrates: tool execution, error recovery, iteration limits,
    and structured result tracking.
    """

    def __init__(
        self,
        complete_fn: CompletionFn,
        tools: list[ToolDefinition],
        tool_handlers: dict[str, Callable[..., str]],
        max_iterations: int = 10,
        system_prompt: str = "You are a helpful assistant with access to tools.",
    ):
        self.complete_fn = complete_fn
        self.tools = tools
        self.tool_handlers = tool_handlers
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call with error handling."""
        handler = self.tool_handlers.get(tool_call.name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_call.name}"})
        try:
            result = handler(**tool_call.arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except TypeError as e:
            return json.dumps({"error": f"Invalid arguments: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def run(self, user_message: str) -> AgentResult:
        """Run the agent loop until completion or max iterations."""
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]
        tool_calls_log: list[dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            response = self.complete_fn(messages, LLMConfig(), tools=self.tools)
            messages.append(response)

            # If no tool calls, the agent is done
            if not response.tool_calls:
                return AgentResult(
                    response=response.content,
                    tool_calls_made=tool_calls_log,
                    iterations=iteration,
                    total_tokens_estimate=sum(len(m.content) // 4 for m in messages),
                )

            # Execute each tool call and append results
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                tool_calls_log.append({
                    "iteration": iteration,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                })
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call.id,
                ))

        # Max iterations reached -- ask model for a summary of what it accomplished
        messages.append(Message(
            role="user",
            content="You have reached the maximum number of tool calls. "
                    "Please summarize what you were able to accomplish.",
        ))
        final = self.complete_fn(messages, LLMConfig())
        return AgentResult(
            response=final.content,
            tool_calls_made=tool_calls_log,
            iterations=iteration,
            total_tokens_estimate=sum(len(m.content) // 4 for m in messages),
        )


# ===========================================================================
# Example 3: Eval Pipeline with LLM-as-Judge
# ===========================================================================

@dataclass
class EvalCase:
    input: str
    expected_output: str
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalScore:
    case: EvalCase
    actual_output: str
    score: float  # 0.0 - 1.0
    reasoning: str
    passed: bool
    latency_ms: float


@dataclass
class EvalReport:
    scores: list[EvalScore]
    total: int
    passed: int
    failed: int
    average_score: float
    average_latency_ms: float
    by_category: dict[str, dict[str, float]]

    def summary(self) -> str:
        lines = [
            f"Eval Report: {self.passed}/{self.total} passed "
            f"({self.passed / self.total * 100:.1f}%)",
            f"Average score: {self.average_score:.3f}",
            f"Average latency: {self.average_latency_ms:.0f}ms",
            "",
            "By category:",
        ]
        for cat, stats in sorted(self.by_category.items()):
            lines.append(
                f"  {cat}: {stats['average_score']:.3f} "
                f"({stats['passed']}/{stats['total']})"
            )
        return "\n".join(lines)


class EvalPipeline:
    """Evaluation pipeline with LLM-as-judge scoring.

    Demonstrates: structured evaluation, scoring rubrics, category-level
    analysis, and reporting.
    """

    JUDGE_PROMPT = """You are evaluating an AI assistant's response.

Input: {input}
Expected output: {expected}
Actual output: {actual}

Score the actual output from 0.0 to 1.0 based on:
- Correctness: Does it convey the same information as the expected output?
- Completeness: Does it cover all key points?
- Conciseness: Is it appropriately detailed without being verbose?

Respond with JSON only:
{{"score": 0.0, "reasoning": "brief explanation"}}"""

    def __init__(
        self,
        system_under_test: CompletionFn,
        judge: CompletionFn,
        pass_threshold: float = 0.7,
    ):
        self.system_under_test = system_under_test
        self.judge = judge
        self.pass_threshold = pass_threshold

    def _run_single(self, case: EvalCase) -> EvalScore:
        """Run a single eval case: generate output, then judge it."""
        start = time.time()
        response = self.system_under_test(
            [Message(role="user", content=case.input)],
            LLMConfig(temperature=0.0),
        )
        latency_ms = (time.time() - start) * 1000
        actual_output = response.content

        # Judge the output
        judge_prompt = self.JUDGE_PROMPT.format(
            input=case.input,
            expected=case.expected_output,
            actual=actual_output,
        )
        judge_response = self.judge(
            [Message(role="user", content=judge_prompt)],
            LLMConfig(temperature=0.0),
        )

        try:
            result = json.loads(judge_response.content)
            score = float(result["score"])
            reasoning = result["reasoning"]
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 0.0
            reasoning = f"Judge returned invalid response: {judge_response.content}"

        return EvalScore(
            case=case,
            actual_output=actual_output,
            score=score,
            reasoning=reasoning,
            passed=score >= self.pass_threshold,
            latency_ms=latency_ms,
        )

    def run(self, cases: list[EvalCase]) -> EvalReport:
        """Run all eval cases and produce a report."""
        scores = [self._run_single(case) for case in cases]

        # Aggregate by category
        by_category: dict[str, dict[str, Any]] = {}
        for s in scores:
            cat = s.case.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "scores": []}
            by_category[cat]["total"] += 1
            by_category[cat]["passed"] += 1 if s.passed else 0
            by_category[cat]["scores"].append(s.score)

        category_stats = {
            cat: {
                "total": data["total"],
                "passed": data["passed"],
                "average_score": sum(data["scores"]) / len(data["scores"]),
            }
            for cat, data in by_category.items()
        }

        return EvalReport(
            scores=scores,
            total=len(scores),
            passed=sum(1 for s in scores if s.passed),
            failed=sum(1 for s in scores if not s.passed),
            average_score=sum(s.score for s in scores) / len(scores) if scores else 0,
            average_latency_ms=(
                sum(s.latency_ms for s in scores) / len(scores) if scores else 0
            ),
            by_category=category_stats,
        )


# ===========================================================================
# Example 4: Streaming Response Handler with Partial JSON Assembly
# ===========================================================================

@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    delta: str
    finish_reason: str | None = None
    tool_call_delta: dict[str, Any] | None = None


class StreamingHandler:
    """Handles streaming LLM responses, assembling text and partial JSON.

    Demonstrates: incremental text assembly, partial JSON buffering,
    tool call reconstruction from streamed fragments, and callback-based
    delivery to the consumer.
    """

    def __init__(
        self,
        on_text: Callable[[str], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
    ):
        self.on_text = on_text or (lambda _: None)
        self.on_complete = on_complete or (lambda _: None)
        self.on_tool_call = on_tool_call or (lambda _: None)
        self._text_buffer: list[str] = []
        self._tool_call_buffers: dict[str, dict[str, Any]] = {}

    def process_chunk(self, chunk: StreamChunk) -> None:
        """Process a single stream chunk."""
        if chunk.delta:
            self._text_buffer.append(chunk.delta)
            self.on_text(chunk.delta)

        if chunk.tool_call_delta:
            tc = chunk.tool_call_delta
            tc_id = tc.get("id", "")

            if tc_id and tc_id not in self._tool_call_buffers:
                self._tool_call_buffers[tc_id] = {
                    "id": tc_id,
                    "name": tc.get("name", ""),
                    "arguments_json": "",
                }

            # Accumulate partial arguments
            if "arguments" in tc:
                for buf in self._tool_call_buffers.values():
                    if buf["id"] == tc_id or (not tc_id and len(self._tool_call_buffers) == 1):
                        buf["arguments_json"] += tc["arguments"]
                        break

        if chunk.finish_reason == "stop":
            full_text = "".join(self._text_buffer)
            self.on_complete(full_text)

        if chunk.finish_reason == "tool_calls":
            for buf in self._tool_call_buffers.values():
                try:
                    args = json.loads(buf["arguments_json"])
                except json.JSONDecodeError:
                    args = {"_raw": buf["arguments_json"]}
                self.on_tool_call(ToolCall(
                    id=buf["id"],
                    name=buf["name"],
                    arguments=args,
                ))

    def process_stream(self, stream: list[StreamChunk]) -> str:
        """Process an entire stream. Returns the full assembled text."""
        for chunk in stream:
            self.process_chunk(chunk)
        return "".join(self._text_buffer)

    @staticmethod
    def try_parse_partial_json(buffer: str) -> dict[str, Any] | None:
        """Attempt to parse partial JSON, returning None if incomplete.

        Useful for showing structured output progressively as it streams.
        """
        try:
            return json.loads(buffer)
        except json.JSONDecodeError:
            # Try closing obvious open structures
            for suffix in ["}", "]}", "\"}", "\"]}"]:
                try:
                    return json.loads(buffer + suffix)
                except json.JSONDecodeError:
                    continue
            return None


# ===========================================================================
# Example 5: Model Router with Cost Tracking
# ===========================================================================

@dataclass
class ModelConfig:
    name: str
    input_cost_per_1m: float  # dollars per 1M input tokens
    output_cost_per_1m: float  # dollars per 1M output tokens
    max_context: int
    avg_latency_ms: float
    capabilities: set[str] = field(default_factory=set)


@dataclass
class RoutingDecision:
    model: str
    reason: str
    estimated_cost: float


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    timestamp: float
    task_type: str


class ModelRouter:
    """Routes requests to the optimal model based on task type and constraints.

    Demonstrates: model selection logic, cost estimation, usage tracking,
    and budget management.
    """

    def __init__(self, models: list[ModelConfig], daily_budget: float = 100.0):
        self.models = {m.name: m for m in models}
        self.daily_budget = daily_budget
        self.usage_log: list[UsageRecord] = []

    def _estimate_cost(
        self, model: ModelConfig, input_tokens: int, output_tokens: int,
    ) -> float:
        return (
            (input_tokens / 1_000_000) * model.input_cost_per_1m
            + (output_tokens / 1_000_000) * model.output_cost_per_1m
        )

    def _daily_spend(self) -> float:
        """Total spend for the current day."""
        today_start = time.time() - (time.time() % 86400)
        return sum(
            r.cost for r in self.usage_log if r.timestamp >= today_start
        )

    def route(
        self,
        task_type: str,
        input_tokens: int,
        estimated_output_tokens: int = 500,
        required_capabilities: set[str] | None = None,
        max_latency_ms: float | None = None,
    ) -> RoutingDecision:
        """Select the optimal model for a request."""
        required_caps = required_capabilities or set()

        # Routing rules by task type
        model_preferences: dict[str, list[str]] = {
            "classification": ["haiku", "gpt-4o-mini", "flash"],
            "extraction": ["haiku", "gpt-4o-mini", "sonnet"],
            "generation": ["sonnet", "gpt-4o", "opus"],
            "reasoning": ["opus", "o1", "sonnet"],
            "embedding": ["embedding-small", "embedding-large"],
        }

        # Budget check — if spending is high, prefer cheaper models
        daily_spend = self._daily_spend()
        budget_pressure = daily_spend / self.daily_budget if self.daily_budget > 0 else 0

        preferred = model_preferences.get(task_type, list(self.models.keys()))

        for model_name in preferred:
            model = self.models.get(model_name)
            if model is None:
                continue

            # Check capabilities
            if not required_caps.issubset(model.capabilities):
                continue

            # Check context window
            if input_tokens + estimated_output_tokens > model.max_context:
                continue

            # Check latency constraint
            if max_latency_ms and model.avg_latency_ms > max_latency_ms:
                continue

            estimated_cost = self._estimate_cost(
                model, input_tokens, estimated_output_tokens,
            )

            # If budget pressure is high, prefer the cheapest option
            if budget_pressure > 0.8 and model_name not in preferred[:2]:
                continue

            return RoutingDecision(
                model=model_name,
                reason=f"Best fit for {task_type} "
                       f"(budget: {budget_pressure:.0%} used)",
                estimated_cost=estimated_cost,
            )

        # Fallback to cheapest available model
        cheapest = min(self.models.values(), key=lambda m: m.input_cost_per_1m)
        return RoutingDecision(
            model=cheapest.name,
            reason="Fallback to cheapest model (no preferred model available)",
            estimated_cost=self._estimate_cost(
                cheapest, input_tokens, estimated_output_tokens,
            ),
        )

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        task_type: str,
    ) -> UsageRecord:
        """Record a completed request for cost tracking."""
        model_config = self.models[model]
        cost = self._estimate_cost(model_config, input_tokens, output_tokens)
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            timestamp=time.time(),
            task_type=task_type,
        )
        self.usage_log.append(record)
        return record

    def cost_report(self) -> dict[str, Any]:
        """Generate a cost report from usage logs."""
        by_model: dict[str, dict[str, float]] = {}
        by_task: dict[str, dict[str, float]] = {}

        for record in self.usage_log:
            # By model
            if record.model not in by_model:
                by_model[record.model] = {"cost": 0, "requests": 0, "tokens": 0}
            by_model[record.model]["cost"] += record.cost
            by_model[record.model]["requests"] += 1
            by_model[record.model]["tokens"] += record.input_tokens + record.output_tokens

            # By task
            if record.task_type not in by_task:
                by_task[record.task_type] = {"cost": 0, "requests": 0}
            by_task[record.task_type]["cost"] += record.cost
            by_task[record.task_type]["requests"] += 1

        return {
            "total_cost": sum(r.cost for r in self.usage_log),
            "total_requests": len(self.usage_log),
            "by_model": by_model,
            "by_task": by_task,
            "daily_budget_remaining": self.daily_budget - self._daily_spend(),
        }


# ===========================================================================
# Example 6: Conversation Memory with Summarization
# ===========================================================================

@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4


class ConversationMemory:
    """Manages conversation history with summarization for long conversations.

    Implements a hybrid strategy: keep recent turns verbatim, summarize
    older turns, always include system prompt. This balances context quality
    with token efficiency.
    """

    def __init__(
        self,
        complete_fn: CompletionFn,
        system_prompt: str,
        max_context_tokens: int = 8000,
        recent_turns_to_keep: int = 10,
        summarize_every_n_turns: int = 5,
    ):
        self.complete_fn = complete_fn
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.recent_turns_to_keep = recent_turns_to_keep
        self.summarize_every_n_turns = summarize_every_n_turns

        self.all_turns: list[ConversationTurn] = []
        self.summary: str = ""
        self._turns_since_last_summary = 0

    def add_turn(self, role: str, content: str) -> None:
        """Add a turn and trigger summarization if needed."""
        self.all_turns.append(ConversationTurn(role=role, content=content))
        self._turns_since_last_summary += 1

        if self._turns_since_last_summary >= self.summarize_every_n_turns:
            self._update_summary()

    def _update_summary(self) -> None:
        """Summarize older turns into a running summary."""
        if len(self.all_turns) <= self.recent_turns_to_keep:
            return

        older_turns = self.all_turns[:-self.recent_turns_to_keep]
        turns_text = "\n".join(
            f"{t.role}: {t.content}" for t in older_turns[-self.summarize_every_n_turns:]
        )

        prompt = (
            f"Current summary of conversation so far:\n{self.summary}\n\n"
            f"New turns to incorporate:\n{turns_text}\n\n"
            "Produce an updated summary that captures all key information, "
            "decisions, and context from the conversation. Be concise but "
            "preserve important details. Output only the summary."
            if self.summary
            else f"Summarize this conversation so far:\n{turns_text}\n\n"
                 "Capture key information, decisions, and context. Be concise."
        )

        response = self.complete_fn(
            [Message(role="user", content=prompt)],
            LLMConfig(temperature=0.0, max_tokens=300),
        )
        self.summary = response.content
        self._turns_since_last_summary = 0

    def get_messages(self) -> list[Message]:
        """Build the message list for an LLM call.

        Structure:
        1. System prompt (always)
        2. Summary of older conversation (if exists)
        3. Recent turns verbatim
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
        ]

        if self.summary:
            messages.append(Message(
                role="system",
                content=f"Summary of earlier conversation:\n{self.summary}",
            ))

        recent = self.all_turns[-self.recent_turns_to_keep:]
        for turn in recent:
            messages.append(Message(role=turn.role, content=turn.content))

        # Check token budget and trim if necessary
        total_tokens = sum(len(m.content) // 4 for m in messages)
        while total_tokens > self.max_context_tokens and len(messages) > 2:
            # Remove the oldest non-system message
            for i, msg in enumerate(messages):
                if msg.role != "system":
                    total_tokens -= len(msg.content) // 4
                    messages.pop(i)
                    break

        return messages

    def get_stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        messages = self.get_messages()
        return {
            "total_turns": len(self.all_turns),
            "recent_turns": min(len(self.all_turns), self.recent_turns_to_keep),
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary),
            "context_tokens_estimate": sum(len(m.content) // 4 for m in messages),
            "max_context_tokens": self.max_context_tokens,
        }


# ===========================================================================
# Example usage / demonstration
# ===========================================================================

def demo_rag_pipeline():
    """Demonstrate the RAG pipeline with a simple hash-based mock embedding."""

    # Mock embedding: deterministic hash-based vectors for demonstration
    def mock_embed(text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]

    # Mock LLM completion
    def mock_complete(messages, config, tools=None):
        context = messages[-1].content
        return Message(role="assistant", content=f"Based on the context, here is my answer.")

    # Build pipeline
    pipeline = SimpleRAGPipeline(
        embed_fn=mock_embed,
        complete_fn=mock_complete,
        chunk_size=200,
        top_k=2,
    )

    # Ingest documents
    docs = [
        Document(id="refunds", text="Our refund policy allows returns within 30 days. "
                 "Items must be in original condition. Digital products are non-refundable. "
                 "Refunds are processed within 5-7 business days."),
        Document(id="shipping", text="We offer free shipping on orders over $50. "
                 "Standard shipping takes 5-7 business days. Express shipping is available "
                 "for $12.99 and arrives in 2-3 business days."),
    ]
    chunk_count = pipeline.ingest(docs)
    print(f"Ingested {len(docs)} documents into {chunk_count} chunks")

    # Query
    answer = pipeline.query("What is the refund policy?")
    print(f"Answer: {answer}")


def demo_model_router():
    """Demonstrate the model router with cost tracking."""
    models = [
        ModelConfig("haiku", 0.80, 4.00, 200_000, 200, {"text"}),
        ModelConfig("sonnet", 3.00, 15.00, 200_000, 500, {"text", "vision", "tools"}),
        ModelConfig("opus", 15.00, 75.00, 200_000, 1000, {"text", "vision", "tools", "reasoning"}),
        ModelConfig("gpt-4o-mini", 0.15, 0.60, 128_000, 300, {"text", "tools"}),
    ]

    router = ModelRouter(models, daily_budget=50.0)

    # Route different task types
    for task in ["classification", "generation", "reasoning"]:
        decision = router.route(task, input_tokens=1000)
        print(f"{task}: {decision.model} (est. ${decision.estimated_cost:.4f}) — {decision.reason}")

    # Record some usage
    router.record_usage("haiku", 1000, 200, 180, "classification")
    router.record_usage("sonnet", 3000, 800, 520, "generation")

    report = router.cost_report()
    print(f"\nCost report: ${report['total_cost']:.4f} total, "
          f"${report['daily_budget_remaining']:.2f} remaining")


if __name__ == "__main__":
    print("=== RAG Pipeline Demo ===")
    demo_rag_pipeline()
    print("\n=== Model Router Demo ===")
    demo_model_router()
