"""
Agents & Tool Use Examples

Python patterns for tool definitions, an agent loop,
and multi-step execution. Provider-agnostic interfaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Literal, Callable, Awaitable


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


@dataclass
class ParameterSchema:
    type: str
    description: str
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """The model's response — either text, tool calls, or both."""
    text: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolCompletionFn(Protocol):
    """Generic completion function that supports tool use."""
    async def __call__(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        config: dict[str, Any],
    ) -> ModelResponse: ...


# A function that executes a tool and returns a result.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Example Tool Definitions
# ---------------------------------------------------------------------------

customer_support_tools: list[ToolDefinition] = [
    ToolDefinition(
        name="lookup_customer",
        description=(
            "Look up a customer by email or ID. Returns customer profile including "
            "name, plan, and account status. Use this when you need to identify who "
            "you're talking to."
        ),
        parameters={
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer's email address"},
                "customer_id": {"type": "string", "description": "Customer ID (format: CUST-XXXXX)"},
            },
            "required": [],
        },
    ),
    ToolDefinition(
        name="search_orders",
        description=(
            "Search for a customer's orders. Returns a list of orders with status, "
            "items, and dates. Requires a customer ID — look up the customer first "
            "if you only have their email."
        ),
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer ID"},
                "status": {
                    "type": "string",
                    "description": "Filter by order status",
                    "enum": ["pending", "shipped", "delivered", "cancelled", "returned"],
                },
                "limit": {"type": "integer", "description": "Max results to return (default: 5)"},
            },
            "required": ["customer_id"],
        },
    ),
    ToolDefinition(
        name="get_order_details",
        description=(
            "Get detailed information about a specific order including items, "
            "shipping, and payment. Use when the customer asks about a specific order."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID (format: ORD-XXXXX)"},
            },
            "required": ["order_id"],
        },
    ),
    ToolDefinition(
        name="initiate_refund",
        description=(
            "Initiate a refund for an order. This is a WRITE operation — only use when "
            "the customer explicitly requests a refund and you've confirmed the order "
            "details. Returns a refund confirmation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to refund"},
                "reason": {
                    "type": "string",
                    "description": "Reason for the refund",
                    "enum": ["defective", "wrong_item", "not_as_described", "changed_mind", "other"],
                },
                "amount": {
                    "type": "integer",
                    "description": "Refund amount in cents. Omit for full refund.",
                },
            },
            "required": ["order_id", "reason"],
        },
    ),
    ToolDefinition(
        name="search_knowledge_base",
        description=(
            "Search the help center knowledge base for articles about policies, "
            "features, and common questions. Use this to find accurate information "
            "before answering policy questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query — be specific"},
            },
            "required": ["query"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool Executor Factory
# ---------------------------------------------------------------------------

def create_tool_executor(
    handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]],
) -> ToolExecutor:
    """
    In production, this dispatches to real APIs/databases.
    Here we show the dispatch pattern with mock data.
    """
    async def executor(name: str, args: dict[str, Any]) -> Any:
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await handler(args)
        except Exception as e:
            # Structured errors help the model recover
            return {
                "error": str(e),
                "suggestion": "Try a different approach or ask the user for more details.",
            }

    return executor


# Mock handlers
async def _lookup_customer(args: dict[str, Any]) -> dict:
    return {
        "customer_id": "CUST-12345",
        "name": "Jane Smith",
        "email": args.get("email", "jane@example.com"),
        "plan": "pro",
        "status": "active",
        "since": "2023-06-15",
    }


async def _search_orders(args: dict[str, Any]) -> dict:
    return {
        "orders": [
            {"order_id": "ORD-98765", "date": "2024-03-01", "status": "delivered", "total": 4999, "items": ["Widget Pro"]},
            {"order_id": "ORD-98770", "date": "2024-03-10", "status": "shipped", "total": 2999, "items": ["Widget Mini"]},
        ]
    }


async def _get_order_details(args: dict[str, Any]) -> dict:
    return {
        "order_id": args.get("order_id"),
        "status": "delivered",
        "items": [{"name": "Widget Pro", "quantity": 1, "price": 4999}],
        "shipping": {"carrier": "UPS", "delivered": "2024-03-05"},
        "payment": {"method": "visa_4242", "total": 4999},
    }


async def _initiate_refund(args: dict[str, Any]) -> dict:
    return {
        "refund_id": "REF-55555",
        "order_id": args.get("order_id"),
        "amount": args.get("amount", 4999),
        "status": "processing",
        "estimated_completion": "3-5 business days",
    }


async def _search_knowledge_base(args: dict[str, Any]) -> dict:
    return {
        "results": [
            {
                "title": "Refund Policy",
                "excerpt": "Full refunds available within 30 days of delivery. Partial refunds after 30 days at our discretion.",
                "url": "/help/refund-policy",
            }
        ]
    }


tool_executor = create_tool_executor({
    "lookup_customer": _lookup_customer,
    "search_orders": _search_orders,
    "get_order_details": _get_order_details,
    "initiate_refund": _initiate_refund,
    "search_knowledge_base": _search_knowledge_base,
})


# ---------------------------------------------------------------------------
# The Agent Loop
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    iteration: int
    tool_call: ToolCall | None = None
    tool_result: Any = None
    response: str | None = None


@dataclass
class AgentOptions:
    max_iterations: int
    model: str
    on_before_tool_call: Callable[[ToolCall], Awaitable[bool]] | None = None
    on_step: Callable[[AgentStep], None] | None = None


@dataclass
class AgentResult:
    response: str
    steps: list[AgentStep]


async def run_agent(
    initial_messages: list[Message],
    tools: list[ToolDefinition],
    execute: ToolExecutor,
    complete: ToolCompletionFn,
    options: AgentOptions,
) -> AgentResult:
    """
    Core agent loop: send messages to the model, execute any requested tools,
    repeat until the model returns a text response or we hit max iterations.
    """
    messages = list(initial_messages)
    steps: list[AgentStep] = []

    for i in range(options.max_iterations):
        # 1. Call the model
        result = await complete(messages, tools, {"model": options.model, "temperature": 0})

        # 2. If the model returns text (no tool calls), we're done
        if not result.tool_calls:
            response = result.text or "I wasn't able to complete the task."
            step = AgentStep(iteration=i, response=response)
            steps.append(step)
            if options.on_step:
                options.on_step(step)
            return AgentResult(response=response, steps=steps)

        # 3. Execute each tool call
        for tool_call in result.tool_calls:
            # Optional: check permissions before executing
            if options.on_before_tool_call:
                allowed = await options.on_before_tool_call(tool_call)
                if not allowed:
                    messages.append(Message(
                        role="tool",
                        content=json.dumps({"error": "Action blocked by policy. Please try a different approach."}),
                        tool_call_id=tool_call.id,
                    ))
                    continue

            # Execute the tool
            tool_result = await execute(tool_call.name, tool_call.arguments)

            # Record the step
            step = AgentStep(iteration=i, tool_call=tool_call, tool_result=tool_result)
            steps.append(step)
            if options.on_step:
                options.on_step(step)

            # Add the assistant's tool call and the result to the conversation
            messages.append(Message(
                role="assistant",
                content=json.dumps({"tool_calls": [{"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}]}),
            ))
            messages.append(Message(
                role="tool",
                content=json.dumps(tool_result),
                tool_call_id=tool_call.id,
            ))

    # Hit max iterations — ask the model for a final response without tools
    messages.append(Message(
        role="system",
        content="You've reached the maximum number of tool calls. Please provide your best response with the information gathered so far.",
    ))
    final_result = await complete(messages, [], {"model": options.model})

    response = final_result.text or "I wasn't able to fully complete the task with the available steps."
    steps.append(AgentStep(iteration=options.max_iterations, response=response))

    return AgentResult(response=response, steps=steps)


# ---------------------------------------------------------------------------
# ReAct Pattern Implementation
# ---------------------------------------------------------------------------

async def run_react_agent(
    user_query: str,
    tools: list[ToolDefinition],
    execute: ToolExecutor,
    complete: ToolCompletionFn,
    max_iterations: int = 10,
    model: str = "gpt-4o",
) -> AgentResult:
    """
    ReAct wraps the agent loop with explicit reasoning.
    The system prompt instructs the model to think before acting.
    """
    system_prompt = (
        "You are a helpful customer support agent. You have access to tools to "
        "look up customer information, orders, and company policies.\n\n"
        "When answering a question:\n"
        "1. THINK about what information you need\n"
        "2. Use the appropriate tool(s) to gather that information\n"
        "3. THINK about whether you have enough information to answer\n"
        "4. Either use more tools or provide your final answer\n\n"
        "Always verify information with tools before making claims. "
        "If you're unsure about a policy, search the knowledge base.\n\n"
        "Be concise and helpful. If you can't resolve the issue, explain what "
        "you've found and suggest next steps."
    )

    return await run_agent(
        initial_messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_query),
        ],
        tools=tools,
        execute=execute,
        complete=complete,
        options=AgentOptions(max_iterations=max_iterations, model=model),
    )


# ---------------------------------------------------------------------------
# Tool Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_tool_call(
    call: ToolCall, tools: list[ToolDefinition]
) -> ValidationResult:
    """
    Validate a tool call against its schema before execution.
    Catches malformed requests before they hit your backend.
    """
    tool = next((t for t in tools if t.name == call.name), None)
    errors: list[str] = []

    if tool is None:
        return ValidationResult(valid=False, errors=[f"Unknown tool: {call.name}"])

    # Check required parameters
    for param in tool.parameters.get("required", []):
        if param not in call.arguments:
            errors.append(f"Missing required parameter: {param}")

    # Check for unexpected parameters
    known_params = set(tool.parameters.get("properties", {}).keys())
    for param in call.arguments:
        if param not in known_params:
            errors.append(f"Unexpected parameter: {param}")

    # Check enum values
    properties = tool.parameters.get("properties", {})
    for key, value in call.arguments.items():
        schema = properties.get(key, {})
        if "enum" in schema and isinstance(value, str) and value not in schema["enum"]:
            errors.append(
                f'Invalid value for {key}: "{value}". '
                f'Expected one of: {", ".join(schema["enum"])}'
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ---------------------------------------------------------------------------
# Permission-Gated Tool Execution
# ---------------------------------------------------------------------------

PermissionLevel = Literal["read", "write", "admin"]

TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    "lookup_customer": "read",
    "search_orders": "read",
    "get_order_details": "read",
    "search_knowledge_base": "read",
    "initiate_refund": "write",
}

CONFIRM_REQUIRED = {"initiate_refund"}

_PERMISSION_ORDER: list[PermissionLevel] = ["read", "write", "admin"]


def create_gated_executor(
    inner_executor: ToolExecutor,
    user_permission: PermissionLevel,
    confirm_fn: Callable[[ToolCall], Awaitable[bool]] | None = None,
) -> ToolExecutor:
    """
    Create a permission-gated tool executor that checks authorization
    and optionally requires human confirmation for sensitive actions.
    """
    async def gated(name: str, args: dict[str, Any]) -> Any:
        required = TOOL_PERMISSIONS.get(name, "admin")
        if _PERMISSION_ORDER.index(user_permission) < _PERMISSION_ORDER.index(required):
            return {
                "error": f"Insufficient permissions for {name}. Required: {required}, have: {user_permission}"
            }

        if name in CONFIRM_REQUIRED and confirm_fn:
            confirmed = await confirm_fn(ToolCall(id="pending", name=name, arguments=args))
            if not confirmed:
                return {"error": "Action cancelled by user."}

        return await inner_executor(name, args)

    return gated


# ---------------------------------------------------------------------------
# Multi-Agent Routing (Simplified)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    tools: list[ToolDefinition]


async def route_to_agent(
    user_message: str,
    agents: list[AgentConfig],
    complete: ToolCompletionFn,
) -> str:
    """
    Route a user message to the appropriate specialized agent.
    The router uses a cheap, fast model to classify intent.
    """
    agent_descriptions = "\n".join(
        f"- {a.name}: {a.system_prompt[:100]}..." for a in agents
    )

    result = await complete(
        [
            Message(
                role="system",
                content=(
                    "You are a router. Based on the user's message, decide which "
                    "agent should handle it.\n\n"
                    f"Available agents:\n{agent_descriptions}\n\n"
                    "Respond with ONLY the agent name, nothing else."
                ),
            ),
            Message(role="user", content=user_message),
        ],
        [],  # No tools — just text classification
        {"model": "gpt-4o-mini", "temperature": 0},
    )

    return (result.text or agents[0].name).strip()


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

async def demo(complete: ToolCompletionFn) -> None:
    result = await run_react_agent(
        "Hi, I'm jane@example.com. I received my Widget Pro order but it's defective. "
        "Can I get a refund?",
        customer_support_tools,
        tool_executor,
        complete,
    )

    print("Response:", result.response)
    for step in result.steps:
        if step.tool_call:
            print(f"  Tool: {step.tool_call.name} → {step.tool_result}")

    # Expected flow:
    # 1. lookup_customer(email="jane@example.com") → customer profile
    # 2. search_orders(customer_id="CUST-12345") → order list
    # 3. search_knowledge_base(query="refund policy") → policy info
    # 4. initiate_refund(order_id="ORD-98765", reason="defective") → refund confirmation
    # 5. Final response summarizing the refund
