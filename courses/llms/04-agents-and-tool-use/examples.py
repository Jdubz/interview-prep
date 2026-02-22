"""
Module 04: Agents & Tool Use — Complete, Runnable Patterns

These examples demonstrate core agent patterns using protocol-based
abstractions. Replace the LLM call with your provider of choice.

All examples are self-contained and focus on the agent/tool concepts,
not Python specifics.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Shared types used across all examples
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool result messages


class LLMClient(Protocol):
    """Protocol for any LLM provider. Implement this for OpenAI, Anthropic, etc."""
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message: ...


# ---------------------------------------------------------------------------
# Example 1: Complete Agent Loop with Tool Execution
# ---------------------------------------------------------------------------

# Tool registry: maps tool names to Python functions
TOOL_REGISTRY: dict[str, Any] = {}


def register_tool(func):
    """Decorator to register a function as an available tool."""
    TOOL_REGISTRY[func.__name__] = func
    return func


@register_tool
async def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city. In production, this calls a weather API."""
    # Simulated response
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "sunny", "humidity": 45},
        "London": {"temp": 14, "condition": "cloudy", "humidity": 78},
        "New York": {"temp": 28, "condition": "partly cloudy", "humidity": 62},
    }
    result = weather_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})
    result["city"] = city
    result["units"] = units
    return result


@register_tool
async def search_orders(order_id: str | None = None, email: str | None = None) -> dict:
    """Search for customer orders by ID or email."""
    if order_id:
        return {"order_id": order_id, "status": "shipped", "items": ["Widget A", "Gadget B"]}
    if email:
        return {"orders": [{"order_id": "ORD-001", "status": "delivered"}]}
    return {"error": "Provide either order_id or email"}


async def execute_tool(tool_call: ToolCall) -> str:
    """Execute a tool call and return the result as a string."""
    func = TOOL_REGISTRY.get(tool_call.name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_call.name}"})

    try:
        result = await func(**tool_call.arguments)
        return json.dumps(result)
    except TypeError as e:
        # Model passed wrong arguments
        return json.dumps({"error": f"Invalid arguments: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


async def agent_loop(
    client: LLMClient,
    user_message: str,
    tools: list[dict],
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 10,
) -> str:
    """
    Core agent loop. Sends messages to the LLM, executes tool calls,
    and loops until the model responds with text or max iterations hit.
    """
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]

    for iteration in range(max_iterations):
        # Step 1: Call the LLM
        response = await client.chat(messages, tools=tools)

        # Step 2: Check if model wants to use tools
        if not response.tool_calls:
            # Model responded with text -- task is complete
            return response.content

        # Step 3: Append assistant message (with tool calls) to history
        messages.append(response)

        # Step 4: Execute each tool call and append results
        for tool_call in response.tool_calls:
            result = await execute_tool(tool_call)
            messages.append(Message(
                role="tool",
                content=result,
                tool_call_id=tool_call.id,
            ))

        # Loop continues: LLM will see tool results and decide next step

    return "I was unable to complete the task within the iteration limit."


# ---------------------------------------------------------------------------
# Example 2: Tool Definitions with JSON Schema
# ---------------------------------------------------------------------------

CUSTOMER_SERVICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base for product information, "
                "policies, and troubleshooting guides. Use when the customer "
                "asks a question about our products or company policies. "
                "Returns relevant articles with titles and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Be specific and include keywords."
                    },
                    "category": {
                        "type": "string",
                        "description": "Narrow results to a specific category",
                        "enum": ["products", "policies", "troubleshooting", "faq"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5)",
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up the status of a customer order by order ID. "
                "Returns current status, tracking info, and estimated delivery. "
                "Use when the customer asks about an existing order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID in format ORD-XXXXX"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a support ticket for issues that cannot be resolved "
                "in conversation. Use as a last resort after attempting to "
                "help the customer directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Brief title describing the issue"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description including what was tried"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Issue severity level"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer's email for follow-up"
                    }
                },
                "required": ["title", "description", "priority"]
            }
        }
    },
]


# ---------------------------------------------------------------------------
# Example 3: Multi-Tool Orchestration
# ---------------------------------------------------------------------------

async def multi_tool_agent(
    client: LLMClient,
    user_message: str,
) -> str:
    """
    Agent that selects from multiple tools based on the user's request.
    The system prompt guides tool selection strategy.
    """
    system_prompt = """You are a customer service agent for TechCorp.

Available actions:
1. Search the knowledge base for product info and policies
2. Look up order status by order ID
3. Create support tickets for unresolved issues

Strategy:
- Always try to answer from the knowledge base first
- Only create a ticket if you cannot resolve the issue
- Be specific in your search queries for better results
- If the customer mentions an order, look it up before answering"""

    return await agent_loop(
        client=client,
        user_message=user_message,
        tools=CUSTOMER_SERVICE_TOOLS,
        system_prompt=system_prompt,
        max_iterations=8,
    )


# ---------------------------------------------------------------------------
# Example 4: Conversation Memory with Summarization
# ---------------------------------------------------------------------------

@dataclass
class ConversationMemory:
    """Manages conversation history with automatic summarization."""
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    window_size: int = 20  # Keep last N messages in full
    summarize_threshold: int = 30  # Trigger summarization at this count

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def get_messages(self) -> list[Message]:
        """Return messages for the LLM, including summary of older messages."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]

        result = list(system_msgs)

        # Prepend summary of older messages if available
        if self.summary:
            result.append(Message(
                role="system",
                content=f"Summary of earlier conversation:\n{self.summary}"
            ))

        # Include recent messages in full
        result.extend(non_system[-self.window_size:])
        return result

    def needs_summarization(self) -> bool:
        non_system = [m for m in self.messages if m.role != "system"]
        return len(non_system) > self.summarize_threshold

    async def summarize(self, client: LLMClient) -> None:
        """Compress older messages into a summary."""
        if not self.needs_summarization():
            return

        non_system = [m for m in self.messages if m.role != "system"]
        old_messages = non_system[:-self.window_size]

        # Ask the LLM to summarize
        summary_messages = [
            Message(role="system", content=(
                "Summarize the following conversation concisely. "
                "Preserve key facts: user preferences, decisions made, "
                "issues discussed, and any commitments."
            )),
            *old_messages,
        ]

        response = await client.chat(summary_messages)
        self.summary = response.content

        # Remove old messages, keep system + recent
        system_msgs = [m for m in self.messages if m.role == "system"]
        recent = non_system[-self.window_size:]
        self.messages = system_msgs + recent


async def agent_with_memory(
    client: LLMClient,
    memory: ConversationMemory,
    user_message: str,
    tools: list[dict],
) -> str:
    """Agent loop that uses conversation memory with summarization."""
    memory.add(Message(role="user", content=user_message))

    # Summarize if history is getting long
    if memory.needs_summarization():
        await memory.summarize(client)

    messages = memory.get_messages()

    for _ in range(10):
        response = await client.chat(messages, tools=tools)

        if not response.tool_calls:
            memory.add(response)
            return response.content

        messages.append(response)
        memory.add(response)

        for tc in response.tool_calls:
            result = await execute_tool(tc)
            tool_msg = Message(role="tool", content=result, tool_call_id=tc.id)
            messages.append(tool_msg)
            memory.add(tool_msg)

    return "Max iterations reached."


# ---------------------------------------------------------------------------
# Example 5: ReAct Pattern Implementation
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are an agent that solves problems step by step.

For each step, use this format:

Thought: [Your reasoning about what to do next]
Action: [Call a tool if needed]
Observation: [You'll see the tool result here]

When you have enough information to answer, respond with:

Thought: [Final reasoning]
Answer: [Your final answer to the user]

Always think before acting. Explain WHY you're taking each action."""


async def react_agent(
    client: LLMClient,
    user_message: str,
    tools: list[dict],
    max_steps: int = 8,
) -> str:
    """
    ReAct agent that explicitly reasons before each tool call.

    The model is prompted to output Thought/Action/Observation traces.
    With native tool use APIs, the "Action" is a tool call and the
    "Observation" is the tool result. The "Thought" appears in the
    model's text output before the tool call.
    """
    messages = [
        Message(role="system", content=REACT_SYSTEM_PROMPT),
        Message(role="user", content=user_message),
    ]

    trace: list[dict] = []  # For debugging: log each step

    for step in range(max_steps):
        response = await client.chat(messages, tools=tools)

        # Log the reasoning (text portion of the response)
        if response.content:
            trace.append({"step": step, "thought": response.content})

        if not response.tool_calls:
            # Model's final answer (no more tools needed)
            trace.append({"step": step, "answer": response.content})
            return response.content

        messages.append(response)

        for tc in response.tool_calls:
            trace.append({
                "step": step,
                "action": tc.name,
                "arguments": tc.arguments,
            })

            result = await execute_tool(tc)
            trace.append({"step": step, "observation": result})

            messages.append(Message(
                role="tool",
                content=result,
                tool_call_id=tc.id,
            ))

    # Provide trace for debugging if max steps hit
    return f"Max steps reached. Trace:\n{json.dumps(trace, indent=2)}"


# ---------------------------------------------------------------------------
# Example 6: Tool Call Validation and Error Handling
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_tool_call(
    tool_call: ToolCall,
    tool_schemas: list[dict],
) -> ValidationResult:
    """Validate a tool call against its JSON schema before execution."""
    # Find the matching schema
    schema = None
    for tool in tool_schemas:
        func_def = tool.get("function", tool)
        if func_def.get("name") == tool_call.name:
            schema = func_def
            break

    if not schema:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown tool: {tool_call.name}. Available: {[t.get('function', t)['name'] for t in tool_schemas]}"]
        )

    params = schema.get("parameters", schema.get("input_schema", {}))
    errors = []

    # Check required parameters
    for required in params.get("required", []):
        if required not in tool_call.arguments:
            errors.append(f"Missing required parameter: '{required}'")

    # Check enum values
    properties = params.get("properties", {})
    for key, value in tool_call.arguments.items():
        if key not in properties:
            errors.append(f"Unknown parameter: '{key}'")
            continue

        prop_schema = properties[key]

        # Type checking
        expected_type = prop_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Parameter '{key}' must be a string, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Parameter '{key}' must be an integer, got {type(value).__name__}")

        # Enum validation
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            errors.append(
                f"Parameter '{key}' must be one of {prop_schema['enum']}, got '{value}'"
            )

        # Range validation
        if "minimum" in prop_schema and isinstance(value, (int, float)):
            if value < prop_schema["minimum"]:
                errors.append(f"Parameter '{key}' must be >= {prop_schema['minimum']}")
        if "maximum" in prop_schema and isinstance(value, (int, float)):
            if value > prop_schema["maximum"]:
                errors.append(f"Parameter '{key}' must be <= {prop_schema['maximum']}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


async def safe_execute_tool(
    tool_call: ToolCall,
    tool_schemas: list[dict],
    requires_approval: set[str] | None = None,
) -> str:
    """
    Execute a tool call with validation, permission checks, and error handling.
    Returns a JSON string suitable for passing back to the LLM.
    """
    requires_approval = requires_approval or {"create_support_ticket", "issue_refund"}

    # Step 1: Validate arguments against schema
    validation = validate_tool_call(tool_call, tool_schemas)
    if not validation.valid:
        return json.dumps({
            "error": "Validation failed",
            "details": validation.errors,
            "suggestion": "Fix the arguments and try again."
        })

    # Step 2: Check if human approval is required
    if tool_call.name in requires_approval:
        # In production, this would present a UI prompt or send a notification
        print(f"[APPROVAL REQUIRED] {tool_call.name}({tool_call.arguments})")
        # For this example, auto-approve
        approved = True
        if not approved:
            return json.dumps({
                "error": "Action not approved by user",
                "suggestion": "Inform the user that the action requires approval."
            })

    # Step 3: Execute with error handling
    try:
        result = await execute_tool(tool_call)
        return result
    except Exception as e:
        return json.dumps({
            "error": f"Execution failed: {str(e)}",
            "suggestion": "Try a different approach or inform the user of the issue."
        })


# ---------------------------------------------------------------------------
# Example 7: Simple Multi-Agent Pattern (Orchestrator + Specialists)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a specialist agent."""
    name: str
    system_prompt: str
    tools: list[dict]
    max_iterations: int = 5


class OrchestratorAgent:
    """
    Routes user requests to specialist agents based on intent.
    The orchestrator itself is an LLM that classifies the request
    and selects the appropriate specialist.
    """

    def __init__(
        self,
        client: LLMClient,
        specialists: dict[str, AgentConfig],
    ):
        self.client = client
        self.specialists = specialists

    async def route(self, user_message: str) -> str:
        """Classify intent and route to the right specialist."""
        # Step 1: Use the LLM to classify the request
        routing_prompt = self._build_routing_prompt()
        messages = [
            Message(role="system", content=routing_prompt),
            Message(role="user", content=user_message),
        ]

        # Force structured output: the orchestrator returns which specialist to use
        routing_tools = [{
            "type": "function",
            "function": {
                "name": "route_to_specialist",
                "description": "Route the user's request to the appropriate specialist agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specialist": {
                            "type": "string",
                            "enum": list(self.specialists.keys()),
                            "description": "Which specialist should handle this request"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the routing decision"
                        }
                    },
                    "required": ["specialist", "reason"]
                }
            }
        }]

        response = await self.client.chat(messages, tools=routing_tools)

        if not response.tool_calls:
            # Model responded with text (simple question, no routing needed)
            return response.content

        routing = response.tool_calls[0].arguments
        specialist_name = routing["specialist"]

        # Step 2: Delegate to the specialist
        specialist = self.specialists[specialist_name]
        return await agent_loop(
            client=self.client,
            user_message=user_message,
            tools=specialist.tools,
            system_prompt=specialist.system_prompt,
            max_iterations=specialist.max_iterations,
        )

    def _build_routing_prompt(self) -> str:
        specialist_descriptions = "\n".join(
            f"- {name}: {config.system_prompt[:100]}..."
            for name, config in self.specialists.items()
        )
        return f"""You are a routing agent. Analyze the user's request and route it
to the most appropriate specialist.

Available specialists:
{specialist_descriptions}

Use the route_to_specialist tool to select the best specialist."""


# Example usage of the orchestrator

ORDER_TOOLS = [CUSTOMER_SERVICE_TOOLS[1]]  # get_order_status only
KB_TOOLS = [CUSTOMER_SERVICE_TOOLS[0]]     # search_knowledge_base only
TICKET_TOOLS = CUSTOMER_SERVICE_TOOLS      # All tools

specialists = {
    "order_specialist": AgentConfig(
        name="Order Specialist",
        system_prompt="You help customers with order-related questions. Look up orders and provide status updates.",
        tools=ORDER_TOOLS,
    ),
    "product_specialist": AgentConfig(
        name="Product Specialist",
        system_prompt="You answer questions about products, features, and policies using the knowledge base.",
        tools=KB_TOOLS,
    ),
    "support_specialist": AgentConfig(
        name="Support Specialist",
        system_prompt="You handle complex issues that may require creating support tickets. Try to resolve first.",
        tools=TICKET_TOOLS,
        max_iterations=8,
    ),
}


async def demo_orchestrator(client: LLMClient) -> None:
    """Demonstrates the orchestrator routing different requests."""
    orchestrator = OrchestratorAgent(client, specialists)

    # These would be routed to different specialists
    queries = [
        "Where is my order ORD-12345?",          # -> order_specialist
        "What's your return policy?",             # -> product_specialist
        "My product is broken and I need help",   # -> support_specialist
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = await orchestrator.route(query)
        print(f"Agent: {response}")
