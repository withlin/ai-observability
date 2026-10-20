import json
import time
from dataclasses import dataclass
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from context import current_context
from llm import ChatClient
from models import TokenUsage
from prompts import (
    CURRENCY_HINT,
    POLICY_HINT,
    SUPERVISOR_HINT,
    SYSTEM_PROMPT,
    WEATHER_HINT,
    WRITER_HINT,
)
from telemetry import Telemetry

SUPERVISOR_AGENT = "supervisor_agent"
WEATHER_AGENT = "weather_agent"
CURRENCY_AGENT = "currency_agent"
POLICY_AGENT = "policy_agent"
WRITER_AGENT = "writer_agent"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    model: str
    next: str
    answer: str
    tool_calls: int
    inference_calls: int
    usage: TokenUsage
    ttft_samples: tuple[float, ...]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    hint: str
    tools: tuple[Callable, ...] = ()


@dataclass(frozen=True)
class GraphDependencies:
    llm: ChatClient
    telemetry: Telemetry
    weather_tools: tuple[Callable, ...]
    currency_tools: tuple[Callable, ...]
    policy_tools: tuple[Callable, ...]


def _tool_definitions(tools: tuple[Callable, ...]) -> str | None:
    if not tools:
        return None
    return json.dumps([convert_to_openai_tool(tool) for tool in tools])


def _merge_usage(state: AgentState, result: TokenUsage) -> TokenUsage:
    return state.get("usage", TokenUsage()).merge(result)


def _with_ttft(state: AgentState, value: float | None) -> tuple[float, ...]:
    samples = state.get("ttft_samples", ())
    return samples + ((value,) if value is not None else ())


class SupervisorNode:
    def __init__(self, llm: ChatClient, telemetry: Telemetry, spec: AgentSpec):
        self._llm = llm
        self._telemetry = telemetry
        self._spec = spec

    def __call__(self, state: AgentState) -> dict:
        context = current_context()
        model = state["model"]
        started = time.perf_counter()
        with self._telemetry.agent_span(
            self._spec.name, self._spec.description, model, context,
            SYSTEM_PROMPT, None, state["messages"],
        ) as span:
            try:
                result = self._llm.complete(
                    model,
                    [SystemMessage(content=SYSTEM_PROMPT), *state["messages"],
                     HumanMessage(content=SUPERVISOR_HINT)],
                    agent_name=self._spec.name,
                )
            except Exception as exc:
                span.set_error(exc)
                self._telemetry.record_agent_duration(
                    self._spec.name, model, time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
                raise
            span.set_usage(result.usage)
            span.set_output([result.message])
        self._telemetry.record_agent_duration(
            self._spec.name, model, time.perf_counter() - started)
        self._telemetry.record_agent_steps(self._spec.name, 1, 0)
        return {
            "next": parse_route(result.message.content),
            "inference_calls": state.get("inference_calls", 0) + 1,
            "usage": _merge_usage(state, result.usage),
            "ttft_samples": _with_ttft(state, result.ttft),
        }


class SpecialistNode:
    def __init__(self, llm: ChatClient, telemetry: Telemetry, spec: AgentSpec):
        self._llm = llm
        self._telemetry = telemetry
        self._spec = spec
        self._tool_definitions = _tool_definitions(spec.tools)
        self._graph = self._build_react_graph()

    def _build_react_graph(self):
        tool_node = ToolNode(list(self._spec.tools), handle_tool_errors=True)
        spec = self._spec
        llm = self._llm

        def call_model(state: AgentState) -> dict:
            result = llm.complete(
                state["model"],
                [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]],
                agent_name=spec.name,
                tools=list(spec.tools),
            )
            return {
                "messages": [result.message],
                "inference_calls": state.get("inference_calls", 0) + 1,
                "usage": _merge_usage(state, result.usage),
                "ttft_samples": _with_ttft(state, result.ttft),
            }

        def call_tools(state: AgentState) -> dict:
            count = 0
            for message in reversed(state.get("messages", [])):
                if isinstance(message, AIMessage):
                    count = len(message.tool_calls or [])
                    break
            output = tool_node.invoke(state)
            return {
                "messages": output["messages"],
                "tool_calls": state.get("tool_calls", 0) + count,
            }

        graph = StateGraph(AgentState)
        graph.add_node("model", call_model)
        graph.add_node("tools", call_tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition)
        graph.add_edge("tools", "model")
        return graph.compile()

    def __call__(self, state: AgentState) -> dict:
        context = current_context()
        model = state["model"]
        started = time.perf_counter()
        with self._telemetry.agent_span(
            self._spec.name, self._spec.description, model, context,
            SYSTEM_PROMPT, self._tool_definitions, state["messages"],
        ) as span:
            try:
                result = self._graph.invoke({
                    "messages": [*state["messages"], HumanMessage(content=self._spec.hint)],
                    "model": model,
                    "conversation_id": state["conversation_id"],
                })
            except Exception as exc:
                span.set_error(exc)
                self._telemetry.record_agent_duration(
                    self._spec.name, model, time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
                raise
            usage = result.get("usage", TokenUsage())
            span.set_usage(usage)
            new_messages = result["messages"][len(state["messages"]) + 1:]
            ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]
            if ai_messages:
                span.set_output(ai_messages[-1:])
        self._telemetry.record_agent_duration(
            self._spec.name, model, time.perf_counter() - started)
        self._telemetry.record_agent_steps(
            self._spec.name,
            result.get("inference_calls", 0),
            result.get("tool_calls", 0),
        )
        return {
            "messages": new_messages,
            "tool_calls": state.get("tool_calls", 0) + result.get("tool_calls", 0),
            "inference_calls": state.get("inference_calls", 0) + result.get("inference_calls", 0),
            "usage": _merge_usage(state, usage),
            "ttft_samples": state.get("ttft_samples", ()) + result.get("ttft_samples", ()),
        }


class WriterNode:
    def __init__(self, llm: ChatClient, telemetry: Telemetry, spec: AgentSpec):
        self._llm = llm
        self._telemetry = telemetry
        self._spec = spec

    def __call__(self, state: AgentState) -> dict:
        context = current_context()
        model = state["model"]
        started = time.perf_counter()
        with self._telemetry.agent_span(
            self._spec.name, self._spec.description, model, context,
            SYSTEM_PROMPT, None, state["messages"],
        ) as span:
            try:
                result = self._llm.complete(
                    model,
                    [SystemMessage(content=SYSTEM_PROMPT), *state["messages"],
                     HumanMessage(content=WRITER_HINT)],
                    agent_name=self._spec.name,
                )
            except Exception as exc:
                span.set_error(exc)
                self._telemetry.record_agent_duration(
                    self._spec.name, model, time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
                raise
            span.set_usage(result.usage)
            span.set_output([result.message])
        self._telemetry.record_agent_duration(
            self._spec.name, model, time.perf_counter() - started)
        self._telemetry.record_agent_steps(self._spec.name, 1, 0)
        return {
            "messages": [result.message],
            "answer": str(result.message.content),
            "inference_calls": state.get("inference_calls", 0) + 1,
            "usage": _merge_usage(state, result.usage),
            "ttft_samples": _with_ttft(state, result.ttft),
        }


def parse_route(text: str) -> str:
    lowered = (text or "").lower()
    for name in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT, "writer"):
        if name in lowered:
            return name
    for name in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT):
        if name.split("_")[0] in lowered:
            return name
    return "writer"


def _route(state: AgentState) -> str:
    decision = state.get("next")
    if decision in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT, WRITER_AGENT):
        return decision
    return WRITER_AGENT


def build_graph(dependencies: GraphDependencies):
    supervisor = AgentSpec(
        SUPERVISOR_AGENT,
        "Routes a travel question to weather, currency, policy, or the writer.",
        SUPERVISOR_HINT,
    )
    weather = AgentSpec(
        WEATHER_AGENT,
        "Weather specialist; uses get_weather for real-time conditions.",
        WEATHER_HINT,
        dependencies.weather_tools,
    )
    currency = AgentSpec(
        CURRENCY_AGENT,
        "Currency specialist; uses convert_currency for real-time rates.",
        CURRENCY_HINT,
        dependencies.currency_tools,
    )
    policy = AgentSpec(
        POLICY_AGENT,
        "Travel-policy specialist; loads the travel_policy skill and runs its budget script.",
        POLICY_HINT,
        dependencies.policy_tools,
    )
    writer = AgentSpec(
        WRITER_AGENT,
        "Composes the final traveler-facing answer from the conversation.",
        WRITER_HINT,
    )

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", SupervisorNode(dependencies.llm, dependencies.telemetry, supervisor))
    graph.add_node(WEATHER_AGENT, SpecialistNode(
        dependencies.llm, dependencies.telemetry, weather))
    graph.add_node(CURRENCY_AGENT, SpecialistNode(
        dependencies.llm, dependencies.telemetry, currency))
    graph.add_node(POLICY_AGENT, SpecialistNode(
        dependencies.llm, dependencies.telemetry, policy))
    graph.add_node("writer", WriterNode(dependencies.llm, dependencies.telemetry, writer))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", _route,
        {
            WEATHER_AGENT: WEATHER_AGENT,
            CURRENCY_AGENT: CURRENCY_AGENT,
            POLICY_AGENT: POLICY_AGENT,
            WRITER_AGENT: "writer",
        },
    )
    graph.add_edge(WEATHER_AGENT, "supervisor")
    graph.add_edge(CURRENCY_AGENT, "supervisor")
    graph.add_edge(POLICY_AGENT, "supervisor")
    graph.add_edge("writer", END)
    return graph.compile()
