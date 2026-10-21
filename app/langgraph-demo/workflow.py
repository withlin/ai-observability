import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from config import Settings
from context import RunContext, active_context
from graph import GraphDependencies, build_graph
from llm import ChatClient
from memory import JsonMemoryStore, MemoryService
from models import TokenUsage
from skills import SkillRepository, SkillTools
from telemetry import Telemetry
from tools import WebTools

WORKFLOW_NAME = "travel_workflow"


@dataclass(frozen=True)
class Conversation:
    user_id: str
    questions: tuple[str, str]


CONVERSATIONS = (
    Conversation(
        user_id="user-001",
        questions=(
            "What's the weather in Tokyo right now, and how much is 250 EUR in JPY?",
            "For that same 5-day Tokyo trip on a mid-range budget, do I need a visa, "
            "what's the baggage allowance, and can you run the budget estimate for me?",
        ),
    ),
    Conversation(
        user_id="user-002",
        questions=(
            "What's the weather in Lisbon tomorrow, and what is 300 USD in EUR?",
            "For that same 4-day Lisbon trip on a mid-range budget, do I need a visa, "
            "what's the baggage allowance, and can you run the budget estimate for me?",
        ),
    ),
    Conversation(
        user_id="user-003",
        questions=(
            "What's the weather in Toronto today, and what is 500 USD in CAD?",
            "For that same 3-day Toronto trip on a budget, do I need a visa, "
            "what's the baggage allowance, and can you run the budget estimate for me?",
        ),
    ),
)


class WorkflowRunner:
    def __init__(self, graph, memory: MemoryService, telemetry: Telemetry,
                 settings: Settings):
        self._graph = graph
        self._memory = memory
        self._telemetry = telemetry
        self._settings = settings

    def _plan(self, index: int):
        conversation = CONVERSATIONS[(index - 1) // 2 % len(CONVERSATIONS)]
        question_index = (index - 1) % 2
        question = conversation.questions[question_index]
        conversation_id = f"conv-{(index - 1) // 2 % len(CONVERSATIONS) + 1:04d}"
        model = self._settings.models[(index - 1) % len(self._settings.models)]
        return conversation, question, question_index, conversation_id, model

    def run(self, index: int) -> None:
        conversation, question, question_index, conversation_id, model = self._plan(index)
        context = RunContext(
            user_id=conversation.user_id,
            conversation_id=conversation_id,
            model=model,
        )
        started = time.perf_counter()
        with active_context(context):
            with self._telemetry.workflow_span(WORKFLOW_NAME, context, question) as span:
                span.set_attribute("gen_ai.request.model", model)
                try:
                    records = self._memory.search(context, question)
                    memory_message = self._memory.context_message(records)
                    messages = ([memory_message] if memory_message else []) + [
                        HumanMessage(
                            content=question,
                            id=f"{conversation_id}-{question_index}",
                        )
                    ]
                    result = self._graph.invoke({
                        "messages": messages,
                        "model": model,
                        "conversation_id": conversation_id,
                    })
                    answer = str(result.get("answer", ""))
                    usage = result.get("usage", TokenUsage())
                    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
                    span.set_attribute(
                        "gen_ai.usage.cache_read.input_tokens", usage.cache_read_tokens)
                    span.set_attribute(
                        "gen_ai.usage.cache_write.input_tokens", usage.cache_write_tokens)
                    span.set_attribute(
                        "gen_ai.usage.reasoning.output_tokens", usage.reasoning_tokens)
                    ttft_samples = result.get("ttft_samples", ())
                    if ttft_samples:
                        average_ttft = round(sum(ttft_samples) / len(ttft_samples), 6)
                        span.set_attribute(
                            "gen_ai.response.time_to_first_chunk", average_ttft)
                        span.set_attribute("poc.ttft", average_ttft)
                    span.set_attribute(
                        "poc.inference_calls", int(result.get("inference_calls", 0)))
                    span.set_attribute(
                        "poc.tool_calls", int(result.get("tool_calls", 0)))
                    span.set_attribute("poc.error_count", 0)
                    span.set_output([AIMessage(content=answer)])
                    self._memory.add(context, question, answer)
                except Exception as exc:
                    span.set_attribute("poc.error_count", 1)
                    span.set_attribute("poc.inference_calls", 0)
                    span.set_attribute("poc.tool_calls", 0)
                    span.set_error(exc)
                    self._telemetry.record_workflow_duration(
                        WORKFLOW_NAME,
                        time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                    print(f"run {index}: FAILED {type(exc).__name__}: {exc}")
                    return
                trace_id = span.trace_id
        self._telemetry.record_workflow_duration(
            WORKFLOW_NAME, time.perf_counter() - started)
        print(
            f"run {index}: model={model} tools={result.get('tool_calls')} "
            f"inferences={result.get('inference_calls')} "
            f"tokens={usage.input_tokens}+{usage.output_tokens} "
            f"answer={answer[:32]!r} trace={trace_id:032x}"
        )

    def run_all(self) -> None:
        index = 0
        try:
            while self._settings.runs == 0 or index < self._settings.runs:
                index += 1
                self.run(index)
                if self._settings.interval:
                    time.sleep(self._settings.interval)
        except KeyboardInterrupt:
            print("\ninterrupted by user")
        print(f"done: {index} invocation(s)")


def main() -> None:
    settings = Settings.from_env()
    telemetry = Telemetry(settings)
    llm = ChatClient(settings, telemetry)
    memory = MemoryService(JsonMemoryStore(Path(settings.memory_path)), telemetry)
    skills = SkillRepository(Path(settings.skills_root))
    skill_tools = tuple(SkillTools(skills, telemetry).tools())
    web = WebTools(telemetry)
    graph = build_graph(GraphDependencies(
        llm=llm,
        telemetry=telemetry,
        weather_tools=(web.weather_tool(),),
        currency_tools=(web.currency_tool(),),
        policy_tools=skill_tools,
    ))
    runner = WorkflowRunner(graph, memory, telemetry, settings)
    try:
        runner.run_all()
    finally:
        telemetry.shutdown()
