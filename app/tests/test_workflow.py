import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import GraphDependencies, build_graph
from llm import ChatClient
from memory import JsonMemoryStore, MemoryService
from skills import SkillRepository, SkillTools
from support import make_settings
from telemetry import Telemetry
from tools import HttpClient, WebTools
from workflow import WorkflowRunner


class FakeGraph:
    def invoke(self, state):
        return {"answer": "ok", "tool_calls": 0, "inference_calls": 0}


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(memory_path=str(Path(self.tmp.name) / "memory.json"))
        self.telemetry = Telemetry(self.settings)
        self.memory = MemoryService(
            JsonMemoryStore(Path(self.settings.memory_path)), self.telemetry)
        self.runner = WorkflowRunner(FakeGraph(), self.memory, self.telemetry, self.settings)

    def tearDown(self):
        self.tmp.cleanup()

    def test_workflow_root_conformance(self):
        self.runner.run(1)
        spans = self.telemetry.finished_spans()
        root = next(span for span in spans if span.name == "invoke_workflow travel_workflow")
        self.assertEqual(root.attributes["gen_ai.operation.name"], "invoke_workflow")
        self.assertEqual(root.attributes["gen_ai.workflow.name"], "travel_workflow")
        self.assertIn("gen_ai.conversation.id", root.attributes)
        self.assertIn("enduser.id", root.attributes)
        self.assertIn("gen_ai.usage.input_tokens", root.attributes)
        self.assertIn("gen_ai.usage.output_tokens", root.attributes)
        self.assertIn("poc.error_count", root.attributes)
        self.assertIn("poc.inference_calls", root.attributes)
        self.assertIn("poc.tool_calls", root.attributes)
        self.assertTrue(any(span.name == "search_memory" for span in spans))
        self.assertTrue(any(span.name == "create_memory" for span in spans))

    def test_graph_builds_with_real_components(self):
        telemetry = Telemetry(self.settings)
        http = HttpClient(5)
        try:
            llm = ChatClient(self.settings, telemetry)
            web = WebTools(http, telemetry)
            skills = SkillTools(SkillRepository(Path(self.settings.skills_root)), telemetry)
            graph = build_graph(GraphDependencies(
                llm=llm,
                telemetry=telemetry,
                weather_tools=(web.weather_tool(),),
                currency_tools=(web.currency_tool(),),
                policy_tools=tuple(skills.tools()),
            ))
            self.assertIsNotNone(graph)
        finally:
            http.close()


if __name__ == "__main__":
    unittest.main()
