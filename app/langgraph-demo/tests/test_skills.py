import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from context import RunContext, active_context
from skills import SkillRepository, SkillTools
from support import make_settings
from telemetry import Telemetry


class SkillTest(unittest.TestCase):
    def setUp(self):
        settings = make_settings(capture_content=False)
        self.telemetry = Telemetry(settings)
        self.repository = SkillRepository(Path(settings.skills_root))
        self.tools = SkillTools(self.repository, self.telemetry)
        self.context = RunContext("user-test", "conv-test", "test-model")

    def _span(self, name):
        return next(span for span in self.telemetry.finished_spans() if span.name == name)

    def test_load_skill_emits_refined_span(self):
        with active_context(self.context):
            text = self.tools.load_skill()
        self.assertIn("travel_policy", text)
        span = self._span("execute_tool load_skill travel_policy")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "execute_tool")
        self.assertEqual(span.attributes["gen_ai.skill.name"], "travel_policy")
        self.assertIn("gen_ai.skill.source.uri", span.attributes)
        self.assertNotIn("gen_ai.tool.call.arguments", span.attributes)

    def test_search_emits_tool_and_retrieval_spans(self):
        with active_context(self.context):
            payload = json.loads(self.tools.search_skill_resources(query="Japan visa", top_k=2))
        ids = [doc["id"] for doc in payload["documents"]]
        self.assertIn("references/entry_requirements.md", ids)
        tool = self._span("execute_tool search_skill_resources")
        retrieval = self._span("retrieval travel-policy-docs")
        self.assertEqual(tool.attributes["gen_ai.tool.name"], "search_skill_resources")
        self.assertEqual(retrieval.attributes["gen_ai.operation.name"], "retrieval")
        self.assertEqual(retrieval.attributes["gen_ai.data_source.id"], "travel-policy-docs")
        self.assertEqual(retrieval.attributes["gen_ai.retrieval.top_k"], 2)
        self.assertNotIn("gen_ai.retrieval.documents", retrieval.attributes)

    def test_estimate_budget_emits_command_span(self):
        with active_context(self.context):
            out = self.tools.estimate_budget("Tokyo", 5, "mid-range")
        payload = json.loads(out)
        self.assertEqual(payload["total_low"], payload["daily_low"] * 5)
        span = self._span("execute_tool estimate_budget travel_policy scripts/estimate_budget.py")
        self.assertEqual(span.attributes["gen_ai.tool.name"], "estimate_budget")
        self.assertEqual(span.attributes["gen_ai.skill.resource.name"], "scripts/estimate_budget.py")
        self.assertEqual(span.attributes["process.exit.code"], 0)

    def test_read_resource_rejects_traversal(self):
        with active_context(self.context):
            with self.assertRaises(ValueError):
                self.tools.read_skill_resource(resource_name="../SKILL.md")


if __name__ == "__main__":
    unittest.main()
