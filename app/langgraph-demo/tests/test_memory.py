import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from context import RunContext, active_context
from memory import JsonMemoryStore, MemoryService
from support import make_settings
from telemetry import Telemetry


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.telemetry = Telemetry(make_settings())
        self.store = JsonMemoryStore(Path(self.tmp.name) / "memory.json")
        self.memory = MemoryService(self.store, self.telemetry)
        self.context = RunContext("user-1", "conv-1", "test-model")

    def tearDown(self):
        self.tmp.cleanup()

    def _span(self, name):
        return next(span for span in self.telemetry.finished_spans() if span.name == name)

    def test_create_then_search(self):
        with active_context(self.context):
            record_id = self.memory.add(self.context, "question", "answer")
            records = self.memory.search(self.context, "question")
        self.assertTrue(record_id.startswith("mem_"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].answer, "answer")
        create = self._span("create_memory")
        search = self._span("search_memory")
        self.assertEqual(create.attributes["gen_ai.memory.record.count"], 1)
        self.assertEqual(create.attributes["gen_ai.memory.record.id"], record_id)
        self.assertEqual(search.attributes["gen_ai.memory.record.count"], 1)
        self.assertEqual(search.attributes["gen_ai.memory.store.id"], "traveler_memory")
        self.assertNotIn("gen_ai.memory.query.text", search.attributes)
        self.assertNotIn("gen_ai.memory.records", search.attributes)

    def test_search_recovers_from_malformed_file(self):
        (Path(self.tmp.name) / "memory.json").write_text("{not json", encoding="utf-8")
        with active_context(self.context):
            records = self.memory.search(self.context, "question")
        self.assertEqual(records, [])
        self.assertIn("error.type", self._span("search_memory").attributes)

    def test_records_are_scoped_to_conversation(self):
        with active_context(self.context):
            self.memory.add(self.context, "q1", "a1")
        other = RunContext("user-2", "conv-2", "test-model")
        with active_context(other):
            records = self.memory.search(other, "q1")
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
