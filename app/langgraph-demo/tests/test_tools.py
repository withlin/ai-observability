import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from context import RunContext, active_context
from support import make_settings
from telemetry import Telemetry
from tools import WebTools


class WebToolsTest(unittest.TestCase):
    def setUp(self):
        self.telemetry = Telemetry(make_settings(in_memory_telemetry=True))
        self.tools = WebTools(self.telemetry)
        self.context = RunContext(
            user_id="user", conversation_id="conv", model="test-model")

    def call(self, tool, *args):
        with active_context(self.context):
            return tool(*args)

    def test_weather_returns_local_data(self):
        result = self.call(self.tools.weather_tool(), "Tokyo")
        self.assertEqual("Tokyo: clear, +24°C", result)
        self.assertEqual(
            "execute_tool get_weather",
            self.telemetry.finished_spans()[-1].name,
        )

    def test_weather_falls_back_for_unknown_city(self):
        result = self.call(self.tools.weather_tool(), "atlantis")
        self.assertEqual("Atlantis: clear, +20°C", result)

    def test_currency_converts_with_mock_rates(self):
        result = json.loads(
            self.call(self.tools.currency_tool(), 250, "eur", "jpy"))
        rate = 150.0 / 0.92
        self.assertEqual("EUR", result["from"])
        self.assertEqual("JPY", result["to"])
        self.assertAlmostEqual(rate, result["rate"])
        self.assertEqual(round(250 * rate, 2), result["converted"])
        self.assertEqual(
            "execute_tool convert_currency",
            self.telemetry.finished_spans()[-1].name,
        )

    def test_currency_rejects_unknown_source(self):
        with self.assertRaises(RuntimeError):
            self.call(self.tools.currency_tool(), 10, "XXX", "USD")

    def test_currency_rejects_unknown_target(self):
        with self.assertRaises(KeyError):
            self.call(self.tools.currency_tool(), 10, "USD", "XXX")


if __name__ == "__main__":
    unittest.main()
