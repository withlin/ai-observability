import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import make_settings
from telemetry import Telemetry, serialize_messages


class TelemetryTest(unittest.TestCase):
    def test_capture_content_comes_from_settings(self):
        self.assertTrue(Telemetry(make_settings(capture_content=True)).capture_content)
        self.assertFalse(Telemetry(make_settings(capture_content=False)).capture_content)

    def test_serialize_messages_uses_spec_shape(self):
        from langchain_core.messages import HumanMessage

        raw = serialize_messages([HumanMessage(content="weather?")])
        self.assertEqual(
            json.loads(raw),
            [{"role": "user", "parts": [{"type": "text", "content": "weather?"}]}],
        )

    def test_resource_reflects_settings(self):
        telemetry = Telemetry(make_settings(
            service_name="travel-assistant",
            resource_attributes={
                "deployment.environment": "staging",
                "service.version": "9.9.9",
            },
        ))
        attributes = telemetry.tracer_provider.resource.attributes
        self.assertEqual("travel-assistant", attributes["service.name"])
        self.assertEqual("staging", attributes["deployment.environment"])
        self.assertEqual("9.9.9", attributes["service.version"])

    def test_instruments_exist(self):
        telemetry = Telemetry(make_settings())
        for name in (
            "inference_duration",
            "inference_ttfc",
            "inference_tpoc",
            "operation_duration",
            "tool_duration",
            "agent_duration",
            "workflow_duration",
        ):
            self.assertTrue(hasattr(telemetry, name), name)


if __name__ == "__main__":
    unittest.main()
