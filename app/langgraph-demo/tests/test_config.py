import base64
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings


class SettingsTest(unittest.TestCase):
    def load(self, **env) -> Settings:
        with patch.dict(os.environ, env, clear=True):
            return Settings.from_env()

    def test_defaults_target_local_http_collector(self):
        settings = self.load()
        self.assertEqual("http://localhost:4318", settings.otlp_endpoint)
        self.assertEqual(
            "http://localhost:4318/v1/traces", settings.otlp_traces_endpoint)
        self.assertEqual(
            "http://localhost:4318/v1/metrics", settings.otlp_metrics_endpoint)
        self.assertEqual({}, settings.otlp_traces_headers)
        self.assertEqual({}, settings.otlp_metrics_headers)
        self.assertFalse(settings.otlp_tls_skip_verify)
        self.assertEqual("langgraph-agent", settings.service_name)

    def test_tls_skip_verify_is_opt_in(self):
        settings = self.load(OTEL_EXPORTER_OTLP_TLS_SKIP_VERIFY="true")
        self.assertTrue(settings.otlp_tls_skip_verify)

    def test_https_endpoint_builds_signal_paths(self):
        settings = self.load(
            OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.example.com/otel/")
        self.assertEqual(
            "https://otlp.example.com/otel/v1/traces",
            settings.otlp_traces_endpoint)
        self.assertEqual(
            "https://otlp.example.com/otel/v1/metrics",
            settings.otlp_metrics_endpoint)

    def test_signal_endpoints_override_base(self):
        settings = self.load(
            OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.example.com",
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=(
                "https://traces.example.com/custom/traces"),
            OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=(
                "https://metrics.example.com/custom/metrics"),
        )
        self.assertEqual(
            "https://traces.example.com/custom/traces",
            settings.otlp_traces_endpoint)
        self.assertEqual(
            "https://metrics.example.com/custom/metrics",
            settings.otlp_metrics_endpoint)

    def test_username_password_become_basic_auth(self):
        settings = self.load(
            OTEL_EXPORTER_OTLP_USERNAME="agent",
            OTEL_EXPORTER_OTLP_PASSWORD="p@ss:word",
        )
        expected = "Basic " + base64.b64encode(b"agent:p@ss:word").decode()
        self.assertEqual(
            expected, settings.otlp_traces_headers["Authorization"])
        self.assertEqual(
            expected, settings.otlp_metrics_headers["Authorization"])

    def test_username_requires_password(self):
        with self.assertRaises(ValueError):
            self.load(OTEL_EXPORTER_OTLP_USERNAME="agent")

    def test_headers_are_parsed_and_explicit_authorization_wins(self):
        settings = self.load(
            OTEL_EXPORTER_OTLP_HEADERS=(
                "x-api-key=abc%20123,Authorization=Bearer%20token"),
            OTEL_EXPORTER_OTLP_USERNAME="agent",
            OTEL_EXPORTER_OTLP_PASSWORD="secret",
        )
        self.assertEqual(
            "abc 123", settings.otlp_traces_headers["x-api-key"])
        self.assertEqual(
            "Bearer token", settings.otlp_traces_headers["Authorization"])

    def test_signal_headers_merge_over_base_headers(self):
        settings = self.load(
            OTEL_EXPORTER_OTLP_HEADERS="x-api-key=base",
            OTEL_EXPORTER_OTLP_TRACES_HEADERS="x-api-key=traces",
        )
        self.assertEqual(
            "traces", settings.otlp_traces_headers["x-api-key"])
        self.assertEqual(
            "base", settings.otlp_metrics_headers["x-api-key"])

    def test_service_name_and_resource_attributes(self):
        settings = self.load(
            OTEL_RESOURCE_ATTRIBUTES=(
                "deployment.environment=staging,service.version=9.9.9"),
            OTEL_SERVICE_NAME="travel-assistant",
        )
        self.assertEqual("travel-assistant", settings.service_name)
        self.assertEqual(
            "staging", settings.resource_attributes["deployment.environment"])

    def test_resource_service_name_used_when_env_missing(self):
        settings = self.load(
            OTEL_RESOURCE_ATTRIBUTES="service.name=from-resource")
        self.assertEqual("from-resource", settings.service_name)


if __name__ == "__main__":
    unittest.main()
