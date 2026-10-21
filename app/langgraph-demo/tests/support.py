from dataclasses import replace
from pathlib import Path

from config import APP_DIR, Settings


def make_settings(**overrides) -> Settings:
    base = Settings(
        otlp_endpoint="http://localhost:4318",
        otlp_traces_endpoint="http://localhost:4318/v1/traces",
        otlp_metrics_endpoint="http://localhost:4318/v1/metrics",
        otlp_traces_headers={},
        otlp_metrics_headers={},
        otlp_certificate=None,
        otlp_tls_skip_verify=False,
        service_name="langgraph-agent",
        resource_attributes={},
        llm_base_url="http://127.0.0.1:4141/v1",
        models=("test-model",),
        runs=1,
        interval=0.0,
        provider="openai",
        server_address="127.0.0.1",
        server_port=4141,
        capture_content=False,
        memory_path=str(Path("/tmp") / "genai-test-memory.json"),
        skills_root=str(APP_DIR / "skills"),
        in_memory_telemetry=True,
    )
    return replace(base, **overrides)
