import base64
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ("gemini-3.8-flash", "gpt-4o", "kimi-k3")
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"
TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def _parse_pairs(raw: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        key = unquote(key.strip())
        if sep and key:
            pairs[key] = unquote(value.strip())
    return pairs


def _signal_endpoint(base: str, signal: str, override: str) -> str:
    override = override.strip()
    if override:
        return override
    return f"{base.rstrip('/')}/v1/{signal}"


def _basic_auth_header() -> dict[str, str]:
    username = os.getenv("OTEL_EXPORTER_OTLP_USERNAME", "")
    password = os.getenv("OTEL_EXPORTER_OTLP_PASSWORD", "")
    if not username and not password:
        return {}
    if not username or not password:
        raise ValueError(
            "OTEL_EXPORTER_OTLP_USERNAME and OTEL_EXPORTER_OTLP_PASSWORD "
            "must be set together"
        )
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _headers(signal: str) -> dict[str, str]:
    headers = _parse_pairs(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
    headers.update(
        _parse_pairs(os.getenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_HEADERS", ""))
    )
    if not any(key.lower() == "authorization" for key in headers):
        headers.update(_basic_auth_header())
    return headers


@dataclass(frozen=True)
class Settings:
    otlp_endpoint: str
    otlp_traces_endpoint: str
    otlp_metrics_endpoint: str
    otlp_traces_headers: dict[str, str]
    otlp_metrics_headers: dict[str, str]
    otlp_certificate: str | None
    otlp_tls_skip_verify: bool
    service_name: str
    resource_attributes: dict[str, str]
    llm_base_url: str
    models: tuple[str, ...]
    runs: int
    interval: float
    provider: str
    server_address: str
    server_port: int
    capture_content: bool
    memory_path: str
    skills_root: str
    in_memory_telemetry: bool

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.getenv("POC_LLM_BASE_URL", "http://127.0.0.1:4141/v1")
        parsed = urlparse(base_url)
        models = tuple(
            model.strip()
            for model in os.getenv("POC_MODELS", ",".join(DEFAULT_MODELS)).split(",")
            if model.strip()
        )
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT
        ).strip()
        resource_attributes = _parse_pairs(
            os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        )
        return cls(
            otlp_endpoint=otlp_endpoint,
            otlp_traces_endpoint=_signal_endpoint(
                otlp_endpoint, "traces",
                os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""),
            ),
            otlp_metrics_endpoint=_signal_endpoint(
                otlp_endpoint, "metrics",
                os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", ""),
            ),
            otlp_traces_headers=_headers("traces"),
            otlp_metrics_headers=_headers("metrics"),
            otlp_certificate=os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE") or None,
            otlp_tls_skip_verify=_env_bool("OTEL_EXPORTER_OTLP_TLS_SKIP_VERIFY"),
            service_name=(
                os.getenv("OTEL_SERVICE_NAME")
                or resource_attributes.get("service.name")
                or "langgraph-agent"
            ),
            resource_attributes=resource_attributes,
            llm_base_url=base_url,
            models=models or DEFAULT_MODELS,
            runs=int(os.getenv("POC_RUNS", "1")),
            interval=float(os.getenv("POC_INTERVAL", "0")),
            provider=os.getenv("POC_PROVIDER_NAME", "openai"),
            server_address=parsed.hostname or "127.0.0.1",
            server_port=parsed.port or 4141,
            capture_content=_env_bool("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"),
            memory_path=os.getenv("POC_MEMORY_PATH", str(APP_DIR / ".state" / "memory.json")),
            skills_root=os.getenv("POC_SKILLS_ROOT", str(APP_DIR / "skills")),
            in_memory_telemetry=_env_bool("POC_TELEMETRY_IN_MEMORY"),
        )
