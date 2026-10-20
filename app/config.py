import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ("gemini-3.8-flash", "gpt-4o", "kimi-k3")
TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


@dataclass(frozen=True)
class Settings:
    otlp_endpoint: str
    llm_base_url: str
    models: tuple[str, ...]
    runs: int
    interval: float
    tool_timeout: float
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
        return cls(
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            llm_base_url=base_url,
            models=models or DEFAULT_MODELS,
            runs=int(os.getenv("POC_RUNS", "1")),
            interval=float(os.getenv("POC_INTERVAL", "0")),
            tool_timeout=float(os.getenv("POC_TOOL_TIMEOUT", "10")),
            provider=os.getenv("POC_PROVIDER_NAME", "openai"),
            server_address=parsed.hostname or "127.0.0.1",
            server_port=parsed.port or 4141,
            capture_content=_env_bool("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"),
            memory_path=os.getenv("POC_MEMORY_PATH", str(APP_DIR / ".state" / "memory.json")),
            skills_root=os.getenv("POC_SKILLS_ROOT", str(APP_DIR / "skills")),
            in_memory_telemetry=_env_bool("POC_TELEMETRY_IN_MEMORY"),
        )
