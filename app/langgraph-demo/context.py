from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_current: ContextVar["RunContext | None"] = ContextVar("run_context", default=None)


@dataclass(frozen=True)
class RunContext:
    user_id: str
    conversation_id: str
    model: str


def current_context() -> RunContext:
    context = _current.get()
    if context is None:
        raise RuntimeError("no active run context")
    return context


@contextmanager
def active_context(context: RunContext):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
