# src/observability/context.py

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_run_id_context: ContextVar[str] = ContextVar(
    "run_id",
    default="unknown-run",
)

_thread_id_context: ContextVar[str] = ContextVar(
    "thread_id",
    default="unknown-thread",
)

_workflow_context: ContextVar[str] = ContextVar(
    "workflow_name",
    default="unknown-workflow",
)

_node_context: ContextVar[str] = ContextVar(
    "node_name",
    default="unknown-node",
)


def get_run_id() -> str:
    return _run_id_context.get()


def get_thread_id() -> str:
    return _thread_id_context.get()


def get_workflow_name() -> str:
    return _workflow_context.get()


def get_node_name() -> str:
    return _node_context.get()


@contextmanager
def bind_execution_context(
    *,
    run_id: str,
    thread_id: str,
    workflow_name: str,
    node_name: str,
) -> Iterator[None]:
    """
    Make execution identifiers available to lower-level operations.

    Context variables propagate through normal synchronous Python calls,
    so clients such as Ollama and Qdrant do not need run_id parameters.
    """

    run_token = _run_id_context.set(run_id)
    thread_token = _thread_id_context.set(thread_id)
    workflow_token = _workflow_context.set(workflow_name)
    node_token = _node_context.set(node_name)

    try:
        yield
    finally:
        _run_id_context.reset(run_token)
        _thread_id_context.reset(thread_token)
        _workflow_context.reset(workflow_token)
        _node_context.reset(node_token)