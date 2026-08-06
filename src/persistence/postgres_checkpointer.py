# src/persistence/postgres_checkpointer.py

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import (
    PostgresSaver,
)

from src.config.settings import get_settings


@contextmanager
def create_postgres_checkpointer(
    *,
    setup: bool = False,
) -> Iterator[PostgresSaver]:
    """
    Create a synchronous LangGraph PostgreSQL checkpointer.

    The context manager must remain open for the entire period in
    which the compiled graph is used.

    Args:
        setup:
            When True, run LangGraph's database migrations before
            yielding the checkpointer. Use during explicit database
            initialization, not on every workflow run.
    """

    settings = get_settings()

    with PostgresSaver.from_conn_string(
        settings.langgraph_database_url
    ) as checkpointer:
        if setup:
            checkpointer.setup()

        yield checkpointer