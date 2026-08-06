# scripts/setup_postgres_checkpointer.py

from src.config.settings import get_settings
from src.observability import (
    configure_observability_logging,
)
from src.persistence import (
    create_postgres_checkpointer,
)


def main() -> None:
    configure_observability_logging()

    settings = get_settings()

    print("=" * 88)
    print("LANGGRAPH POSTGRES CHECKPOINTER SETUP")
    print("=" * 88)
    print(
        "Database URL configured:",
        bool(settings.langgraph_database_url),
    )

    with create_postgres_checkpointer(
        setup=True
    ):
        print(
            "LangGraph PostgreSQL checkpoint "
            "tables initialized successfully."
        )


if __name__ == "__main__":
    main()