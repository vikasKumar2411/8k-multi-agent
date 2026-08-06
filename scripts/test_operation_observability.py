from src.observability import (
    OperationType,
    bind_execution_context,
    configure_observability_logging,
    observe_operation,
)


def main() -> None:
    configure_observability_logging()

    with bind_execution_context(
        run_id="test-run-001",
        thread_id="test-thread-001",
        workflow_name="bounded_sec_research",
        node_name="test_node",
    ):
        with observe_operation(
            operation_type=OperationType.LLM,
            operation_name="test_llm_call",
            provider="ollama",
            model_name="qwen2.5:7b-instruct",
            input_text="Test prompt for observability.",
            attributes={
                "temperature": 0.0,
                "structured_output": True,
            },
        ) as observation:
            output = '{"result": "success"}'

            observation.set_output_text(
                output
            )

            observation.set_attribute(
                "schema_validation_success",
                True,
            )


if __name__ == "__main__":
    main()