from pprint import pprint

from src.agents.query_planner import QueryPlanningAgent


TEST_QUERIES = [
    "What operational metrics did Tesla report in 2024?",
    "Compare Tesla and Ford operational updates from 2024.",
    "Find executive resignations and appointments.",
    "Show cybersecurity incidents disclosed in 2024.",
    "What happened at Apple?",
    "Should I buy Tesla stock?",
    "Predict NVIDIA's stock price next month.",
    "What is the capital of France?",
]


def main() -> None:
    agent = QueryPlanningAgent()

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"User query: {query}")
        print("=" * 80)

        try:
            plan = agent.plan(query)
            pprint(plan.model_dump(mode="json"))
        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()