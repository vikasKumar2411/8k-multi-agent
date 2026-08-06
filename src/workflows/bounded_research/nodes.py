# src/workflows/bounded_research/nodes.py

from typing import Any

from src.agents.answer_generator import AnswerGenerator
from src.agents.answer_schemas import FinalAnswer
from src.agents.evidence_analyzer import EvidenceAnalysisAgent
from src.agents.query_planner import QueryPlanningAgent
from src.agents.schemas import AnalysisResult, QueryPlan
from src.agents.verification_schemas import VerificationResult
from src.agents.verifier import VerificationAgent
from src.retrieval.coordinator import RetrievalCoordinator
from src.retrieval.schemas import CoverageResult, EvidenceItem
from src.workflows.bounded_research.state import (
    BoundedResearchState,
)


def dump_model(model: Any) -> dict[str, Any]:
    """
    Convert a Pydantic model into JSON-compatible graph state.
    """
    return model.model_dump(
        mode="json",
        exclude_none=False,
    )


def load_plan(
    state: BoundedResearchState,
) -> QueryPlan:
    """
    Reconstruct QueryPlan from persisted graph state.
    """
    return QueryPlan.model_validate(
        state["query_plan"]
    )


def load_evidence(
    state: BoundedResearchState,
) -> list[EvidenceItem]:
    """
    Reconstruct evidence items from persisted graph state.
    """
    return [
        EvidenceItem.model_validate(item)
        for item in state.get("evidence", [])
    ]


def load_coverage(
    state: BoundedResearchState,
) -> CoverageResult:
    """
    Reconstruct coverage result from persisted graph state.
    """
    return CoverageResult.model_validate(
        state["coverage"]
    )


def load_analysis(
    state: BoundedResearchState,
) -> AnalysisResult:
    """
    Reconstruct analysis result from persisted graph state.
    """
    return AnalysisResult.model_validate(
        state["analysis"]
    )


def load_verification(
    state: BoundedResearchState,
) -> VerificationResult:
    """
    Reconstruct verification result from persisted graph state.
    """
    return VerificationResult.model_validate(
        state["verification"]
    )


class BoundedResearchNodes:
    """
    LangGraph node implementations for the bounded SEC workflow.

    Agents operate on Pydantic models internally, while graph state
    stores JSON-compatible dictionaries for safe checkpointing.
    """

    def __init__(
        self,
        *,
        planner: QueryPlanningAgent | None = None,
        retrieval_coordinator: RetrievalCoordinator | None = None,
        analyzer: EvidenceAnalysisAgent | None = None,
        verifier: VerificationAgent | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.planner = planner or QueryPlanningAgent()

        self.retrieval_coordinator = (
            retrieval_coordinator
            or RetrievalCoordinator()
        )

        self.analyzer = (
            analyzer
            or EvidenceAnalysisAgent()
        )

        self.verifier = (
            verifier
            or VerificationAgent()
        )

        self.answer_generator = (
            answer_generator
            or AnswerGenerator()
        )

    def plan_query(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        user_query = (
            state.get("user_query", "")
            .strip()
        )

        if not user_query:
            raise ValueError(
                "user_query cannot be empty"
            )

        plan = self.planner.plan(
            user_query
        )

        return {
            "query_plan": dump_model(plan),
            "current_node": "plan_query",
            "workflow_status": "planned",
            "error": None,
        }

    def respond_to_rejection(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        plan = load_plan(state)

        final_answer = FinalAnswer(
            answer=(
                plan.rejection_reason
                or (
                    "This request is outside the "
                    "supported scope."
                )
            ),
            citations=[],
            covered_symbols=[],
            missing_symbols=[],
            answerable=False,
            verification_complete=False,
            warnings=[],
        )

        return {
            "final_answer": dump_model(
                final_answer
            ),
            "current_node": (
                "respond_to_rejection"
            ),
            "workflow_status": "rejected",
            "error": None,
        }

    def respond_to_clarification(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        plan = load_plan(state)

        final_answer = FinalAnswer(
            answer=(
                plan.clarification_question
                or (
                    "Additional clarification "
                    "is required."
                )
            ),
            citations=[],
            covered_symbols=[],
            missing_symbols=[],
            answerable=False,
            verification_complete=False,
            warnings=[],
        )

        return {
            "final_answer": dump_model(
                final_answer
            ),
            "current_node": (
                "respond_to_clarification"
            ),
            "workflow_status": (
                "clarification_required"
            ),
            "error": None,
        }

    def retrieve_evidence(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        plan = load_plan(state)

        outcome = (
            self.retrieval_coordinator
            .retrieve(
                plan=plan,
                candidate_limit=30,
                result_limit=8,
                retry_candidate_limit=60,
                retry_result_limit=8,
                max_chunks_per_filing=2,
            )
        )

        return {
            "evidence": [
                dump_model(item)
                for item in outcome.evidence
            ],
            "coverage": dump_model(
                outcome.coverage
            ),
            "retrieval_attempts": (
                outcome.attempts
            ),
            "retried_symbols": (
                outcome.retried_symbols
            ),
            "current_node": (
                "retrieve_evidence"
            ),
            "workflow_status": (
                "evidence_retrieved"
            ),
            "error": None,
        }

    def analyze_evidence(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        user_query = state["user_query"]
        plan = load_plan(state)
        evidence = load_evidence(state)
        coverage = load_coverage(state)

        if not evidence:
            analysis = AnalysisResult(
                findings=[],
                analyzed_evidence_ids=[],
                unused_evidence_ids=[],
                missing_symbols=(
                    coverage.missing_symbols
                ),
                analysis_complete=False,
                limitations=[
                    (
                        "No evidence was available "
                        "for analysis."
                    )
                ],
            )
        else:
            analysis = self.analyzer.analyze(
                user_query=user_query,
                plan=plan,
                evidence=evidence,
                coverage=coverage,
            )

        return {
            "analysis": dump_model(
                analysis
            ),
            "current_node": (
                "analyze_evidence"
            ),
            "workflow_status": (
                "evidence_analyzed"
            ),
            "error": None,
        }

    def verify_analysis(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        verification = self.verifier.verify(
            user_query=state["user_query"],
            plan=load_plan(state),
            analysis=load_analysis(state),
            evidence=load_evidence(state),
            coverage=load_coverage(state),
        )

        return {
            "verification": dump_model(
                verification
            ),
            "current_node": (
                "verify_analysis"
            ),
            "workflow_status": (
                "analysis_verified"
            ),
            "error": None,
        }

    def generate_answer(
        self,
        state: BoundedResearchState,
    ) -> dict[str, Any]:
        final_answer = (
            self.answer_generator
            .generate(
                user_query=(
                    state["user_query"]
                ),
                plan=load_plan(state),
                analysis=load_analysis(
                    state
                ),
                verification=(
                    load_verification(
                        state
                    )
                ),
                coverage=load_coverage(
                    state
                ),
                evidence=load_evidence(
                    state
                ),
            )
        )

        return {
            "final_answer": dump_model(
                final_answer
            ),
            "current_node": (
                "generate_answer"
            ),
            "workflow_status": (
                "completed"
            ),
            "error": None,
        }