from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventCategory(StrEnum):
    LEADERSHIP = "leadership"
    EARNINGS = "earnings"
    RESTRUCTURING = "restructuring"
    MATERIAL_AGREEMENT = "material_agreement"
    ACQUISITION = "acquisition"
    BANKRUPTCY = "bankruptcy"
    CYBERSECURITY = "cybersecurity"
    FINANCIAL_OBLIGATION = "financial_obligation"
    OPERATIONAL = "operational"
    OTHER = "other"


class TaskType(StrEnum):
    EVENT_SEARCH = "event_search"
    COMPANY_TIMELINE = "company_timeline"
    COMPANY_COMPARISON = "company_comparison"
    EVENT_SUMMARY = "event_summary"
    METRIC_EXTRACTION = "metric_extraction"
    FILING_LOOKUP = "filing_lookup"
    UNSUPPORTED = "unsupported"


class QueryPlan(BaseModel):
    """Validated retrieval plan for the SEC 8-K copilot."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    in_scope: bool
    rejection_reason: str | None = None

    symbols: list[str] = Field(default_factory=list)

    start_date: date | None = None
    end_date: date | None = None

    event_category: EventCategory | None = None
    task_type: TaskType

    retrieval_query: str | None = None

    clarification_needed: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "QueryPlan":
        self.symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in self.symbols
                if symbol and symbol.strip()
            }
        )

        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError(
                    "start_date cannot be after end_date"
                )

        if not self.in_scope:
            if not self.rejection_reason:
                raise ValueError(
                    "Out-of-scope requests require rejection_reason"
                )

            self.task_type = TaskType.UNSUPPORTED
            self.event_category = None
            self.retrieval_query = None
            self.clarification_needed = False
            self.clarification_question = None

        if self.in_scope and not self.retrieval_query:
            raise ValueError(
                "In-scope requests require retrieval_query"
            )

        if self.clarification_needed:
            if not self.clarification_question:
                raise ValueError(
                    "clarification_needed requires "
                    "clarification_question"
                )

        if not self.clarification_needed:
            self.clarification_question = None

        return self