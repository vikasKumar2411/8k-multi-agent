from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerCitation(BaseModel):
    """
    Evidence reference attached to a final answer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: str = Field(min_length=1)
    symbol: str | None = None
    company_name: str | None = None
    title: str | None = None
    filing_release_date: str | None = None
    accession_number: str | None = None


class FinalAnswer(BaseModel):
    """
    Final user-facing output from the bounded research workflow.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    answer: str = Field(min_length=1)

    citations: list[AnswerCitation] = Field(
        default_factory=list
    )

    covered_symbols: list[str] = Field(
        default_factory=list
    )
    missing_symbols: list[str] = Field(
        default_factory=list
    )

    answerable: bool
    verification_complete: bool

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_result(self) -> "FinalAnswer":
        self.covered_symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in self.covered_symbols
                if symbol.strip()
            }
        )

        self.missing_symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in self.missing_symbols
                if symbol.strip()
            }
        )

        self.warnings = list(
            dict.fromkeys(
                warning.strip()
                for warning in self.warnings
                if warning.strip()
            )
        )

        citation_map = {
            citation.evidence_id: citation
            for citation in self.citations
        }

        self.citations = [
            citation_map[evidence_id]
            for evidence_id in sorted(citation_map)
        ]

        return self