from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class EvidenceItem(BaseModel):
    """
    Normalized evidence passed to downstream analysis agents.

    One EvidenceItem represents one retrieved filing chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: str = Field(min_length=1)

    point_id: str = Field(min_length=1)
    accession_number: str = Field(min_length=1)
    chunk_id: int = Field(ge=0)

    symbol: str = Field(min_length=1)
    company_name: str = Field(min_length=1)

    title: str = Field(min_length=1)
    release_datetime: datetime

    filing_type: str = "8-K"

    excerpt: str | None = None
    chunk_text: str = Field(min_length=1)

    keywords: list[str] = Field(default_factory=list)
    chunk_item_numbers: list[str] = Field(default_factory=list)
    filing_item_numbers: list[str] = Field(default_factory=list)

    vector_score: float
    rerank_score: float

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()

        if not normalized:
            raise ValueError("symbol cannot be empty")

        return normalized

    @classmethod
    def from_search_result(
        cls,
        result: dict[str, Any],
        *,
        evidence_index: int,
    ) -> "EvidenceItem":
        """
        Convert one SECQdrantStore.search() result into EvidenceItem.
        """
        payload = result.get("payload") or {}

        point_id = str(result.get("id") or "").strip()

        accession_number = str(
            payload.get("accession_number") or ""
        ).strip()

        symbol = str(
            payload.get("symbol") or ""
        ).strip()

        company_name = str(
            payload.get("company_name") or ""
        ).strip()

        title = str(
            payload.get("title") or ""
        ).strip()

        chunk_text = str(
            payload.get("chunk_text") or ""
        ).strip()

        release_datetime = payload.get("release_datetime")

        if not point_id:
            raise ValueError(
                "Search result is missing point id"
            )

        if not accession_number:
            raise ValueError(
                "Search result payload is missing accession_number"
            )

        if not symbol:
            raise ValueError(
                "Search result payload is missing symbol"
            )

        if not company_name:
            company_name = symbol

        if not title:
            title = "Untitled SEC filing"

        if not chunk_text:
            raise ValueError(
                "Search result payload is missing chunk_text"
            )

        if not release_datetime:
            raise ValueError(
                "Search result payload is missing release_datetime"
            )

        vector_score = float(
            result.get(
                "vector_score",
                result.get("score", 0.0),
            )
        )

        rerank_score = float(
            result.get(
                "rerank_score",
                vector_score,
            )
        )

        return cls(
            evidence_id=f"evidence-{evidence_index}",
            point_id=point_id,
            accession_number=accession_number,
            chunk_id=int(payload.get("chunk_id", 0)),
            symbol=symbol,
            company_name=company_name,
            title=title,
            release_datetime=release_datetime,
            filing_type=str(
                payload.get("filing_type") or "8-K"
            ),
            excerpt=(
                str(payload["excerpt"]).strip()
                if payload.get("excerpt")
                else None
            ),
            chunk_text=chunk_text,
            keywords=[
                str(value)
                for value in payload.get("keywords") or []
            ],
            chunk_item_numbers=[
                str(value)
                for value in (
                    payload.get("chunk_item_numbers") or []
                )
            ],
            filing_item_numbers=[
                str(value)
                for value in (
                    payload.get("filing_item_numbers") or []
                )
            ],
            vector_score=vector_score,
            rerank_score=rerank_score,
        )


class CoverageResult(BaseModel):
    """
    Describes whether retrieved evidence covers the entities
    requested by the query plan.
    """

    model_config = ConfigDict(extra="forbid")

    requested_symbols: list[str] = Field(default_factory=list)
    retrieved_symbols: list[str] = Field(default_factory=list)
    missing_symbols: list[str] = Field(default_factory=list)

    evidence_count: int = Field(ge=0)
    filing_count: int = Field(ge=0)

    complete: bool
    retry_recommended: bool

    message: str