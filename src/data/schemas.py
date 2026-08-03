from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FilingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accession_number: str
    release_datetime: datetime
    title: str
    filing_type: str
    keywords: list[str] = Field(default_factory=list)
    exchange: str | None = None
    symbol: str
    company_name: str
    excerpt: str
    raw_text: str
    dataset_version: str = "v1"


class FilingChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accession_number: str
    chunk_id: int
    chunk_text: str

    release_datetime: datetime
    title: str
    filing_type: str
    symbol: str
    company_name: str

    keywords: list[str] = Field(default_factory=list)

    chunk_item_numbers: list[str] = Field(default_factory=list)
    filing_item_numbers: list[str] = Field(default_factory=list)

    exchange: str | None = None
    excerpt: str | None = None
    dataset_version: str = "v1"