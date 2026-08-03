from typing import Any

import pandas as pd

from src.data.schemas import FilingDocument


def parse_keywords(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def normalize_optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def normalize_filing_row(
    row: pd.Series,
    *,
    dataset_version: str = "v1",
) -> FilingDocument:
    release_datetime = pd.to_datetime(
        row["release_datetime"],
        utc=True,
        errors="raise",
    ).to_pydatetime()

    raw_text = str(row["raw_text"]).strip()
    if not raw_text:
        raise ValueError("raw_text is empty")

    return FilingDocument(
        accession_number=str(row["sec_accession_number"]).strip(),
        release_datetime=release_datetime,
        title=str(row["title"]).strip(),
        filing_type=str(row["sec_filing_type"]).strip(),
        keywords=parse_keywords(row.get("keywords")),
        exchange=normalize_optional_string(row.get("exchange")),
        symbol=str(row["symbol"]).strip().upper(),
        company_name=str(row["company_name"]).strip(),
        excerpt=str(row["excerpt"]).strip(),
        raw_text=raw_text,
        dataset_version=dataset_version,
    )