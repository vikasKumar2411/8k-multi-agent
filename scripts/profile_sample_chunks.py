from pathlib import Path

import pandas as pd

from src.data.chunking import chunk_filing
from src.data.normalization import normalize_filing_row


DATA_PATH = Path("data/raw/sec_8k_filings.parquet")
SAMPLE_SIZE = 100
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    sample_df = (
        df.drop_duplicates(subset=["sec_accession_number"])
        .head(SAMPLE_SIZE)
        .copy()
    )

    records = []

    for _, row in sample_df.iterrows():
        filing = normalize_filing_row(row)

        chunks = chunk_filing(
            filing,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        records.append(
            {
                "accession_number": filing.accession_number,
                "symbol": filing.symbol,
                "title": filing.title,
                "raw_characters": len(filing.raw_text),
                "chunk_count": len(chunks),
            }
        )

    profile = pd.DataFrame(records)

    print("\nChunk-count statistics:")
    print(
        profile["chunk_count"].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )

    print("\nLargest filings:")
    print(
        profile.sort_values("chunk_count", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()