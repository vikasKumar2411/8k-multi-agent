from pathlib import Path

import pandas as pd

from src.data.chunking import chunk_filing
from src.data.normalization import normalize_filing_row
from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.retrieval.qdrant_store import SECQdrantStore


DATA_PATH = Path("data/raw/sec_8k_filings.parquet")

SAMPLE_SIZE = 100
EMBEDDING_BATCH_SIZE = 16
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.resolve()}"
        )

    df = pd.read_parquet(DATA_PATH)

    required_columns = {
        "sec_accession_number",
        "release_datetime",
        "title",
        "sec_filing_type",
        "keywords",
        "exchange",
        "symbol",
        "company_name",
        "excerpt",
        "raw_text",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sample_df = (
        df.drop_duplicates(subset=["sec_accession_number"])
        .head(SAMPLE_SIZE)
        .copy()
    )

    chunks = []
    failed_rows = []

    for index, row in sample_df.iterrows():
        try:
            filing = normalize_filing_row(row)
            filing_chunks = chunk_filing(
                filing,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks.extend(filing_chunks)
        except Exception as exc:
            failed_rows.append(
                {
                    "index": int(index),
                    "error": str(exc),
                }
            )

    print(f"Filings selected: {len(sample_df)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Failed filings: {len(failed_rows)}")

    embedding_client = OllamaEmbeddingClient()
    vector_store = SECQdrantStore()

    inserted = 0

    for batch_number, chunk_batch in enumerate(
        batched(chunks, EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        texts = [chunk.chunk_text for chunk in chunk_batch]
        vectors = embedding_client.embed_texts(texts)

        payloads = [
            {
                "accession_number": chunk.accession_number,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "release_datetime": chunk.release_datetime.isoformat(),
                "title": chunk.title,
                "filing_type": chunk.filing_type,
                "symbol": chunk.symbol,
                "company_name": chunk.company_name,
                "keywords": chunk.keywords,
                "chunk_item_numbers": chunk.chunk_item_numbers,
                "filing_item_numbers": chunk.filing_item_numbers,
                "exchange": chunk.exchange,
                "excerpt": chunk.excerpt,
                "dataset_version": chunk.dataset_version,
            }
            for chunk in chunk_batch
        ]

        count = vector_store.upsert_chunks(
            vectors=vectors,
            payloads=payloads,
        )
        inserted += count

        print(
            f"Batch {batch_number}: inserted {count} chunks "
            f"(total={inserted})"
        )

    print("\nIngestion complete")
    print(f"Total chunks inserted: {inserted}")

    if failed_rows:
        print("\nFailures:")
        for failure in failed_rows[:10]:
            print(failure)


if __name__ == "__main__":
    main()