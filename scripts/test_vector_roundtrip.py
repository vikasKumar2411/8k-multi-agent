from pprint import pprint

from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.retrieval.qdrant_store import SECQdrantStore


def main() -> None:
    embedding_client = OllamaEmbeddingClient()
    vector_store = SECQdrantStore()

    document_text = (
        "Tesla reported record quarterly vehicle production "
        "and delivery results."
    )

    document_vector = embedding_client.embed_query(document_text)

    print(f"Embedding dimension: {len(document_vector)}")

    point_id = vector_store.upsert_chunk(
        accession_number="0000950170-24-000282",
        chunk_id=0,
        vector=document_vector,
        payload={
            "accession_number": "0000950170-24-000282",
            "symbol": "TSLA",
            "company_name": "Tesla, Inc.",
            "filing_type": "8-K",
            "title": (
                "Tesla Reports Record Vehicle Production "
                "and Deliveries"
            ),
            "release_datetime": "2024-01-02T21:29:17Z",
            "chunk_id": 0,
            "chunk_text": document_text,
            "dataset_version": "v1",
        },
    )

    print(f"Inserted point: {point_id}")

    query = "What did Tesla report about vehicle deliveries?"
    query_vector = embedding_client.embed_query(query)

    results = vector_store.search(
        query_vector=query_vector,
        limit=3,
    )

    print("\nSearch results:")
    pprint(results)


if __name__ == "__main__":
    main()