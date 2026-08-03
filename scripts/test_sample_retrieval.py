from pprint import pprint

from src.embeddings.ollama_client import OllamaEmbeddingClient
from src.retrieval.qdrant_store import SECQdrantStore


TEST_CASES = [
    {
        "query": "vehicle production and delivery results",
        "symbols": ["TSLA"],
        "event_category": "operational",
    },
    {
        "query": "executive resignation or leadership appointment",
        "symbols": None,
        "event_category": "leadership",
    },
    {
        "query": "quarterly earnings and financial results",
        "symbols": None,
        "event_category": "earnings",
    },
]

def main() -> None:
    embedding_client = OllamaEmbeddingClient()
    vector_store = SECQdrantStore()

    for test_case in TEST_CASES[:1]:
        query = test_case["query"]
        symbols = test_case["symbols"]

        query_vector = embedding_client.embed_query(query)

        event_category = test_case["event_category"]

        results = vector_store.search(
            query_vector=query_vector,
            symbols=symbols,
            event_category=event_category,
            limit=5,
        )

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Symbols: {symbols}")
        print("=" * 80)

        simplified = [
            {
                "vector_score": round(
                    result["vector_score"],
                    4,
                ),
                "rerank_score": round(
                    result["rerank_score"],
                    4,
                ),
                "symbol": result["payload"].get("symbol"),
                "title": result["payload"].get("title"),
                "chunk_item_numbers": result["payload"].get(
                "chunk_item_numbers"
                ),
                "filing_item_numbers": result["payload"].get(
                    "filing_item_numbers"
                ),
                "accession_number": result["payload"].get(
                    "accession_number"
                ),
                "chunk_id": result["payload"].get("chunk_id"),
                "chunk_preview": result["payload"]
                .get("chunk_text", "")[:300],
            }
            for result in results
        ]

        pprint(simplified)


if __name__ == "__main__":
    main()