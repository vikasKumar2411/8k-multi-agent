from typing import Any
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.config.settings import get_settings
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
)
from src.retrieval.reranker import rerank_results

class SECQdrantStore:
    def __init__(self) -> None:
        settings = get_settings()

        self.collection_name = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url)

    @staticmethod
    def make_point_id(accession_number: str, chunk_id: int) -> str:
        key = f"{accession_number}:{chunk_id}"
        return str(uuid5(NAMESPACE_URL, key))

    def upsert_chunk(
        self,
        *,
        accession_number: str,
        chunk_id: int,
        vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        point_id = self.make_point_id(accession_number, chunk_id)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )

        return point_id


    def search(
        self,
        *,
        query_vector: list[float],
        limit: int = 5,
        symbols: list[str] | None = None,
        event_category: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None

        if symbols:
            normalized_symbols = [
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            ]

            if len(normalized_symbols) == 1:
                symbol_condition = FieldCondition(
                    key="symbol",
                    match=MatchValue(
                        value=normalized_symbols[0]
                    ),
                )
            else:
                symbol_condition = FieldCondition(
                    key="symbol",
                    match=MatchAny(
                        any=normalized_symbols
                    ),
                )

            query_filter = Filter(
                must=[symbol_condition]
            )

        candidate_limit = max(limit * 6, 30)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )

        raw_results = [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in response.points
        ]

        return rerank_results(
            raw_results,
            event_category=event_category,
            limit=limit,
            max_chunks_per_filing=2,
        )


    def upsert_chunks(
        self,
        *,
        vectors: Sequence[list[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> int:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have equal length")

        points: list[PointStruct] = []

        for vector, payload in zip(vectors, payloads, strict=True):
            accession_number = str(payload["accession_number"])
            chunk_id = int(payload["chunk_id"])

            point_id = self.make_point_id(
                accession_number=accession_number,
                chunk_id=chunk_id,
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return 0

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)