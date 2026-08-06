from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    PointStruct,
)

from src.config.settings import get_settings
from src.observability import (
    OperationType,
    observe_operation,
)
from src.retrieval.reranker import (
    rerank_results,
)


class SECQdrantStore:
    def __init__(self) -> None:
        settings = get_settings()

        self.collection_name = (
            settings.qdrant_collection
        )

        self.qdrant_url = (
            settings.qdrant_url
        )

        self.client = QdrantClient(
            url=self.qdrant_url,
        )

    @staticmethod
    def make_point_id(
        accession_number: str,
        chunk_id: int,
    ) -> str:
        key = (
            f"{accession_number}:{chunk_id}"
        )

        return str(
            uuid5(
                NAMESPACE_URL,
                key,
            )
        )

    def upsert_chunk(
        self,
        *,
        accession_number: str,
        chunk_id: int,
        vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        point_id = self.make_point_id(
            accession_number=accession_number,
            chunk_id=chunk_id,
        )

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        with observe_operation(
            operation_type=OperationType.DATABASE,
            operation_name="qdrant_upsert_chunk",
            provider="qdrant",
            attributes={
                "collection_name": (
                    self.collection_name
                ),
                "point_count": 1,
                "vector_dimension": len(vector),
                "wait_for_completion": True,
            },
        ) as observation:
            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=[point],
                wait=True,
            )

            observation.update_attributes(
                {
                    "upserted_point_count": 1,
                    "upsert_success": True,
                }
            )

        return point_id

    def search(
        self,
        *,
        query_vector: list[float],
        limit: int = 5,
        candidate_limit: int | None = None,
        query_filter: Filter | None = None,
        event_category: str | None = None,
        max_chunks_per_filing: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Search indexed SEC filing chunks and apply deterministic
        reranking.
        """
        if limit <= 0:
            return []

        if not query_vector:
            raise ValueError(
                "query_vector cannot be empty"
            )

        resolved_candidate_limit = (
            candidate_limit
            if candidate_limit is not None
            else max(limit * 6, 30)
        )

        if resolved_candidate_limit < limit:
            raise ValueError(
                "candidate_limit cannot be smaller than limit"
            )

        with observe_operation(
            operation_type=(
                OperationType.VECTOR_SEARCH
            ),
            operation_name=(
                "qdrant_similarity_search"
            ),
            provider="qdrant",
            attributes={
                "collection_name": (
                    self.collection_name
                ),
                "query_vector_dimension": (
                    len(query_vector)
                ),
                "requested_limit": limit,
                "candidate_limit": (
                    resolved_candidate_limit
                ),
                "filter_applied": (
                    query_filter is not None
                ),
                "event_category": event_category,
                "with_payload": True,
                "with_vectors": False,
            },
        ) as observation:
            response = self.client.query_points(
                collection_name=(
                    self.collection_name
                ),
                query=query_vector,
                query_filter=query_filter,
                limit=resolved_candidate_limit,
                with_payload=True,
                with_vectors=False,
            )

            raw_results = [
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": (
                        point.payload
                        or {}
                    ),
                }
                for point in response.points
            ]

            vector_scores = [
                float(result["score"])
                for result in raw_results
            ]

            observation.update_attributes(
                {
                    "result_count": len(raw_results),
                    "empty_result": (
                        len(raw_results) == 0
                    ),
                    "top_vector_score": (
                        max(vector_scores)
                        if vector_scores
                        else None
                    ),
                    "lowest_vector_score": (
                        min(vector_scores)
                        if vector_scores
                        else None
                    ),
                }
            )

        # Reranking has its own operation event, allowing us to
        # distinguish database latency from ranking latency.
        return rerank_results(
            raw_results,
            event_category=event_category,
            limit=limit,
            max_chunks_per_filing=(
                max_chunks_per_filing
            ),
        )

    def upsert_chunks(
        self,
        *,
        vectors: Sequence[list[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> int:
        if len(vectors) != len(payloads):
            raise ValueError(
                "vectors and payloads must have equal length"
            )

        points: list[PointStruct] = []

        for vector, payload in zip(
            vectors,
            payloads,
            strict=True,
        ):
            accession_number = str(
                payload["accession_number"]
            )

            chunk_id = int(
                payload["chunk_id"]
            )

            point_id = self.make_point_id(
                accession_number=(
                    accession_number
                ),
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

        vector_dimensions = {
            len(vector)
            for vector in vectors
        }

        with observe_operation(
            operation_type=OperationType.DATABASE,
            operation_name="qdrant_upsert_chunks",
            provider="qdrant",
            attributes={
                "collection_name": (
                    self.collection_name
                ),
                "point_count": len(points),
                "vector_dimensions": sorted(
                    vector_dimensions
                ),
                "wait_for_completion": True,
            },
        ) as observation:
            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )

            observation.update_attributes(
                {
                    "upserted_point_count": len(points),
                    "upsert_success": True,
                }
            )

        return len(points)