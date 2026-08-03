from typing import Any

from src.retrieval.schemas import EvidenceItem


def build_evidence_items(
    search_results: list[dict[str, Any]],
) -> list[EvidenceItem]:
    """
    Convert reranked Qdrant results into validated evidence items.

    Invalid individual results raise an error rather than being
    silently discarded because downstream agents must know exactly
    what evidence they received.
    """
    evidence: list[EvidenceItem] = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):
        evidence.append(
            EvidenceItem.from_search_result(
                result,
                evidence_index=index,
            )
        )

    return evidence