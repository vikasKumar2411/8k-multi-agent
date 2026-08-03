from typing import Any


EVENT_ITEM_MAP: dict[str, set[str]] = {
    "leadership": {"5.02"},
    "earnings": {"2.02"},
    "restructuring": {"2.05"},
    "material_agreement": {"1.01"},
    "acquisition": {"1.01", "2.01"},
    "bankruptcy": {"1.03"},
    "cybersecurity": {"1.05"},
    "financial_obligation": {"2.03", "2.04"},
    "operational": {"2.02", "7.01", "8.01"},
}


TITLE_TERMS: dict[str, tuple[str, ...]] = {
    "leadership": (
        "resignation",
        "appointment",
        "leadership",
        "executive",
        "officer",
        "director",
        "retirement",
        "chairman",
        "chief financial officer",
        "chief executive officer",
    ),
    "earnings": (
        "financial results",
        "quarterly results",
        "earnings",
        "revenue",
        "profitability",
        "fiscal year",
        "outlook",
    ),
    "restructuring": (
        "restructuring",
        "workforce reduction",
        "layoff",
        "cost reduction",
        "organizational changes",
        "exit activities",
    ),
    "cybersecurity": (
        "cybersecurity",
        "cyber incident",
        "data breach",
        "security incident",
    ),
    "operational": (
        "production",
        "deliveries",
        "operational update",
        "manufacturing",
        "shipments",
        "sales update",
    ),
}


ACTUAL_EVENT_TERMS: dict[str, tuple[str, ...]] = {
    "leadership": (
        "resigned",
        "resignation of",
        "appointed",
        "appointment of",
        "retire effective",
        "will retire",
        "terminated",
        "role and position",
        "departure of",
        "stepped down",
        "named as",
    ),
    "earnings": (
        "announced its financial results",
        "reports financial results",
        "quarter ended",
        "fiscal quarter",
        "net sales",
        "net income",
        "revenue",
        "earnings per share",
    ),
    "restructuring": (
        "restructuring plan",
        "workforce reduction",
        "position will be eliminated",
        "organizational changes",
        "termination benefits",
        "exit activities",
    ),
    "cybersecurity": (
        "cybersecurity incident",
        "material cybersecurity incident",
        "unauthorized access",
        "data breach",
        "security incident",
        "network intrusion",
    ),
    "operational": (
        "produced approximately",
        "delivered over",
        "production results",
        "delivery results",
        "units produced",
        "units delivered",
    ),
}


BOILERPLATE_TERMS: tuple[str, ...] = (
    "articles of incorporation",
    "certificate of incorporation",
    "amended and restated bylaws",
    "these bylaws",
    "vacancy or vacancies",
    "directors then in office",
    "ordinary resolution",
    "power to fill such vacancy",
    "shall hold office",
    "underwriting agreement",
    "representations and warranties",
)


def calculate_rerank_score(
    result: dict[str, Any],
    *,
    event_category: str | None,
) -> float:
    """
    Combine vector similarity with deterministic SEC and event signals.

    The score is intentionally bounded by signal family:
    - one SEC item boost,
    - one title boost,
    - one actual-event-language boost,
    - one boilerplate penalty.

    This prevents overlapping keywords from adding excessive score.
    """
    base_score = float(result.get("score", 0.0))
    payload = result.get("payload") or {}

    title = str(payload.get("title", "")).lower()
    chunk_text = str(payload.get("chunk_text", "")).lower()

    chunk_item_numbers = set(
        payload.get("chunk_item_numbers") or []
    )
    filing_item_numbers = set(
        payload.get("filing_item_numbers") or []
    )

    rerank_score = base_score

    if not event_category:
        return rerank_score

    normalized_category = event_category.strip().lower()
    expected_items = EVENT_ITEM_MAP.get(normalized_category, set())

    # Stronger signal when the relevant item number appears in this chunk.
    if expected_items.intersection(chunk_item_numbers):
        rerank_score += 0.18

    # Weaker signal when the item appears elsewhere in the same filing.
    elif expected_items.intersection(filing_item_numbers):
        rerank_score += 0.08

    # Apply at most one title-level relevance boost.
    if any(
        term in title
        for term in TITLE_TERMS.get(normalized_category, ())
    ):
        rerank_score += 0.08

    # Apply at most one actual-event-language boost.
    if any(
        term in chunk_text
        for term in ACTUAL_EVENT_TERMS.get(
            normalized_category,
            (),
        )
    ):
        rerank_score += 0.10

    # Apply at most one boilerplate penalty.
    if any(
        term in chunk_text
        for term in BOILERPLATE_TERMS
    ):
        rerank_score -= 0.15

    return rerank_score


def rerank_results(
    results: list[dict[str, Any]],
    *,
    event_category: str | None,
    limit: int,
    max_chunks_per_filing: int = 2,
) -> list[dict[str, Any]]:
    """
    Rerank vector-search results and enforce filing-level diversity.

    Args:
        results:
            Raw Qdrant results. Each result should contain:
            - score
            - payload

        event_category:
            Optional normalized event category such as:
            - leadership
            - earnings
            - restructuring
            - cybersecurity
            - operational

        limit:
            Maximum number of final results.

        max_chunks_per_filing:
            Maximum number of chunks allowed from one accession number.

    Returns:
        Reranked and diversified results.
    """
    if limit <= 0:
        return []

    if max_chunks_per_filing <= 0:
        raise ValueError(
            "max_chunks_per_filing must be positive"
        )

    scored_results: list[dict[str, Any]] = []

    for result in results:
        updated_result = dict(result)

        vector_score = float(result.get("score", 0.0))
        rerank_score = calculate_rerank_score(
            result,
            event_category=event_category,
        )

        updated_result["vector_score"] = vector_score
        updated_result["rerank_score"] = rerank_score

        scored_results.append(updated_result)

    scored_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    accession_counts: dict[str, int] = {}

    for result in scored_results:
        payload = result.get("payload") or {}

        accession_number = str(
            payload.get("accession_number", "")
        ).strip()

        # Fall back to point ID if accession metadata is unavailable.
        diversity_key = (
            accession_number
            or str(result.get("id", "")).strip()
        )

        count = accession_counts.get(diversity_key, 0)

        if count >= max_chunks_per_filing:
            continue

        selected.append(result)
        accession_counts[diversity_key] = count + 1

        if len(selected) >= limit:
            break

    return selected