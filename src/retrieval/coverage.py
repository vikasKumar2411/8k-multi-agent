from src.agents.schemas import QueryPlan, TaskType
from src.retrieval.schemas import (
    CoverageResult,
    EvidenceItem,
)


def validate_evidence_coverage(
    *,
    plan: QueryPlan,
    evidence: list[EvidenceItem],
) -> CoverageResult:
    """
    Verify that retrieval covered the companies requested by the plan.

    For company comparisons, every requested symbol must have at least
    one evidence item.

    For single-company queries, the requested symbol must appear.

    For broad event searches without symbols, coverage is based on
    whether any evidence was found.
    """
    if not plan.in_scope:
        raise ValueError(
            "Cannot validate evidence for an out-of-scope plan"
        )

    if plan.clarification_needed:
        raise ValueError(
            "Cannot validate evidence before clarification"
        )

    requested_symbols = sorted(
        {
            symbol.strip().upper()
            for symbol in plan.symbols
            if symbol.strip()
        }
    )

    retrieved_symbols = sorted(
        {
            item.symbol.strip().upper()
            for item in evidence
            if item.symbol.strip()
        }
    )

    missing_symbols = sorted(
        set(requested_symbols) - set(retrieved_symbols)
    )

    filing_count = len(
        {
            item.accession_number
            for item in evidence
        }
    )

    evidence_count = len(evidence)

    if not requested_symbols:
        complete = evidence_count > 0
    else:
        complete = not missing_symbols and evidence_count > 0

    retry_recommended = _should_retry(
        plan=plan,
        evidence_count=evidence_count,
        missing_symbols=missing_symbols,
    )

    message = _build_coverage_message(
        requested_symbols=requested_symbols,
        retrieved_symbols=retrieved_symbols,
        missing_symbols=missing_symbols,
        evidence_count=evidence_count,
        complete=complete,
    )

    return CoverageResult(
        requested_symbols=requested_symbols,
        retrieved_symbols=retrieved_symbols,
        missing_symbols=missing_symbols,
        evidence_count=evidence_count,
        filing_count=filing_count,
        complete=complete,
        retry_recommended=retry_recommended,
        message=message,
    )


def _should_retry(
    *,
    plan: QueryPlan,
    evidence_count: int,
    missing_symbols: list[str],
) -> bool:
    if evidence_count == 0:
        return True

    if (
        plan.task_type == TaskType.COMPANY_COMPARISON
        and missing_symbols
    ):
        return True

    if plan.symbols and missing_symbols:
        return True

    return False


def _build_coverage_message(
    *,
    requested_symbols: list[str],
    retrieved_symbols: list[str],
    missing_symbols: list[str],
    evidence_count: int,
    complete: bool,
) -> str:
    if evidence_count == 0:
        return (
            "No matching evidence was found in the indexed corpus."
        )

    if missing_symbols:
        missing_text = ", ".join(missing_symbols)
        retrieved_text = (
            ", ".join(retrieved_symbols)
            if retrieved_symbols
            else "none"
        )

        return (
            f"Evidence was retrieved for {retrieved_text}, but no "
            f"matching evidence was found for {missing_text}."
        )

    if requested_symbols and complete:
        covered_text = ", ".join(requested_symbols)

        return (
            f"Evidence coverage is complete for {covered_text}."
        )

    return (
        f"Retrieved {evidence_count} evidence item"
        f"{'' if evidence_count == 1 else 's'}."
    )