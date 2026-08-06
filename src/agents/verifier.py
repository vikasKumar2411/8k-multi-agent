import json
from typing import Any

from pydantic import ValidationError

from src.agents.schemas import AnalysisResult, QueryPlan
from src.agents.verification_schemas import (
    ClaimType,
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
    VerificationSummary,
)
from src.embeddings.ollama_chat_client import OllamaChatClient
from src.retrieval.schemas import (
    CoverageResult,
    EvidenceItem,
)


class VerificationAgent:
    """
    Verifies analyzer findings against retrieved SEC evidence.

    This agent does not generate the final user-facing answer.
    """

    def __init__(
        self,
        chat_client: OllamaChatClient | None = None,
    ) -> None:
        self.chat_client = chat_client or OllamaChatClient()

    @staticmethod
    def build_system_prompt() -> str:
        return """
You are the Verification Agent for an SEC Form 8-K
Event Intelligence Copilot.

Your task is to independently verify structured claims against
the supplied evidence.

Rules:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not answer the user's question conversationally.
4. Return only the structured verification object.
5. Verify each summary, metric, and material limitation separately.
6. A claim is verified only when the cited evidence directly supports it.
7. A claim is partially_supported when only part of the claim is supported.
8. A claim is unsupported when no supplied evidence supports it.
9. A claim is contradicted when supplied evidence directly conflicts with it.
10. Check exact metric values, units, and reporting periods.
11. Preserve the distinction between filing release date and reporting period.
12. Never treat a filing released in 2024 as proof that its metrics describe 2024.
13. Reject invented values, unsupported precision, or incorrect reporting periods.
14. A limitation can itself be contradicted by evidence.
15. Every verified or partially supported claim must cite valid evidence_ids.
16. corrected_claim should contain a concise evidence-supported correction
    when a claim is unsupported, contradicted, or only partially supported.
17. Do not silently repair claims. Record the original status and correction.
18. answerable=true when at least one material claim is verified and the
    final answer can clearly disclose coverage limitations.
19. verification_complete=false when important requested-company evidence
    is missing.
""".strip()

    @staticmethod
    def build_user_prompt(
        *,
        user_query: str,
        plan: QueryPlan,
        analysis: AnalysisResult,
        evidence: list[EvidenceItem],
        coverage: CoverageResult,
    ) -> str:
        compact_evidence = [
            {
                "evidence_id": item.evidence_id,
                "symbol": item.symbol,
                "company_name": item.company_name,
                "title": item.title,
                "filing_release_date": (
                    item.release_datetime.date().isoformat()
                ),
                "accession_number": item.accession_number,
                "chunk_id": item.chunk_id,
                "excerpt": item.excerpt,
                "chunk_text": item.chunk_text[:7000],
            }
            for item in evidence
        ]

        payload: dict[str, Any] = {
            "user_query": user_query,
            "query_plan": plan.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "coverage": coverage.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "analysis": analysis.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "evidence": compact_evidence,
        }

        return (
            "Verify every claim in this analysis against the "
            "provided SEC evidence.\n\n"
            + json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
        )

    def verify(
        self,
        *,
        user_query: str,
        plan: QueryPlan,
        analysis: AnalysisResult,
        evidence: list[EvidenceItem],
        coverage: CoverageResult,
    ) -> VerificationResult:
        cleaned_query = user_query.strip()

        if not cleaned_query:
            raise ValueError("user_query cannot be empty")

        if not plan.in_scope:
            raise ValueError(
                "Cannot verify an out-of-scope request"
            )

        if plan.clarification_needed:
            raise ValueError(
                "Cannot verify before clarification is resolved"
            )

        if not evidence:
            return self._empty_verification_result(
                coverage=coverage,
            )

        raw_verification = self.chat_client.generate_json(
            system_prompt=self.build_system_prompt(),
            user_prompt=self.build_user_prompt(
                user_query=cleaned_query,
                plan=plan,
                analysis=analysis,
                evidence=evidence,
                coverage=coverage,
            ),
            response_model=VerificationResult,
            temperature=0.0,
        )

        repaired = self._repair_verification(
            raw_verification=raw_verification,
            analysis=analysis,
            evidence=evidence,
            coverage=coverage,
        )

        try:
            return VerificationResult.model_validate(repaired)
        except ValidationError as exc:
            raise RuntimeError(
                "Verification Agent returned invalid output.\n"
                f"Raw output: {raw_verification}\n"
                f"Repaired output: {repaired}\n"
                f"Validation error: {exc}"
            ) from exc

    @staticmethod
    def _empty_verification_result(
        *,
        coverage: CoverageResult,
    ) -> VerificationResult:
        return VerificationResult(
            claims=[],
            verified_finding_symbols=[],
            missing_symbols=coverage.missing_symbols,
            summary=VerificationSummary(
                total_claims=0,
                verified_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                verified_ratio=0.0,
            ),
            answerable=True,
            verification_complete=False,
            limitations=[
                "No evidence was available for verification."
            ],
        )

    @staticmethod
    def _repair_verification(
        *,
        raw_verification: dict[str, Any],
        analysis: AnalysisResult,
        evidence: list[EvidenceItem],
        coverage: CoverageResult,
    ) -> dict[str, Any]:
        """
        Enforce evidence references, summary counts, and coverage rules.
        """
        repaired = dict(raw_verification)

        valid_evidence_ids = {
            item.evidence_id
            for item in evidence
        }

        valid_symbols = {
            item.symbol
            for item in evidence
        }

        raw_claims = repaired.get("claims")

        if not isinstance(raw_claims, list):
            raw_claims = []

        claims: list[dict[str, Any]] = []

        for index, raw_claim in enumerate(
            raw_claims,
            start=1,
        ):
            if not isinstance(raw_claim, dict):
                continue

            claim = dict(raw_claim)

            claim["claim_id"] = str(
                claim.get("claim_id")
                or f"claim-{index}"
            )

            claim_type = str(
                claim.get("claim_type")
                or ClaimType.SUMMARY.value
            ).strip().lower()

            valid_claim_types = {
                value.value
                for value in ClaimType
            }

            if claim_type not in valid_claim_types:
                claim_type = ClaimType.SUMMARY.value

            claim["claim_type"] = claim_type

            status = str(
                claim.get("status")
                or VerificationStatus.UNSUPPORTED.value
            ).strip().lower()

            valid_statuses = {
                value.value
                for value in VerificationStatus
            }

            if status not in valid_statuses:
                status = VerificationStatus.UNSUPPORTED.value

            claim["status"] = status

            raw_evidence_ids = (
                claim.get("evidence_ids") or []
            )

            if isinstance(raw_evidence_ids, str):
                raw_evidence_ids = [raw_evidence_ids]

            evidence_ids = sorted(
                {
                    str(evidence_id)
                    for evidence_id in raw_evidence_ids
                    if str(evidence_id) in valid_evidence_ids
                }
            )

            claim["evidence_ids"] = evidence_ids

            if (
                status == VerificationStatus.VERIFIED.value
                and not evidence_ids
            ):
                claim["status"] = (
                    VerificationStatus.UNSUPPORTED.value
                )

            symbol = claim.get("symbol")

            if symbol:
                normalized_symbol = str(symbol).strip().upper()
                claim["symbol"] = (
                    normalized_symbol
                    if normalized_symbol in valid_symbols
                    else None
                )
            else:
                claim["symbol"] = None

            claim_text = str(
                claim.get("claim_text") or ""
            ).strip()

            rationale = str(
                claim.get("rationale") or ""
            ).strip()

            if not claim_text:
                continue

            if not rationale:
                rationale = (
                    "The verifier did not provide a detailed rationale."
                )

            claim["claim_text"] = claim_text
            claim["rationale"] = rationale

            corrected_claim = str(
                claim.get("corrected_claim") or ""
            ).strip()

            claim["corrected_claim"] = (
                corrected_claim or None
            )

            claims.append(claim)

        repaired["claims"] = claims

        status_counts = {
            VerificationStatus.VERIFIED.value: 0,
            VerificationStatus.PARTIALLY_SUPPORTED.value: 0,
            VerificationStatus.UNSUPPORTED.value: 0,
            VerificationStatus.CONTRADICTED.value: 0,
        }

        for claim in claims:
            status_counts[claim["status"]] += 1

        total_claims = len(claims)
        verified_claims = status_counts[
            VerificationStatus.VERIFIED.value
        ]

        verified_ratio = (
            verified_claims / total_claims
            if total_claims
            else 0.0
        )

        repaired["summary"] = {
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "partially_supported_claims": status_counts[
                VerificationStatus.PARTIALLY_SUPPORTED.value
            ],
            "unsupported_claims": status_counts[
                VerificationStatus.UNSUPPORTED.value
            ],
            "contradicted_claims": status_counts[
                VerificationStatus.CONTRADICTED.value
            ],
            "verified_ratio": verified_ratio,
        }

        verified_symbols = sorted(
            {
                str(claim["symbol"]).strip().upper()
                for claim in claims
                if (
                    claim.get("symbol")
                    and claim["status"]
                    == VerificationStatus.VERIFIED.value
                )
            }
        )

        repaired["verified_finding_symbols"] = verified_symbols
        repaired["missing_symbols"] = list(
            coverage.missing_symbols
        )

        answerable = verified_claims > 0

        repaired["answerable"] = answerable
        repaired["verification_complete"] = (
            answerable
            and not coverage.missing_symbols
        )

        raw_limitations = repaired.get("limitations") or []

        if isinstance(raw_limitations, str):
            raw_limitations = [raw_limitations]

        limitations = [
            str(value).strip()
            for value in raw_limitations
            if str(value).strip()
        ]

        if coverage.missing_symbols:
            missing_text = ", ".join(
                coverage.missing_symbols
            )

            limitation = (
                "No matching evidence was found for requested "
                f"symbol(s): {missing_text}."
            )

            if limitation not in limitations:
                limitations.append(limitation)

        if not claims and analysis.findings:
            limitations.append(
                "The verifier did not produce claim-level results "
                "for the supplied analysis."
            )

        repaired["limitations"] = list(
            dict.fromkeys(limitations)
        )

        return repaired