from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimType(StrEnum):
    SUMMARY = "summary"
    METRIC = "metric"
    LIMITATION = "limitation"


class ClaimVerification(BaseModel):
    """
    Verification outcome for one extracted claim.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    claim_id: str = Field(min_length=1)
    claim_type: ClaimType

    symbol: str | None = None
    claim_text: str = Field(min_length=1)

    status: VerificationStatus
    rationale: str = Field(min_length=1)

    evidence_ids: list[str] = Field(default_factory=list)

    corrected_claim: str | None = None

    @model_validator(mode="after")
    def validate_verification(self) -> "ClaimVerification":
        self.evidence_ids = sorted(
            {
                evidence_id.strip()
                for evidence_id in self.evidence_ids
                if evidence_id.strip()
            }
        )

        if (
            self.status == VerificationStatus.VERIFIED
            and not self.evidence_ids
        ):
            raise ValueError(
                "Verified claims require at least one evidence_id"
            )

        return self


class VerificationSummary(BaseModel):
    """
    Aggregate counts across all verified claims.
    """

    model_config = ConfigDict(extra="forbid")

    total_claims: int = Field(ge=0)
    verified_claims: int = Field(ge=0)
    partially_supported_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    contradicted_claims: int = Field(ge=0)

    verified_ratio: float = Field(ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """
    Structured output produced by the Verification Agent.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    claims: list[ClaimVerification] = Field(default_factory=list)

    verified_finding_symbols: list[str] = Field(
        default_factory=list
    )
    missing_symbols: list[str] = Field(default_factory=list)

    summary: VerificationSummary

    answerable: bool
    verification_complete: bool

    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_result(self) -> "VerificationResult":
        self.verified_finding_symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in self.verified_finding_symbols
                if symbol.strip()
            }
        )

        self.missing_symbols = sorted(
            {
                symbol.strip().upper()
                for symbol in self.missing_symbols
                if symbol.strip()
            }
        )

        return self