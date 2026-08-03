import re
from datetime import date
from typing import Any

from pydantic import ValidationError

from src.agents.schemas import (
    EventCategory,
    QueryPlan,
    TaskType,
)
from src.config.settings import get_settings
from src.embeddings.ollama_chat_client import OllamaChatClient


SUPPORTED_SCOPE = """
The copilot answers questions grounded in indexed SEC Form 8-K filings.

Supported event categories:
- leadership
- earnings
- restructuring
- material_agreement
- acquisition
- bankruptcy
- cybersecurity
- financial_obligation
- operational
- other

Supported task types:
- event_search
- company_timeline
- company_comparison
- event_summary
- metric_extraction
- filing_lookup

In-scope examples:
- What leadership changes did Microsoft disclose in 2024?
- Compare Tesla and Ford operational updates.
- Find cybersecurity incidents disclosed by public companies.
- Extract production and delivery figures reported by Tesla.
- Summarize a specific 8-K filing.
- Show restructuring events for Amazon.

Out-of-scope examples:
- Should I buy Tesla stock?
- Predict NVIDIA's stock price.
- Calculate a company valuation.
- Provide legal advice.
- Answer unrelated general-knowledge questions.
- Perform analysis requiring complete 10-K, 10-Q, stock-price,
  valuation, market, or external-news data.

The system is a research copilot, not an investment, legal,
valuation, prediction, or trading system.
""".strip()


COMPANY_SYMBOL_MAP: dict[str, str] = {
    "tesla": "TSLA",
    "ford": "F",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
}


CATEGORY_QUERY_DEFAULTS: dict[EventCategory, str] = {
    EventCategory.LEADERSHIP: (
        "executive resignation appointment retirement "
        "leadership transition officer director"
    ),
    EventCategory.EARNINGS: (
        "quarterly earnings financial results revenue "
        "net income outlook"
    ),
    EventCategory.RESTRUCTURING: (
        "restructuring workforce reduction layoffs "
        "cost reduction organizational changes"
    ),
    EventCategory.MATERIAL_AGREEMENT: (
        "material definitive agreement contract partnership"
    ),
    EventCategory.ACQUISITION: (
        "acquisition merger divestiture transaction"
    ),
    EventCategory.BANKRUPTCY: (
        "bankruptcy receivership restructuring proceedings"
    ),
    EventCategory.CYBERSECURITY: (
        "cybersecurity incident data breach unauthorized access"
    ),
    EventCategory.FINANCIAL_OBLIGATION: (
        "debt financing credit agreement financial obligation"
    ),
    EventCategory.OPERATIONAL: (
        "production deliveries manufacturing shipments "
        "operational results"
    ),
    EventCategory.OTHER: "material corporate event",
}


OUT_OF_SCOPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(should i buy|should i sell|buy .* stock|sell .* stock)\b",
        "Investment recommendations are outside the copilot's scope.",
    ),
    (
        r"\b(predict|forecast)\b.*\b(stock|share|price)\b",
        "Stock-price predictions are outside the copilot's scope.",
    ),
    (
        r"\b(valuation|fair value|price target|discounted cash flow|dcf)\b",
        "Company valuation is outside the copilot's scope.",
    ),
    (
        r"\b(legal advice|is this legal|lawsuit strategy)\b",
        "Legal advice is outside the copilot's scope.",
    ),
)


CATEGORY_PATTERNS: tuple[
    tuple[EventCategory, tuple[str, ...]],
    ...
] = (
    (
        EventCategory.LEADERSHIP,
        (
            "executive",
            "resignation",
            "resignations",
            "resigned",
            "appointment",
            "appointments",
            "appointed",
            "leadership",
            "officer",
            "director",
            "retirement",
            "cfo",
            "ceo",
            "chairman",
        ),
    ),
    (
        EventCategory.EARNINGS,
        (
            "earnings",
            "financial results",
            "quarterly results",
            "revenue",
            "net income",
            "profitability",
            "outlook",
            "fiscal quarter",
        ),
    ),
    (
        EventCategory.RESTRUCTURING,
        (
            "restructuring",
            "layoff",
            "layoffs",
            "workforce reduction",
            "cost reduction",
            "organizational changes",
            "exit activities",
        ),
    ),
    (
        EventCategory.CYBERSECURITY,
        (
            "cybersecurity",
            "cyber incident",
            "cyber incidents",
            "data breach",
            "security incident",
            "unauthorized access",
        ),
    ),
    (
        EventCategory.ACQUISITION,
        (
            "acquisition",
            "acquire",
            "acquired",
            "merger",
            "divestiture",
            "transaction completed",
        ),
    ),
    (
        EventCategory.BANKRUPTCY,
        (
            "bankruptcy",
            "receivership",
            "chapter 11",
        ),
    ),
    (
        EventCategory.FINANCIAL_OBLIGATION,
        (
            "debt",
            "credit agreement",
            "financial obligation",
            "financing",
            "loan agreement",
        ),
    ),
    (
        EventCategory.MATERIAL_AGREEMENT,
        (
            "material agreement",
            "definitive agreement",
            "partnership agreement",
            "contract",
        ),
    ),
    (
        EventCategory.OPERATIONAL,
        (
            "operational",
            "operations",
            "production",
            "deliveries",
            "manufacturing",
            "shipments",
            "units produced",
            "units delivered",
        ),
    ),
)


BROAD_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\bwhat happened (at|with|to)\b",
    r"\bshow (me )?(the )?events for\b",
    r"\bwhat did .+ disclose\b",
    r"\bfind .+ corporate events\b",
)


class QueryPlanningAgent:
    def __init__(
        self,
        chat_client: OllamaChatClient | None = None,
    ) -> None:
        self.chat_client = chat_client or OllamaChatClient()
        self.settings = get_settings()

    @staticmethod
    def build_system_prompt() -> str:
        today = date.today().isoformat()

        return f"""
You are the Query Planning Agent for an SEC 8-K Event
Intelligence Copilot.

Current date: {today}

Convert the user request into a structured retrieval plan.

{SUPPORTED_SCOPE}

Mandatory rules:

1. Do not answer the user's research question.
2. Return every field required by the JSON schema.
3. Only produce the structured query plan.
4. Questions about corporate events disclosed in 8-K filings
   are in scope.
5. For every out-of-scope request, provide rejection_reason.
6. For every in-scope request, provide retrieval_query.
7. Normalize ticker symbols to uppercase.
8. Convert explicit years into complete date ranges.
9. Broad company-event questions are in scope but require
   clarification.
10. Investment advice, stock prediction, valuation, legal advice,
    and unrelated general knowledge are out of scope.
11. retrieval_query must contain concise natural-language semantic
    search text. Never generate SQL, database syntax, code, JSON,
    metadata filters, or API calls. Dates, ticker symbols, and event
    categories belong in their dedicated fields.
""".strip()

    @staticmethod
    def build_user_prompt(user_query: str) -> str:
        return f"""
Create a retrieval plan for this request:

{user_query}
""".strip()

    @staticmethod
    def _extract_explicit_year(
        user_query: str,
    ) -> int | None:
        match = re.search(
            r"\b(19\d{2}|20\d{2})\b",
            user_query,
        )

        return int(match.group(1)) if match else None

    @staticmethod
    def _deterministic_scope_rejection(
        user_query: str,
    ) -> str | None:
        lowered = user_query.lower()

        for pattern, reason in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, lowered):
                return reason

        unrelated_patterns = (
            r"\bcapital of\b",
            r"\bweather\b",
            r"\brecipe\b",
            r"\bsports score\b",
        )

        if any(
            re.search(pattern, lowered)
            for pattern in unrelated_patterns
        ):
            return (
                "The request is unrelated to SEC 8-K corporate "
                "event research."
            )

        return None

    @staticmethod
    def _resolve_known_symbols(
        user_query: str,
    ) -> list[str]:
        lowered = user_query.lower()
        symbols: set[str] = set()

        for company_name, symbol in COMPANY_SYMBOL_MAP.items():
            if re.search(
                rf"\b{re.escape(company_name)}\b",
                lowered,
            ):
                symbols.add(symbol)

        return sorted(symbols)

    @staticmethod
    def _normalize_symbols(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            return []

        return sorted(
            {
                str(symbol).strip().upper()
                for symbol in candidates
                if str(symbol).strip()
            }
        )

    @staticmethod
    def _infer_event_category(
        user_query: str,
    ) -> EventCategory:
        lowered = user_query.lower()

        for category, terms in CATEGORY_PATTERNS:
            if any(term in lowered for term in terms):
                return category

        return EventCategory.OTHER

    @staticmethod
    def _is_broad_question(
        user_query: str,
    ) -> bool:
        lowered = user_query.lower()

        return any(
            re.search(pattern, lowered)
            for pattern in BROAD_QUESTION_PATTERNS
        )

    @staticmethod
    def _is_clearly_in_scope(
        user_query: str,
    ) -> bool:
        lowered = user_query.lower()
        known_symbols = QueryPlanningAgent._resolve_known_symbols(
            user_query
        )

        category_match = any(
            term in lowered
            for _, terms in CATEGORY_PATTERNS
            for term in terms
        )

        if category_match:
            return True

        sec_terms = (
            "8-k",
            "8k",
            "filing",
            "filings",
            "disclosed",
            "disclosure",
            "disclosures",
            "corporate event",
            "corporate events",
        )

        if any(term in lowered for term in sec_terms):
            return True

        broad_event_terms = (
            "event",
            "events",
            "disclosure",
            "disclosures",
            "disclosed",
            "announcement",
            "announcements",
        )

        if (
            known_symbols
            and any(term in lowered for term in broad_event_terms)
        ):
            return True

        if QueryPlanningAgent._is_broad_question(user_query):
            return True

        return False

    @staticmethod
    def _infer_task_type(
        user_query: str,
        *,
        symbol_count: int,
    ) -> TaskType:
        lowered = user_query.lower()

        if "compare" in lowered or symbol_count > 1:
            return TaskType.COMPANY_COMPARISON

        if any(
            term in lowered
            for term in (
                "metric",
                "metrics",
                "figures",
                "numbers",
                "how many",
                "how much",
            )
        ):
            return TaskType.METRIC_EXTRACTION

        if any(
            term in lowered
            for term in (
                "timeline",
                "chronological",
                "over time",
            )
        ):
            return TaskType.COMPANY_TIMELINE

        if any(
            term in lowered
            for term in (
                "summarize",
                "summary",
            )
        ):
            return TaskType.EVENT_SUMMARY

        if any(
            term in lowered
            for term in (
                "find filing",
                "find filings",
                "locate filing",
                "locate filings",
            )
        ):
            return TaskType.FILING_LOOKUP

        return TaskType.EVENT_SEARCH

    @staticmethod
    def _normalize_enum_value(
        value: Any,
        enum_type: type[EventCategory] | type[TaskType],
        default: EventCategory | TaskType,
    ) -> str:
        if value is None:
            return default.value

        normalized = str(value).strip().lower()
        valid_values = {item.value for item in enum_type}

        return (
            normalized
            if normalized in valid_values
            else default.value
        )

    @staticmethod
    def _build_retrieval_query(
        *,
        category: EventCategory,
        user_query: str,
    ) -> str:
        """
        Build deterministic natural-language text for embedding search.

        Company, date, and category restrictions are handled separately
        through metadata filters and reranking.
        """
        base_query = CATEGORY_QUERY_DEFAULTS[category]
        lowered = user_query.lower()

        additional_terms: list[str] = []

        if category == EventCategory.LEADERSHIP:
            if "resignation" in lowered or "resignations" in lowered:
                additional_terms.append("executive resignation")

            if "appointment" in lowered or "appointments" in lowered:
                additional_terms.append("executive appointment")

        elif category == EventCategory.OPERATIONAL:
            if "metric" in lowered or "metrics" in lowered:
                additional_terms.extend(
                    [
                        "reported operational metrics",
                        "quantitative production figures",
                    ]
                )

        elif category == EventCategory.EARNINGS:
            if "quarterly" in lowered:
                additional_terms.append(
                    "quarterly financial results"
                )

        elif category == EventCategory.CYBERSECURITY:
            if "incident" in lowered or "incidents" in lowered:
                additional_terms.append(
                    "reported cybersecurity incidents"
                )

        terms = [
            base_query,
            *additional_terms,
        ]

        return " ".join(dict.fromkeys(terms))

    def _repair_plan(
        self,
        raw_plan: dict[str, Any],
        *,
        user_query: str,
    ) -> dict[str, Any]:
        repaired = dict(raw_plan)

        deterministic_rejection = (
            self._deterministic_scope_rejection(user_query)
        )

        if deterministic_rejection:
            return {
                "in_scope": False,
                "rejection_reason": deterministic_rejection,
                "symbols": [],
                "start_date": None,
                "end_date": None,
                "event_category": None,
                "task_type": TaskType.UNSUPPORTED.value,
                "retrieval_query": None,
                "clarification_needed": False,
                "clarification_question": None,
            }

        known_symbols = self._resolve_known_symbols(user_query)
        model_symbols = self._normalize_symbols(
            repaired.get("symbols")
        )

        symbols = sorted(
            set(model_symbols).union(known_symbols)
        )

        deterministic_in_scope = self._is_clearly_in_scope(
            user_query
        )

        in_scope = (
            True
            if deterministic_in_scope
            else bool(repaired.get("in_scope", False))
        )

        if not in_scope:
            rejection_reason = str(
                repaired.get("rejection_reason")
                or (
                    "The request is outside the SEC 8-K event "
                    "research scope."
                )
            ).strip()

            return {
                "in_scope": False,
                "rejection_reason": rejection_reason,
                "symbols": [],
                "start_date": None,
                "end_date": None,
                "event_category": None,
                "task_type": TaskType.UNSUPPORTED.value,
                "retrieval_query": None,
                "clarification_needed": False,
                "clarification_question": None,
            }

        inferred_category = self._infer_event_category(
            user_query
        )

        model_category = self._normalize_enum_value(
            repaired.get("event_category"),
            EventCategory,
            inferred_category,
        )

        if model_category == EventCategory.OTHER.value:
            category = inferred_category
        else:
            category = EventCategory(model_category)

        task_type = self._infer_task_type(
            user_query,
            symbol_count=len(symbols),
        )

        explicit_year = self._extract_explicit_year(user_query)

        start_date = repaired.get("start_date")
        end_date = repaired.get("end_date")

        if explicit_year is not None:
            start_date = start_date or f"{explicit_year}-01-01"
            end_date = end_date or f"{explicit_year}-12-31"

        retrieval_query = self._build_retrieval_query(
            category=category,
            user_query=user_query,
        )

        clarification_needed = self._is_broad_question(
            user_query
        )

        clarification_question = None

        if clarification_needed:
            clarification_question = (
                "What type of corporate event or date range "
                "should I search?"
            )
            category = EventCategory.OTHER
            task_type = TaskType.EVENT_SEARCH
            retrieval_query = CATEGORY_QUERY_DEFAULTS[
                EventCategory.OTHER
            ]

        return {
            "in_scope": True,
            "rejection_reason": None,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "event_category": category.value,
            "task_type": task_type.value,
            "retrieval_query": retrieval_query,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
        }

    def plan(self, user_query: str) -> QueryPlan:
        cleaned_query = user_query.strip()

        if not cleaned_query:
            raise ValueError("user_query cannot be empty")

        raw_plan = self.chat_client.generate_json(
            system_prompt=self.build_system_prompt(),
            user_prompt=self.build_user_prompt(cleaned_query),
            response_model=QueryPlan,
            temperature=self.settings.query_planner_temperature,
        )

        repaired_plan = self._repair_plan(
            raw_plan,
            user_query=cleaned_query,
        )

        try:
            return QueryPlan.model_validate(repaired_plan)
        except ValidationError as exc:
            raise RuntimeError(
                "Query planner returned an invalid plan after "
                f"deterministic repair.\n"
                f"Raw plan: {raw_plan}\n"
                f"Repaired plan: {repaired_plan}\n"
                f"Validation error: {exc}"
            ) from exc