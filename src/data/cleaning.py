import re


ITEM_PATTERN = re.compile(
    r"\bItem\s+(\d+\.\d+)\b",
    flags=re.IGNORECASE,
)


def clean_filing_text(text: str) -> str:
    """Normalize common formatting problems in raw SEC filing text."""
    if not text:
        return ""

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\r", "\n")

    # Add a line break when an SEC item heading is joined to text.
    cleaned = re.sub(
        r"(?i)(Item\s+\d+\.\d+)(?=[A-Z])",
        r"\1\n",
        cleaned,
    )

    # Preserve paragraph boundaries while normalizing internal whitespace.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def extract_item_numbers(text: str) -> list[str]:
    """Extract unique SEC 8-K item numbers from a text chunk."""
    return sorted(set(ITEM_PATTERN.findall(text)))