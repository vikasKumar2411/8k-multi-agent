from src.data.cleaning import clean_filing_text, extract_item_numbers
from src.data.schemas import FilingChunk, FilingDocument


def split_long_paragraph(
    paragraph: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a paragraph that is larger than the configured chunk size."""
    chunks: list[str] = []
    start = 0

    while start < len(paragraph):
        end = min(start + chunk_size, len(paragraph))

        # Prefer ending at a word boundary when possible.
        if end < len(paragraph):
            word_boundary = paragraph.rfind(" ", start, end)

            if word_boundary > start:
                end = word_boundary

        chunk = paragraph[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(paragraph):
            break

        next_start = end - chunk_overlap

        # Prevent a non-advancing loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def split_text(
    text: str,
    *,
    chunk_size: int = 2500,
    chunk_overlap: int = 300,
) -> list[str]:
    """Split filing text into paragraph-aware overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_parts:
                completed_chunk = " ".join(current_parts).strip()

                if completed_chunk:
                    chunks.append(completed_chunk)

                current_parts = []
                current_length = 0

            long_paragraph_chunks = split_long_paragraph(
                paragraph,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks.extend(long_paragraph_chunks)
            continue

        separator_length = 1 if current_parts else 0
        projected_length = (
            current_length
            + separator_length
            + len(paragraph)
        )

        if current_parts and projected_length > chunk_size:
            completed_chunk = " ".join(current_parts).strip()

            if completed_chunk:
                chunks.append(completed_chunk)

            overlap_text = completed_chunk[-chunk_overlap:].strip()

            # Try not to start the overlap in the middle of a word.
            if overlap_text:
                first_space = overlap_text.find(" ")

                if first_space != -1:
                    overlap_text = overlap_text[first_space + 1 :].strip()

            current_parts = []

            if overlap_text:
                current_parts.append(overlap_text)

            current_parts.append(paragraph)

            current_length = sum(
                len(part) for part in current_parts
            ) + max(0, len(current_parts) - 1)

        else:
            current_parts.append(paragraph)
            current_length = projected_length

    if current_parts:
        completed_chunk = " ".join(current_parts).strip()

        if completed_chunk:
            chunks.append(completed_chunk)

    return chunks


def chunk_filing(
    filing: FilingDocument,
    *,
    chunk_size: int = 2500,
    chunk_overlap: int = 300,
) -> list[FilingChunk]:
    """Clean and split one SEC filing into structured chunks."""
    cleaned_text = clean_filing_text(filing.raw_text)

    filing_item_numbers = extract_item_numbers(cleaned_text)

    text_chunks = split_text(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        FilingChunk(
            accession_number=filing.accession_number,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            release_datetime=filing.release_datetime,
            title=filing.title,
            filing_type=filing.filing_type,
            symbol=filing.symbol,
            company_name=filing.company_name,
            keywords=filing.keywords,
            exchange=filing.exchange,
            excerpt=filing.excerpt,
            chunk_item_numbers=extract_item_numbers(chunk_text),
            filing_item_numbers=filing_item_numbers,
            dataset_version=filing.dataset_version,
        )
        for chunk_id, chunk_text in enumerate(text_chunks)
    ]