import re
from pathlib import Path

from ..state import ResearchSource


def search_documents(
    topic: str,
    directory: Path,
    limit: int = 3,
) -> list[ResearchSource]:
    if limit < 1:
        raise ValueError("limit must be positive.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    terms = set(re.findall(r"\w+", topic.casefold()))
    if not terms:
        return []

    matches = []

    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        for paragraph in re.split(r"\n\s*\n", text):
            excerpt = paragraph.strip()
            words = set(re.findall(r"\w+", excerpt.casefold()))
            score = len(terms & words)

            if score:
                matches.append((score, path, excerpt))

    matches.sort(key=lambda match: match[0], reverse=True)

    return [
        ResearchSource(
            source_id=f"S{index}",
            title=path.stem,
            location=str(path.resolve()),
            excerpt=excerpt,
        )
        for index, (_, path, excerpt) in enumerate(
            matches[:limit], start=1
        )
    ]