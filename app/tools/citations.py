import re

from ..state import ResearchSource


def check_citations(
    article: str,
    sources: list[ResearchSource],
) -> list[str]:
    available_ids = {source.source_id for source in sources}
    cited_ids = set(re.findall(r"\[(S[0-9]+)\]", article))
    problems = []

    if not sources:
        problems.append(
            "No sources are available. Relevant documents are needed."
        )
        return problems

    if not cited_ids:
        problems.append(
            "Add source citations such as [S1] to support factual claims."
        )

    unknown_ids = cited_ids - available_ids
    if unknown_ids:
        references = ", ".join(sorted(unknown_ids))
        problems.append(
            f"Remove or correct unknown source references: {references}."
        )

    return problems