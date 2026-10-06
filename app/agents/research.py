from pathlib import Path

from ..state import AgentState
from ..tools.local_search import search_documents


class ResearchAgent:
    def __init__(self, directory: Path):
        self.directory = directory

    def run(self, state: AgentState) -> dict[str, object]:
        sources = search_documents(state.topic, self.directory)

        return {
            "sources": sources,
            "research_notes": [
                f"[{source.source_id}] {source.excerpt}"
                for source in sources
            ],
        }