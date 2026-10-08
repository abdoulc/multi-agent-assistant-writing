from app.state import AgentState, ResearchSource
from app.tools.web_search import WebSearch


class WebResearchAgent:
    def __init__(self, search_tool: WebSearch):
        self.search_tool = search_tool

    def run(
        self,
        state: AgentState,
        *,
        query: str | None = None,
    ) -> dict[str, object]:
        search_query = state.topic if query is None else query
        results = self.search_tool.search(search_query, limit=3)

        sources = [
            ResearchSource(
                source_id=f"S{index}",
                title=result.title,
                location=str(result.url),
                excerpt=result.snippet,
            )
            for index, result in enumerate(results, start=1)
        ]

        return {
            "sources": sources,
            "research_notes": [
                f"[{source.source_id}] {source.excerpt}"
                for source in sources
            ],
        }