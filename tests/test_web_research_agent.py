import unittest
from unittest.mock import Mock

from app.agents.web_research import WebResearchAgent
from app.state import AgentState
from app.tools.web_search import WebSearch, WebSearchResult


class TestWebResearchAgent(unittest.TestCase):
    def setUp(self):
        self.tool = Mock(spec=WebSearch)
        self.agent = WebResearchAgent(self.tool)
        self.state = AgentState(topic="meeting notes")

    def test_converts_results_without_mutating_state(self):
        self.tool.search.return_value = [
            WebSearchResult(
                title="Meeting notes",
                url="https://example.com/notes",
                snippet="Keep unresolved questions separate.",
            ),
            WebSearchResult(
                title="Meeting summary",
                url="https://example.com/summary",
                snippet="Ask for confirmation before sending.",
            ),
        ]
        original_state = self.state.model_dump()

        update = self.agent.run(self.state)

        self.tool.search.assert_called_once_with("meeting notes", limit=3)
        sources = update["sources"]
        self.assertEqual([s.source_id for s in sources], ["S1", "S2"])
        self.assertEqual(sources[0].title, "Meeting notes")
        self.assertEqual(sources[0].location, "https://example.com/notes")
        self.assertEqual(
            sources[0].excerpt, "Keep unresolved questions separate."
        )
        self.assertEqual(update["research_notes"], [
            "[S1] Keep unresolved questions separate.",
            "[S2] Ask for confirmation before sending.",
        ])
        self.assertEqual(self.state.model_dump(), original_state)
        self.state.apply(update)
        self.assertEqual(self.state.sources, sources)

    def test_uses_explicit_query_without_changing_topic(self):
        self.tool.search.return_value = []

        self.agent.run(self.state, query="meeting minutes")

        self.tool.search.assert_called_once_with("meeting minutes", limit=3)
        self.assertEqual(self.state.topic, "meeting notes")

    def test_returns_empty_update_when_no_results(self):
        self.tool.search.return_value = []

        update = self.agent.run(self.state)

        self.assertEqual(update, {"sources": [], "research_notes": []})

    def test_propagates_tool_failure(self):
        error = RuntimeError("Web search unavailable.")
        self.tool.search.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            self.agent.run(self.state)

        self.assertIs(caught.exception, error)