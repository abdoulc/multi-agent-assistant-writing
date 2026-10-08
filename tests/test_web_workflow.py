import json
import unittest
from unittest.mock import Mock, patch

from app.llm import OllamaClient
from app.tools.web_search import WebSearch, WebSearchResult
from app.workflow import run_workflow


class TestWebWorkflow(unittest.TestCase):
    def setUp(self):
        self.client = Mock(spec=OllamaClient)
        self.tool = Mock(spec=WebSearch)
        local_agent = patch("app.workflow.ResearchAgent")
        self.local_agent = local_agent.start()
        self.addCleanup(local_agent.stop)
        rewriter = patch("app.workflow.QueryRewriter")
        self.rewriter = rewriter.start()
        self.addCleanup(rewriter.stop)

    def tearDown(self):
        self.local_agent.assert_not_called()
        self.rewriter.assert_not_called()

    def test_passes_web_sources_to_writer_and_editor(self):
        self.tool.search.return_value = [WebSearchResult(
            title="Meeting notes",
            url="https://example.com/notes",
            snippet="Keep unresolved questions separate.",
        )]
        self.client.generate.side_effect = [
            "Keep unresolved questions separate [S1].",
            json.dumps({"approved": True, "feedback": ""}),
        ]

        state = run_workflow(
            "meeting notes", client=self.client,
            research_mode="web", search_tool=self.tool,
        )

        self.tool.search.assert_called_once_with("meeting notes", limit=3)
        self.assertTrue(state.approved)
        self.assertEqual(state.rounds, 1)
        self.assertEqual(state.topic, "meeting notes")
        self.assertEqual(state.sources[0].location, "https://example.com/notes")
        self.assertEqual(state.research_notes, [
            "[S1] Keep unresolved questions separate."
        ])
        self.assertEqual(self.client.generate.call_count, 2)
        for entry in self.client.generate.call_args_list:
            self.assertEqual(entry.args[1]["sources"], [
                source.model_dump() for source in state.sources
            ])

    def test_empty_web_search_stops_without_model_calls(self):
        self.tool.search.return_value = []

        state = run_workflow(
            "meeting notes", client=self.client,
            research_mode="web", search_tool=self.tool,
        )

        self.tool.search.assert_called_once_with("meeting notes", limit=3)
        self.client.generate.assert_not_called()
        self.assertEqual(state.sources, [])
        self.assertEqual(state.research_notes, [])
        self.assertEqual(state.article, "")
        self.assertEqual(state.rounds, 0)
        self.assertFalse(state.approved)
        self.assertEqual(
            state.feedback,
            "No web sources found. Refine the topic and try again.",
        )

    def test_provider_failure_is_propagated_without_model_calls(self):
        error = RuntimeError("Web search unavailable.")
        self.tool.search.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            run_workflow(
                "meeting notes", client=self.client,
                research_mode="web", search_tool=self.tool,
            )

        self.assertIs(caught.exception, error)
        self.tool.search.assert_called_once_with("meeting notes", limit=3)
        self.client.generate.assert_not_called()

    def test_invalid_configuration_stops_before_search(self):
        cases = [
            ("unknown", None, "Research mode must be 'local' or 'web'."),
            ("web", None, "Web research requires a search tool."),
            ("local", self.tool, "A search tool requires web research mode."),
        ]
        for mode, tool, message in cases:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    run_workflow(
                        "meeting notes", client=self.client,
                        research_mode=mode, search_tool=tool,
                    )
                self.assertEqual(str(caught.exception), message)

        self.tool.search.assert_not_called()
        self.client.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
