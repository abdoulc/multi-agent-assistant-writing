import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pydantic import ValidationError

from app.llm import OllamaClient
from app.state import AgentState
from app.workflow import run_workflow


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.documents_directory = Path(folder.name)
        (self.documents_directory / "ai.txt").write_text(
            "An AI assistant can summarize meeting notes.", encoding="utf-8"
        )

    def test_uses_injected_client_for_revision(self):
        client = Mock(spec=OllamaClient)
        client.generate.side_effect = [
            "Short draft [S1].",
            json.dumps({"approved": False, "feedback": "Add a concrete example."}),
            "An AI assistant summarizes notes [S1]. Fictional example: a meeting.",
            json.dumps({"approved": True, "feedback": ""}),
        ]

        state = run_workflow(
            "AI", client=client, documents_directory=self.documents_directory
        )

        self.assertTrue(state.approved)
        self.assertEqual(state.rounds, 2)
        self.assertEqual(state.feedback, "")
        self.assertEqual(client.generate.call_count, 4)
        context = client.generate.call_args_list[2].args[1]
        self.assertEqual(context["previous_article"], "Short draft [S1].")
        self.assertEqual(context["editor_feedback"], "Add a concrete example.")
        self.assertEqual(len(state.sources), 1)
        for entry in client.generate.call_args_list:
            self.assertEqual(
                entry.args[1]["sources"],
                [source.model_dump() for source in state.sources],
            )

    def test_stops_without_model_calls_when_directory_is_empty(self):
        client = Mock(spec=OllamaClient)
        with tempfile.TemporaryDirectory() as folder:
            state = run_workflow(
                "AI", client=client, documents_directory=Path(folder)
            )

        client.generate.assert_not_called()
        self.assertEqual(state.sources, [])
        self.assertEqual(state.research_notes, [])
        self.assertEqual(state.article, "")
        self.assertFalse(state.approved)
        self.assertEqual(state.rounds, 0)
        self.assertEqual(
            state.feedback,
            "No relevant sources found. Add documents or refine the topic.",
        )

    def test_stops_without_model_calls_when_documents_do_not_match(self):
        client = Mock(spec=OllamaClient)
        state = run_workflow(
            "Gardening", client=client,
            documents_directory=self.documents_directory,
        )

        client.generate.assert_not_called()
        self.assertEqual(state.sources, [])
        self.assertEqual(state.research_notes, [])
        self.assertEqual(state.article, "")
        self.assertFalse(state.approved)
        self.assertEqual(state.rounds, 0)
        self.assertEqual(
            state.feedback,
            "No relevant sources found. Add documents or refine the topic.",
        )

    def test_stops_after_max_rounds(self):
        client = Mock(spec=OllamaClient)
        rejection = json.dumps({"approved": False, "feedback": "Add more details."})
        client.generate.side_effect = ["A draft article [S1].", rejection] * 3

        state = run_workflow(
            "AI", max_rounds=3, client=client,
            documents_directory=self.documents_directory,
        )

        self.assertIsInstance(state, AgentState)
        self.assertEqual(client.generate.call_count, 6)
        self.assertFalse(state.approved)
        self.assertEqual(state.rounds, 3)
        self.assertEqual(state.feedback, "Add more details.")

    def test_stops_when_approved(self):
        client = Mock(spec=OllamaClient)
        client.generate.side_effect = [
            "An approved article [S1].",
            json.dumps({"approved": True, "feedback": ""}),
        ]

        state = run_workflow(
            "AI", max_rounds=3, client=client,
            documents_directory=self.documents_directory,
        )

        self.assertEqual(state.article, "An approved article [S1].")
        self.assertEqual(client.generate.call_count, 2)
        self.assertTrue(state.approved)
        self.assertEqual(state.rounds, 1)

    def test_invalid_evaluation_stops_workflow(self):
        client = Mock(spec=OllamaClient)
        client.generate.side_effect = ["A draft article [S1].", "Invalid JSON."]


        with self.assertRaises(ValidationError):
            run_workflow(
                "AI", max_rounds=3, client=client,
                documents_directory=self.documents_directory,
            )

        self.assertEqual(client.generate.call_count, 2)

    def test_missing_citation_is_revised_before_model_review(self):
        client = Mock(spec=OllamaClient)
        client.generate.side_effect = [
            "An uncited draft.",
            "An AI assistant can summarize meeting notes [S1].",
            json.dumps({"approved": True, "feedback": ""}),
        ]

        state = run_workflow(
            "AI", client=client, documents_directory=self.documents_directory
        )

        self.assertTrue(state.approved)
        self.assertEqual(state.rounds, 2)
        self.assertEqual(client.generate.call_count, 3)
        calls = client.generate.call_args_list
        self.assertNotIn("output_schema", calls[0].kwargs)
        self.assertNotIn("output_schema", calls[1].kwargs)
        self.assertIn("output_schema", calls[2].kwargs)
        self.assertIn("Add source citations", calls[1].args[1]["editor_feedback"])


if __name__ == "__main__":
    unittest.main()
