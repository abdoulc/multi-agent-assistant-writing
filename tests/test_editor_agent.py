import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from app.agents import EditorAgent
from app.agents.editor import EDITOR_SYSTEM_PROMPT
from app.llm import OllamaClient
from app.state import AgentState, EditorDecision, ResearchSource


class TestEditorAgent(unittest.TestCase):
    def setUp(self):
        self.source = ResearchSource(
            source_id="S1", title="AI notes", location="documents/ai.txt",
            excerpt="An AI assistant can summarize meeting notes.",
        )

    def test_rejects_citation_problems_without_calling_model(self):
        cases = [
            ("Draft [S1].", [], "No sources"),
            ("Draft without citations.", [self.source], "Add source citations"),
            ("Draft [S1] [S99].", [self.source], "S99"),
        ]
        for article, sources, expected_feedback in cases:
            with self.subTest(article=article, sources=sources):
                client = Mock(spec=OllamaClient)
                state = AgentState(topic="AI", article=article, sources=sources)
                initial_state = state.model_dump()

                decision = EditorAgent(client).run(state)

                self.assertFalse(decision["approved"])
                self.assertIn(expected_feedback, decision["feedback"])
                client.generate.assert_not_called()
                self.assertEqual(state.model_dump(), initial_state)

    def test_sends_context_and_schema_without_modifying_state(self):
        source = ResearchSource(
            source_id="S1", title="AI notes", location="documents/ai.txt",
            excerpt="An AI assistant can summarize meeting notes.",
        )
        state = AgentState(
            topic="AI",
            research_notes=["A sample note."],
            article="Initial draft [S1].",
            feedback="Clarify the example.",
            sources=[source],
        )
        initial_state = state.model_dump()
        client = Mock(spec=OllamaClient)
        client.generate.return_value = json.dumps({
            "approved": False,
            "feedback": "Add a concrete example.",
        })

        decision = EditorAgent(client).run(state)

        self.assertEqual(decision, {
            "approved": False,
            "feedback": "Add a concrete example.",
        })
        client.generate.assert_called_once_with(
            EDITOR_SYSTEM_PROMPT,
            {
                "topic": "AI",
                "research_notes": ["A sample note."],
                "article": "Initial draft [S1].",
                "previous_feedback": "Clarify the example.",
                "sources": [source.model_dump()],
            },
            output_schema=EditorDecision.model_json_schema(),
        )
        self.assertEqual(state.model_dump(), initial_state)

    def test_returns_validated_approval(self):
        client = Mock(spec=OllamaClient)
        client.generate.return_value = '{"approved": true}'

        decision = EditorAgent(client).run(
            AgentState(topic="AI", article="Draft [S1].", sources=[self.source])
        )

        self.assertEqual(decision, {"approved": True, "feedback": ""})

    def test_rejects_invalid_model_decisions(self):
        responses = [
            "This is not JSON.",
            '{"feedback": "Add details."}',
            '{"approved": true, "feedback": "Rewrite the article."}',
            '{"approved": false, "feedback": "   "}',
            '{"approved": "undecided", "feedback": ""}',
        ]
        for response in responses:
            with self.subTest(response=response):
                client = Mock(spec=OllamaClient)
                client.generate.return_value = response
                state = AgentState(
                    topic="AI", article="Original draft [S1].", sources=[self.source]
                )
                initial_state = state.model_dump()

                with self.assertRaises(ValidationError):
                    EditorAgent(client).run(state)

                self.assertEqual(state.model_dump(), initial_state)


if __name__ == "__main__":
    unittest.main()
