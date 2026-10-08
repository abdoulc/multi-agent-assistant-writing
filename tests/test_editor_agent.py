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
        client.generate.assert_called_once()

    def test_repairs_invalid_decision_without_mutating_state(self):
        invalid_responses = [
            "Invalid JSON.",
            '{"approved": true, "feedback": "Add details."}',
        ]
        corrections = [
            {"approved": False, "feedback": "Add details."},
            {"approved": True, "feedback": ""},
        ]
        for invalid in invalid_responses:
            for corrected in corrections:
                with self.subTest(invalid=invalid, corrected=corrected):
                    client = Mock(spec=OllamaClient)
                    client.generate.side_effect = [invalid, json.dumps(corrected)]
                    state = AgentState(
                        topic="AI", article="Draft [S1].",
                        research_notes=["[S1] An AI assistant summarizes notes."],
                        feedback="Clarify the example.", sources=[self.source],
                    )
                    initial_state = state.model_dump()

                    decision = EditorAgent(client).run(state)

                    self.assertEqual(decision, corrected)
                    self.assertEqual(client.generate.call_count, 2)
                    first_call, repair_call = client.generate.call_args_list
                    self.assertEqual(first_call.args[0], EDITOR_SYSTEM_PROMPT)
                    self.assertNotEqual(repair_call.args[0], first_call.args[0])
                    repair_context = repair_call.args[1]
                    self.assertNotIn("invalid_decision", first_call.args[1])
                    self.assertEqual(repair_context["invalid_decision"], invalid)
                    self.assertTrue(repair_context["validation_feedback"].strip())
                    for key, value in first_call.args[1].items():
                        self.assertEqual(repair_context[key], value)
                    for entry in (first_call, repair_call):
                        self.assertEqual(
                            entry.kwargs["output_schema"],
                            EditorDecision.model_json_schema(),
                        )
                    self.assertEqual(state.model_dump(), initial_state)

    def test_client_failure_is_not_retried(self):
        for during_repair in (False, True):
            with self.subTest(during_repair=during_repair):
                client = Mock(spec=OllamaClient)
                error = RuntimeError("The Ollama request timed out.")
                client.generate.side_effect = (
                    ["Invalid JSON.", error] if during_repair else [error]
                )
                state = AgentState(
                    topic="AI", article="Draft [S1].", sources=[self.source]
                )
                initial_state = state.model_dump()

                with self.assertRaises(RuntimeError) as caught:
                    EditorAgent(client).run(state)

                self.assertIs(caught.exception, error)
                self.assertEqual(client.generate.call_count, 2 if during_repair else 1)
                self.assertEqual(state.model_dump(), initial_state)

    def test_rejects_invalid_model_decisions_after_one_repair(self):
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

                self.assertEqual(client.generate.call_count, 2)
                self.assertEqual(state.model_dump(), initial_state)


if __name__ == "__main__":
    unittest.main()
