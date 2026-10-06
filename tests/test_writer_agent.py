import unittest
from unittest.mock import patch

from app.agents import WriterAgent
from app.agents.writer import WRITER_SYSTEM_PROMPT
from app.llm import OllamaClient
from app.state import AgentState, ResearchSource


class TestWriterAgent(unittest.TestCase):
    def test_writer_passes_context_to_model(self):
        source = ResearchSource(
            source_id="S1", title="AI notes", location="documents/ai.txt",
            excerpt="An AI assistant can summarize meeting notes.",
        )
        state = AgentState(
            topic="AI",
            research_notes=["A sample note."],
            article="Initial draft.",
            feedback="Add a fictional example.",
            sources=[source],
        )
        initial_state = state.model_dump()

        with patch(
            "app.llm.OllamaClient.generate",
            return_value="Revised article.",
        ) as generate:
            update = WriterAgent(OllamaClient()).run(state)

        self.assertEqual(update, {"article": "Revised article."})
        generate.assert_called_once_with(WRITER_SYSTEM_PROMPT, {
            "topic": "AI",
            "research_notes": ["A sample note."],
            "previous_article": "Initial draft.",
            "editor_feedback": "Add a fictional example.",
            "sources": [source.model_dump()],
        })
        self.assertEqual(state.model_dump(), initial_state)


if __name__ == "__main__":
    unittest.main()
