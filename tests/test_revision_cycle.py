import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.agents import EditorAgent, ResearchAgent, WriterAgent
from app.agents.editor import EDITOR_SYSTEM_PROMPT
from app.agents.writer import WRITER_SYSTEM_PROMPT
from app.llm import OllamaClient
from app.state import AgentState


class TestRevisionCycle(unittest.TestCase):
    def test_feedback_leads_to_approved_revision(self):
        client = Mock(spec=OllamaClient)
        feedback = "Add a concrete example."
        revised_article = "An AI assistant summarizes notes [S1]. Fictional example: a meeting."
        client.generate.side_effect = [
            "Short draft [S1].",
            json.dumps({"approved": False, "feedback": feedback}),
            revised_article,
            json.dumps({"approved": True, "feedback": ""}),
        ]
        writer = WriterAgent(client)
        editor = EditorAgent(client)
        state = AgentState(topic="AI")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        directory = Path(folder.name)
        excerpt = "An AI assistant can summarize meeting notes."
        (directory / "ai.txt").write_text(excerpt, encoding="utf-8")
        state.apply(ResearchAgent(directory).run(state))
        self.assertEqual(len(state.sources), 1)
        self.assertEqual(state.sources[0].excerpt, excerpt)
        self.assertEqual(state.research_notes, [f"[S1] {excerpt}"])
        state.apply(writer.run(state))
        first_article = state.article

        decision = editor.run(state)
        self.assertFalse(decision["approved"])
        self.assertEqual(decision["feedback"], feedback)
        state.apply(decision)
        state.apply(writer.run(state))
        state.apply(editor.run(state))

        self.assertEqual(state.article, revised_article)
        self.assertNotIn(feedback, state.article)
        self.assertTrue(state.approved)
        self.assertEqual(state.feedback, "")
        self.assertEqual(client.generate.call_count, 4)
        calls = client.generate.call_args_list
        for entry in calls:
            self.assertEqual(
                entry.args[1]["sources"],
                [source.model_dump() for source in state.sources],
            )
        self.assertEqual(
            [entry.args[0] for entry in calls],
            [WRITER_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT,
             WRITER_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT],
        )
        revision_context = calls[2].args[1]
        self.assertEqual(revision_context["previous_article"], first_article)
        self.assertEqual(revision_context["editor_feedback"], feedback)
        review_context = calls[3].args[1]
        self.assertEqual(review_context["article"], revised_article)
        self.assertEqual(review_context["previous_feedback"], feedback)


if __name__ == "__main__":
    unittest.main()
