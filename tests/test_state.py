import unittest

from pydantic import ValidationError

from app.state import EditorDecision, SearchQuery


class TestEditorDecision(unittest.TestCase):
    def test_approval_without_feedback_is_valid(self):
        decision = EditorDecision(approved=True)

        self.assertEqual(
            decision.model_dump(),
            {"approved": True, "feedback": ""},
        )

    def test_rejection_with_feedback_is_valid(self):
        decision = EditorDecision(
            approved=False,
            feedback="Add an example.",
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.feedback, "Add an example.")

    def test_approval_with_feedback_is_rejected(self):
        with self.assertRaises(ValidationError):
            EditorDecision(
                approved=True,
                feedback="Rewrite the article.",
            )

    def test_rejection_without_meaningful_feedback_is_rejected(self):
        for feedback in ("", "   "):
            with self.subTest(feedback=feedback):
                with self.assertRaises(ValidationError):
                    EditorDecision(
                        approved=False,
                        feedback=feedback,
                    )


class TestSearchQuery(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        query = SearchQuery(query="  meeting notes\n")

        self.assertEqual(query.query, "meeting notes")

    def test_rejects_empty_or_blank_query(self):
        for value in ("", "   ", "\t\n"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    SearchQuery(query=value)

    def test_requires_a_string_query(self):
        for data in ({}, {"query": None}, {"query": 123}, {"query": []}):
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    SearchQuery.model_validate(data)


if __name__ == "__main__":
    unittest.main()
