import unittest

from app.state import ResearchSource
from app.tools.citations import check_citations


class TestCitations(unittest.TestCase):
    def setUp(self):
        self.sources = [ResearchSource(
            source_id="S1", title="AI notes", location="documents/ai.txt",
            excerpt="An AI assistant can summarize meeting notes.",
        )]

    def test_accepts_known_citations(self):
        self.assertEqual(check_citations("Notes [S1]. More notes [S1].", self.sources), [])

    def test_rejects_missing_sources(self):
        problems = check_citations("Notes [S1].", [])
        self.assertEqual(len(problems), 1)
        self.assertIn("No sources", problems[0])

    def test_rejects_missing_citations(self):
        problems = check_citations("Notes without references.", self.sources)
        self.assertEqual(len(problems), 1)
        self.assertIn("Add source citations", problems[0])

    def test_rejects_unknown_citations_even_with_valid_reference(self):
        problems = check_citations("Notes [S1] [S99] [S42] [S99].", self.sources)
        self.assertEqual(len(problems), 1)
        self.assertIn("S42, S99", problems[0])
        self.assertEqual(problems[0].count("S99"), 1)


if __name__ == "__main__":
    unittest.main()
