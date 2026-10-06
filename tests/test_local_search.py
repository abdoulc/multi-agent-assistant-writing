import tempfile
import unittest
from pathlib import Path

from app.tools.local_search import search_documents


class TestLocalSearch(unittest.TestCase):
    def test_returns_best_matching_excerpt(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            document = directory / "assistant.txt"
            document.write_text(
                "An assistant can organize files.\n\n"
                "A meeting assistant prepares a summary.",
                encoding="utf-8",
            )

            sources = search_documents("meeting assistant", directory, limit=1)

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].source_id, "S1")
            self.assertEqual(sources[0].title, "assistant")
            self.assertEqual(
                sources[0].excerpt,
                "A meeting assistant prepares a summary.",
            )
            self.assertEqual(sources[0].location, str(document.resolve()))

    def test_returns_no_sources_when_nothing_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "notes.txt").write_text(
                "Gardening requires regular watering.",
                encoding="utf-8",
            )

            sources = search_documents("robotics", directory)

            self.assertEqual(sources, [])

    def test_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing"

            with self.assertRaises(FileNotFoundError):
                search_documents("assistant", missing)

    def test_rejects_non_positive_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            for limit in (0, -1):
                with self.subTest(limit=limit):
                    with self.assertRaises(ValueError):
                        search_documents("assistant", Path(folder), limit=limit)


if __name__ == "__main__":
    unittest.main()
