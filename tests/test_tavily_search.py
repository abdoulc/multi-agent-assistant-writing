import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from app.tools.tavily_search import TavilySearch


class TestTavilySearch(unittest.TestCase):
    def setUp(self):
        self.client = TavilySearch("test-key", timeout=10)

    def make_response(self, payload):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    @patch("app.tools.tavily_search.urlopen")
    def test_returns_validated_results(self, mock_urlopen):
        mock_urlopen.return_value = self.make_response({
            "results": [{
                "title": "Meeting notes",
                "url": "https://example.com/notes",
                "content": "Keep unresolved questions separate.",
            }]
        })

        results = self.client.search("meeting notes", limit=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Meeting notes")
        self.assertEqual(str(results[0].url), "https://example.com/notes")
        self.assertEqual(
            results[0].snippet, "Keep unresolved questions separate."
        )
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["query"], "meeting notes")
        self.assertEqual(payload["max_results"], 2)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 10)

    @patch("app.tools.tavily_search.urlopen")
    def test_returns_empty_list_when_no_results(self, mock_urlopen):
        mock_urlopen.return_value = self.make_response({"results": []})

        self.assertEqual(self.client.search("unknown topic"), [])

    @patch("app.tools.tavily_search.urlopen")
    def test_propagates_http_failure(self, mock_urlopen):
        error = HTTPError(
            "https://api.tavily.com/search", 429, "Too Many Requests", {}, None
        )
        mock_urlopen.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "HTTP error: 429") as caught:
            self.client.search("meeting notes")

        self.assertIs(caught.exception.__cause__, error)
    
    @patch("app.tools.tavily_search.urlopen")
    def test_timeout_preserves_original_error(self, mock_urlopen):
        error = TimeoutError("Connection timed out")
        mock_urlopen.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "timed out") as caught:
            self.client.search("meeting notes")

        self.assertIs(caught.exception.__cause__, error)

    @patch("app.tools.tavily_search.urlopen")
    def test_rejects_invalid_json(self, mock_urlopen):
        mock_urlopen.return_value = io.BytesIO(b"not valid JSON")

        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            self.client.search("meeting notes")

    @patch("app.tools.tavily_search.urlopen")
    def test_rejects_invalid_response(self, mock_urlopen):
        payloads = [
            {},
            {"results": None},
            {"results": [{}]},
            {"results": [{
                "title": "Meeting notes",
                "url": "not-a-url",
                "content": "A summary.",
            }]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                mock_urlopen.return_value = self.make_response(payload)

                with self.assertRaisesRegex(RuntimeError, "invalid response"):
                    self.client.search("meeting notes")

    @patch("app.tools.tavily_search.urlopen")
    def test_invalid_input_does_not_call_provider(self, mock_urlopen):
        for query, limit in [("   ", 3), ("notes", 0), ("notes", 21)]:
            with self.subTest(query=query, limit=limit):
                with self.assertRaises(ValueError):
                    self.client.search(query, limit=limit)

        mock_urlopen.assert_not_called()

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            TavilySearch("   ")

        with self.assertRaises(ValueError):
            TavilySearch("test-key", timeout=0)