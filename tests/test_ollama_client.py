import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from app.llm import OllamaClient
from app.state import EditorDecision


class TestOllamaClient(unittest.TestCase):
    def test_sends_output_schema_when_provided(self):
        schema = EditorDecision.model_json_schema()
        decision_json = json.dumps({
            "approved": False,
            "feedback": "Add a concrete example.",
        })
        response = io.BytesIO(
            json.dumps({"message": {"content": decision_json}}).encode("utf-8")
        )

        with patch("app.llm.ollama.urlopen", return_value=response) as request:
            result = OllamaClient().generate(
                "Review the article.",
                {"article": "A short draft."},
                output_schema=schema,
            )

        request.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["format"], schema)
        self.assertEqual(result, decision_json)

    def test_uses_configured_connection_settings(self):
        client = OllamaClient(
            base_url="http://ollama:11434/",
            model="test-model",
            timeout=12,
        )
        response = io.BytesIO(b'{"message": {"content": "Article."}}')

        with patch("app.llm.ollama.urlopen", return_value=response) as request:
            self.assertEqual(client.generate("Write an article.", {}), "Article.")

        http_request = request.call_args.args[0]
        payload = json.loads(http_request.data)
        self.assertEqual(http_request.full_url, "http://ollama:11434/api/chat")
        self.assertEqual(request.call_args.kwargs, {"timeout": 12})
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["content"], "Write an article.")

    def test_reports_connection_failure(self):
        error = URLError("Connection refused")

        with patch("app.llm.ollama.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(
                RuntimeError,
                "Could not connect to Ollama: Connection refused",
            ) as raised:
                OllamaClient().generate("Write an article.", {"topic": "AI"})

        request.assert_called_once()
        self.assertIs(raised.exception.__cause__, error)

    def test_reports_http_error(self):
        error = HTTPError(
            "http://localhost:11434/api/chat",
            503,
            "Service Unavailable",
            hdrs=None,
            fp=None,
        )

        with patch("app.llm.ollama.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Ollama returned HTTP status 503\.",
            ) as raised:
                OllamaClient().generate("Write an article.", {"topic": "AI"})

        request.assert_called_once()
        self.assertIs(raised.exception.__cause__, error)

    def test_reports_timeout(self):
        error = TimeoutError("The connection timed out")

        with patch("app.llm.ollama.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(
                RuntimeError,
                r"The Ollama request timed out\.",
            ) as raised:
                OllamaClient().generate("Write an article.", {"topic": "AI"})

        request.assert_called_once()
        self.assertIs(raised.exception.__cause__, error)

    def test_extracts_article_and_sends_context(self):
        context = {
            "topic": "AI",
            "research_notes": ["A sample note."],
            "previous_article": "",
            "editor_feedback": "",
        }
        response_body = {
            "message": {"content": "  Generated article.  "}
        }
        response = io.BytesIO(json.dumps(response_body).encode("utf-8"))

        with patch("app.llm.ollama.urlopen", return_value=response) as request:
            article = OllamaClient().generate("Write an article.", context)

        self.assertEqual(article, "Generated article.")
        request.assert_called_once()

        http_request = request.call_args.args[0]
        payload = json.loads(http_request.data)

        self.assertEqual(http_request.get_method(), "POST")
        self.assertFalse(payload["stream"])
        self.assertNotIn("format", payload)
        self.assertEqual(
            json.loads(payload["messages"][1]["content"]),
            context,
        )

    def test_rejects_empty_or_invalid_article(self):
        for content in ("", "   ", None, 123):
            with self.subTest(content=content):
                response_body = {"message": {"content": content}}
                response = io.BytesIO(
                    json.dumps(response_body).encode("utf-8")
                )

                with patch("app.llm.ollama.urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        OllamaClient().generate("Write an article.", {"topic": "AI"})


if __name__ == "__main__":
    unittest.main()
