import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from app.agents.query_rewriter import QueryRewriter, QUERY_REWRITER_SYSTEM_PROMPT
from app.llm import OllamaClient
from app.state import SearchQuery


class TestQueryRewriter(unittest.TestCase):
    def test_sends_context_and_returns_validated_query(self):
        client = Mock(spec=OllamaClient)
        client.generate.return_value = json.dumps({"query": "  meeting notes  "})

        result = QueryRewriter(client).run(
            topic="Meeting summaries", previous_query="meeting minutes"
        )

        self.assertIsInstance(result, SearchQuery)
        self.assertEqual(result.query, "meeting notes")
        client.generate.assert_called_once_with(
            QUERY_REWRITER_SYSTEM_PROMPT,
            {"topic": "Meeting summaries", "previous_query": "meeting minutes"},
            output_schema=SearchQuery.model_json_schema(),
        )

    def test_rejects_invalid_model_responses(self):
        responses = [
            "Not JSON.",
            "{}",
            '{"query": ""}',
            '{"query": "   "}',
            '{"query": null}',
            '{"query": 123}',
        ]
        for response in responses:
            with self.subTest(response=response):
                client = Mock(spec=OllamaClient)
                client.generate.return_value = response

                with self.assertRaises(ValidationError):
                    QueryRewriter(client).run("Meeting summaries", "meeting minutes")

                client.generate.assert_called_once()

    def test_propagates_client_failure(self):
        client = Mock(spec=OllamaClient)
        error = RuntimeError("The Ollama request timed out.")
        client.generate.side_effect = error

        with self.assertRaises(RuntimeError) as raised:
            QueryRewriter(client).run("Meeting summaries", "meeting minutes")

        self.assertIs(raised.exception, error)
        client.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
