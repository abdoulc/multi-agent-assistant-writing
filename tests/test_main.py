import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMain(unittest.TestCase):
    def test_loads_project_environment_and_passes_configured_client(self):
        environment = {
            "OLLAMA_BASE_URL": "http://localhost:11435",
            "OLLAMA_MODEL": "test-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("sys.argv", ["app.main", "meeting", "assistant"]),
            patch("dotenv.load_dotenv") as load_environment,
            patch("app.llm.OllamaClient") as client_type,
            patch("app.tools.tavily_search.TavilySearch") as search_type,
            patch("app.workflow.run_workflow") as workflow,
        ):
            runpy.run_module("app.main", run_name="__main__")

        load_environment.assert_called_once_with(
            Path(__file__).resolve().parents[1] / ".env", override=False
        )
        client_type.assert_called_once_with(
            base_url="http://localhost:11435", model="test-model", timeout=300
        )
        workflow.assert_called_once_with(
            "meeting assistant", client=client_type.return_value,
            research_mode="local", search_tool=None,
        )
        search_type.assert_not_called()

    def test_web_mode_injects_search_provider(self):
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": " test-key "}, clear=True),
            patch("sys.argv", ["app.main", "meeting notes", "--research", "web"]),
            patch("dotenv.load_dotenv"),
            patch("app.llm.OllamaClient") as client_type,
            patch("app.tools.tavily_search.TavilySearch") as search_type,
            patch("app.workflow.run_workflow") as workflow,
        ):
            runpy.run_module("app.main", run_name="__main__")

        search_type.assert_called_once_with(api_key="test-key")
        workflow.assert_called_once_with(
            "meeting notes", client=client_type.return_value,
            research_mode="web", search_tool=search_type.return_value,
        )

    def test_web_mode_without_valid_key_stops_before_execution(self):
        for environment in ({}, {"TAVILY_API_KEY": "   "}):
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment, clear=True),
                patch("sys.argv", ["app.main", "notes", "--research", "web"]),
                patch("dotenv.load_dotenv"),
                patch("sys.stderr"),
                patch("app.llm.OllamaClient") as client_type,
                patch("app.tools.tavily_search.TavilySearch") as search_type,
                patch("app.workflow.run_workflow") as workflow,
            ):
                with self.assertRaises(SystemExit) as caught:
                    runpy.run_module("app.main", run_name="__main__")

                self.assertEqual(caught.exception.code, 2)
                client_type.assert_not_called()
                search_type.assert_not_called()
                workflow.assert_not_called()

    def test_default_topic_uses_local_mode(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.argv", ["app.main"]),
            patch("dotenv.load_dotenv"),
            patch("app.llm.OllamaClient") as client_type,
            patch("app.tools.tavily_search.TavilySearch") as search_type,
            patch("app.workflow.run_workflow") as workflow,
        ):
            runpy.run_module("app.main", run_name="__main__")

        workflow.assert_called_once_with(
            "Demonstration topic", client=client_type.return_value,
            research_mode="local", search_tool=None,
        )
        search_type.assert_not_called()

    def test_invalid_research_mode_stops_before_execution(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.argv", ["app.main", "notes", "--research", "unknown"]),
            patch("dotenv.load_dotenv"),
            patch("sys.stderr"),
            patch("app.llm.OllamaClient") as client_type,
            patch("app.tools.tavily_search.TavilySearch") as search_type,
            patch("app.workflow.run_workflow") as workflow,
        ):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_module("app.main", run_name="__main__")

        self.assertEqual(caught.exception.code, 2)
        client_type.assert_not_called()
        search_type.assert_not_called()
        workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
