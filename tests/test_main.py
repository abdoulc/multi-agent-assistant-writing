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
            "meeting assistant", client=client_type.return_value
        )


if __name__ == "__main__":
    unittest.main()
