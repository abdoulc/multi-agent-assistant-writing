import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .llm import OllamaClient
from .workflow import run_workflow


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Demonstration topic"
    client = OllamaClient(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "llama3.2:1b"),
        timeout=300,
    )
    run_workflow(topic, client=client)
