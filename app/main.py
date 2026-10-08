import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .llm import OllamaClient
from .tools.tavily_search import TavilySearch
from .workflow import run_workflow


if __name__ == "__main__":
    load_dotenv(
        Path(__file__).resolve().parent.parent / ".env",
        override=False,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*", default=[])
    parser.add_argument(
        "--research",
        choices=("local", "web"),
        default="local",
    )
    args = parser.parse_args()
    topic = " ".join(args.topic) or "Demonstration topic"

    search_tool = None
    if args.research == "web":
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            parser.error("Web research requires TAVILY_API_KEY.")
        search_tool = TavilySearch(api_key=api_key)

    client = OllamaClient(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "llama3.2:1b"),
        timeout=300,
    )
    run_workflow(
        topic,
        client=client,
        research_mode=args.research,
        search_tool=search_tool,
    )