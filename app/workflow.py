from .agents import EditorAgent, ResearchAgent, WriterAgent
from .llm import OllamaClient
from .state import AgentState, EditorDecision
from pathlib import Path
from .agents.query_rewriter import QueryRewriter
from typing import Literal
from .agents.web_research import WebResearchAgent
from .tools.web_search import WebSearch

def run_workflow(
    initial_topic: str,
    max_rounds: int = 3,
    *,
    client: OllamaClient | None = None,
    documents_directory: Path | None = None,
    research_mode: Literal["local", "web"] = "local",
    search_tool: WebSearch | None = None,
) -> AgentState:
    if max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer.")
    if research_mode not in ("local", "web"):
        raise ValueError("Research mode must be 'local' or 'web'.")
    if research_mode == "web" and search_tool is None:
        raise ValueError("Web research requires a search tool.")
    if research_mode == "local" and search_tool is not None:
        raise ValueError("A search tool requires web research mode.")
    llm_client = client if client is not None else OllamaClient()
    directory = (
        documents_directory
        if documents_directory is not None
        else Path(__file__).resolve().parent.parent / "documents"
    )
    research: ResearchAgent | WebResearchAgent
    if research_mode == "web":
        assert search_tool is not None
        research = WebResearchAgent(search_tool)
    else:
        research = ResearchAgent(directory)

    writer = WriterAgent(llm_client)
    editor = EditorAgent(llm_client)
    state = AgentState(topic=initial_topic)
    print("Initial state:", state.model_dump())

    state.apply(research.run(state))
    print("After research:", state.model_dump())

    if research_mode == "web" and not state.sources:
        state.apply({
            "feedback": (
                "No web sources found. Refine the topic and try again."
            ),
        })
        print("Stopped:", state.feedback)
        return state

    if not state.sources:
        rewriter = QueryRewriter(llm_client)
        rewritten = rewriter.run(
            topic=state.topic,
            previous_query=state.topic,
        )

        original_query = " ".join(state.topic.casefold().split())
        alternative_query = " ".join(rewritten.query.casefold().split())

        if alternative_query != original_query:
            print("Retrying research with query:", rewritten.query)
            state.apply(research.run(state, query=rewritten.query))
            print("After second research:", state.model_dump())

        if not state.sources:
            state.apply({
                "feedback": (
                    "No relevant sources found after query reformulation. "
                    "Add documents or refine the topic."
                ),
            })
            print("Stopped:", state.feedback)
            return state

    for round_number in range(1, max_rounds + 1):
        state.apply({"rounds": round_number})
        print(f"--- Round {round_number} ---")

        state.apply(writer.run(state))
        print("After writer:", {"article_len": len(state.article)})

        decision = EditorDecision.model_validate(editor.run(state))
        state.apply(decision.model_dump())
        if state.approved:
            print("Approved. Final article:\n", state.article)
            return state
        print("Feedback:", state.feedback)

    print("Max rounds reached. Final state:\n", state.model_dump())
    return state
