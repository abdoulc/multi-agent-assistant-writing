from .agents import EditorAgent, ResearchAgent, WriterAgent
from .llm import OllamaClient
from .state import AgentState, EditorDecision
from pathlib import Path

def run_workflow(
    initial_topic: str,
    max_rounds: int = 3,
    *,
    client: OllamaClient | None = None,
    documents_directory: Path | None = None,
) -> AgentState:
    if max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer.")
    
    llm_client = client if client is not None else OllamaClient()
    directory = (
        documents_directory
        if documents_directory is not None
        else Path(__file__).resolve().parent.parent / "documents"
    )
    research = ResearchAgent(directory)
    writer = WriterAgent(llm_client)
    editor = EditorAgent(llm_client)
    state = AgentState(topic=initial_topic)
    print("Initial state:", state.model_dump())

    state.apply(research.run(state))
    print("After research:", state.model_dump())

    if not state.sources:
        state.apply({
            "feedback": "No relevant sources found. Add documents or refine the topic.",
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
