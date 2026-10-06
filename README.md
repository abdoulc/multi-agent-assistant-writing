# Multi-Agent Writing Assistant

**Autonomous system for generating and improving articles through a multi-agent revision loop.**

## 🎯 What This System Does

Agents automatically write high-quality articles:

1. **ResearchAgent** → Searches for sources in local documents
2. **WriterAgent** → Generates an article based on found sources
3. **EditorAgent** → Evaluates the article (citations, clarity, relevance)
   - ✅ Approved? → Article finalized
   - ❌ Rejected? → Proposes corrections → WriterAgent rewrites → Loop (max 3 rounds)

**Result:** High-quality article produced automatically with iterative revisions.

## Structure

```text
.env.example                # Configuration template
.gitignore
.dockerignore
docker-compose.yml          # Services and model storage
Dockerfile
requirements.txt
ROADMAP.md
SPRINTS.md
app/
    main.py                 # CLI entry point
    workflow.py             # Research, revision loop, stopping conditions
    state.py                # AgentState with Pydantic validation
    agents/
        research.py         # Local document keyword search
        writer.py           # LLM-based article generation
        editor.py           # Citation checks and LLM review
    llm/
        ollama.py           # HTTP client for Ollama
    tools/
        local_search.py     # Keyword matching against .txt documents
        citations.py        # Citation validation
documents/                  # Local source documents
tests/                      # Unit tests with mocked LLM
```

## Architecture Pattern

- **Agents:** Pure functions (state → decision + updates)
- **State Management:** Centralized Pydantic models with validation
- **Workflow:** Orchestrates agents in a revision loop with stopping conditions
- **Backend:** Ollama (llama3.2:1b local model)

## Local Setup

Python 3.11+ required. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add documents to `documents/` directory:

```powershell
mkdir documents
echo "Your document content" > documents/sample.txt
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| OLLAMA_BASE_URL | http://localhost:11434/v1 | Ollama API endpoint |
| OLLAMA_MODEL | llama3.2:1b | Model to use |
| DOCUMENTS_DIRECTORY | ./documents | Documents directory |
| MAX_ROUNDS | 3 | Max revision rounds |

## Run Locally

Start Ollama with the model:

```powershell
# Ollama must be running with llama3.2:1b installed
python -m app.main --topic "Quantum Computing Basics"
python -m unittest discover -s tests -v
```

## Docker

```powershell
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:1b
docker compose run --rm --build app python -m app.main --topic "meeting assistant"
```
