# Source-Grounded Multi-Agent Writing Workflow

A Python CLI that retrieves local or web sources, drafts an article with a local LLM, and runs a bounded editorial revision loop. Demonstrates tool integration, typed state, structured model outputs, validation, and failure recovery.

**Scope:** a code-orchestrated workflow with specialized agent roles, not a fully autonomous agent. Code selects the execution order and search mode; the LLM drafts, reviews, and reformulates unsuccessful local queries. No agent framework is required.

## Engineering Highlights

| Capability | Implementation |
| --- | --- |
| Tool integration | Local keyword search or Tavily web search, selected through the CLI |
| Dependency injection | `WebSearch` protocol; HTTP adapters separated from agent behavior |
| Shared evidence | Both search modes produce `ResearchSource` objects with IDs, locations, and excerpts |
| Validated state | Pydantic validates updates and editorial decision consistency |
| Structured outputs | JSON Schema supplied to Ollama, followed by parsing and business validation |
| Bounded recovery | One local query reformulation; one correction per invalid editor decision; three writing rounds by default |
| Testability | Unit and workflow tests simulate model/provider responses and failures without external services |

## Execution Flow

```text
CLI: topic + local/web mode
  -> ResearchAgent or WebResearchAgent
  -> Shared sources and research notes
  -> WriterAgent
  -> EditorAgent: citation checks, then LLM review
       -> Approved: return article
       -> Rejected: pass feedback to writer (within round limit)
```

Local search matches words against paragraphs in UTF-8 `.txt` files. Without matching sources, the LLM reformulates once; an equivalent query is not searched again. Web search retrieves up to three provider excerpts, not complete pages. Empty web results stop generation without a local fallback.

Citation checks reject missing or unknown `[S1]` references before model review. The editor assesses claims against excerpts. Contradictory decisions, such as approval with required changes, trigger one correction attempt. A second invalid decision or a network failure propagates as an error. Exhausted writing rounds return an unapproved state.

## Quick Start

Run from this project directory. Requires Python 3.11+ and either Docker Compose or an existing Ollama instance.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# First setup only: do not overwrite an existing .env.
Copy-Item .env.example .env
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:1b
python -m app.main "meeting assistant" --research local
```

Place local documents in `documents/`. If Ollama already runs on port 11434, use that instance instead of starting another, or configure another port and matching base URL.

For web mode, set a real `TAVILY_API_KEY` in the project `.env`, then run:

```powershell
python -m app.main "meeting notes" --research web
```

To run the application in Docker after starting Ollama and pulling the model:

```powershell
docker compose run --rm --build app python -m app.main "meeting notes" --research web
```

## Configuration

The CLI loads the project `.env` without overriding existing environment variables. Never commit real API keys.

| Variable | Default / behavior |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434`; native `/api/chat`, not `/v1` |
| `OLLAMA_MODEL` | `llama3.2:1b` |
| `OLLAMA_PORT` | `11434`; Docker host port only |
| `TAVILY_API_KEY` | Required only for web mode |

Compose sets the application container's Ollama URL to `http://ollama:11434`. The CLI uses a 300-second Ollama timeout. `max_rounds` and `documents_directory` are Python arguments to `run_workflow`, not environment variables or CLI options.

With default settings, inference stays local. Web mode sends queries to Tavily and consumes provider quota. Console output currently includes source excerpts and generated content: avoid sensitive inputs or shared logs.

## Verification

```powershell
python -m unittest discover -s tests -v
```

**66 tests passing as of 2026-10-08.** Coverage includes retrieval, provider errors, source conversion, CLI configuration, citation gates, decision repair, state preservation, and bounded execution. External calls are mocked: tests verify program behavior, not factual accuracy or live model reliability.

## Code Map

```text
app/main.py          CLI configuration and dependency assembly
app/workflow.py      Routing, state transitions, stopping conditions
app/state.py         Source, state, query, and decision models
app/agents/          Local/web research, query rewriting, writing, editing
app/tools/           Search protocol, Tavily adapter, local search, citations
app/llm/ollama.py     Native Ollama HTTP adapter
tests/               Unit and workflow regression tests
documents/           Local source corpus
```

## Limits and Next Milestones

- **Evidence quality:** citation existence does not prove support or truth; LLM review is fallible. No full-page extraction or semantic search yet.
- **Evaluation:** add a versioned dataset and measure source relevance, supported claims, task success, latency, and model comparisons against a simpler baseline.
- **Observability:** replace state dumps with structured events, run IDs, timings, and model-call counts, with content redaction.
- **Agent autonomy:** add bounded model-selected tool calls, permissions, and human approval before side effects. Multiple roles alone do not demonstrate autonomous planning.
- **Production readiness:** add persistent execution/resume, CI, deployment checks, and operational budgets. Prompt-injection defenses rely on instructions and validation, not an isolation boundary.

This repository demonstrates a tested foundation for agentic workflows, not production readiness or guaranteed article quality.
