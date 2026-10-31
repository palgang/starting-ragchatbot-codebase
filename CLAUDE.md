# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system for answering questions about course materials. FastAPI backend + vanilla JS/HTML frontend, ChromaDB for vector storage, Anthropic Claude for generation. On Windows, use Git Bash to run the commands below.

## Commands

```bash
# Install dependencies (uv is the package manager; there is no pip/requirements.txt)
uv sync

# Run the app (from repo root) - auto-reloads on change
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any one-off python inside backend/ with the project's venv
uv run python <script.py>
```

- Web UI: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`
- Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).
- There is no test suite, linter, or formatter configured in this repo currently.

## Architecture

### Request flow

`frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`) → `AIGenerator.generate_response()` (`backend/ai_generator.py`) → Claude, with the `search_course_content` tool available → on tool use, `ToolManager.execute_tool()` → `CourseSearchTool.execute()` (`backend/search_tools.py`) → `VectorStore.search()` (`backend/vector_store.py`) → ChromaDB.

Claude decides for itself whether to call the search tool (course-specific questions) or answer directly (general knowledge) — this is steered entirely by the system prompt in `AIGenerator.SYSTEM_PROMPT`, not by code branching. The tool-use loop in `AIGenerator._handle_tool_execution` only supports **one** round of tool calls: the initial response, then one follow-up call with tools removed to force a final text answer.

### Core components (`backend/`)

- **`rag_system.py`** — `RAGSystem` is the orchestrator all other layers go through. It wires together the document processor, vector store, AI generator, session manager, and tool manager, and exposes `add_course_document`/`add_course_folder`, `query`, and `get_course_analytics`.
- **`ai_generator.py`** — Wraps the Anthropic SDK call. Holds the static system prompt and the tool-execution loop. `temperature=0`, `max_tokens=800`, model comes from `Config.ANTHROPIC_MODEL`.
- **`search_tools.py`** — `Tool`/`ToolManager` abstraction so new tools can be registered without touching `AIGenerator`. `CourseSearchTool` is the only tool currently registered; it also tracks `last_sources` (for citations returned to the frontend) which `ToolManager` reads/resets after each query.
- **`vector_store.py`** — Owns two ChromaDB collections:
  - `course_catalog`: one doc per course (id = course title), used only to semantically resolve a fuzzy/partial course name to its exact title. Metadata: `title`, `instructor`, `course_link`, `lesson_count`, `lessons_json` (serialized list of `{lesson_number, lesson_title, lesson_link}`).
  - `course_content`: the actual chunked text, filterable by `course_title`/`lesson_number`. Metadata per chunk: `course_title`, `lesson_number`, `chunk_index`.

  `VectorStore.search()` does the two-step resolve-then-filter-search; callers never query ChromaDB directly.
- **`document_processor.py`** — Parses the expected course document format (see below) into a `Course` + list of `CourseChunk`, and does sentence-aware chunking with overlap (`CHUNK_SIZE`/`CHUNK_OVERLAP` from config). Chunks are prefixed with lesson/course context text so each chunk is self-describing once embedded.
- **`session_manager.py`** — In-memory (non-persistent) per-session message history, truncated to `MAX_HISTORY` exchanges. History is formatted as a plain text block and injected into the system prompt, not passed as structured message turns.
- **`models.py`** — Pydantic models `Course`, `Lesson`, `CourseChunk`. `Course.title` is the unique identifier used as the ChromaDB id in `course_catalog` and embedded in `course_content` chunk ids.
- **`config.py`** — Single `Config` dataclass (loaded from `.env` via `python-dotenv`) holding all tunables: chunk size/overlap, max search results, max history, embedding model name, Chroma path, Anthropic model/key.
- **`app.py`** — FastAPI app. Two endpoints: `POST /api/query`, `GET /api/courses`. On startup, auto-loads every file in `../docs` into the vector store (skipping courses whose title already exists — see `RAGSystem.add_course_folder`). Serves `frontend/` as static files at `/`, with no-cache headers for development.

### Course document format

Files in `docs/` (`.txt`, `.pdf`, `.docx`) are expected to follow:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```
`document_processor.py` parses this positionally (first lines = course metadata, then repeated lesson markers matched by regex) — it is not a generic document parser, and changes to the format need corresponding regex updates there.

### Data persistence

ChromaDB data persists on disk at `backend/chroma_db/` (path from `Config.CHROMA_PATH`, resolved relative to the backend working directory). Deleting that directory resets the knowledge base; it will be repopulated from `docs/` on next startup.
