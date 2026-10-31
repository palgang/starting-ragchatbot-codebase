"""Live diagnostics against the real Anthropic API and the real ChromaDB on disk.

These tests are opt-in because they make small API calls and load the
sentence-transformers model. Enable them with:

    RUN_LIVE_DIAGNOSTICS=1 uv run pytest backend/tests/test_live_diagnostics.py -v

Each test checks one link in the chain behind "query failed", in order, so the
first failing test names the broken component.
"""

import dataclasses
import os
from pathlib import Path

import pytest

from config import Config

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DIAGNOSTICS") != "1",
    reason="Set RUN_LIVE_DIAGNOSTICS=1 to run live diagnostics",
)

# Absolute path so the result does not depend on the current working directory.
# Config.CHROMA_PATH is relative ("./chroma_db"), so running from the repo root
# silently opens an empty database.
CHROMA_PATH = str(Path(__file__).resolve().parents[1] / "chroma_db")


@pytest.fixture(scope="module")
def live_config():
    return dataclasses.replace(Config(), CHROMA_PATH=CHROMA_PATH)


@pytest.fixture(scope="module")
def live_store(live_config):
    from vector_store import VectorStore

    return VectorStore(
        live_config.CHROMA_PATH, live_config.EMBEDDING_MODEL, live_config.MAX_RESULTS
    )


# 1. Configuration -----------------------------------------------------------


def test_api_key_is_configured():
    key = Config().ANTHROPIC_API_KEY
    assert key, "ANTHROPIC_API_KEY is empty. Check the .env file in the repo root."


# 2. Anthropic API -----------------------------------------------------------


def test_configured_model_accepts_minimal_request(live_config):
    import anthropic

    client = anthropic.Anthropic(api_key=live_config.ANTHROPIC_API_KEY)
    try:
        reply = client.messages.create(
            model=live_config.ANTHROPIC_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "Say ok"}],
        )
    except anthropic.APIError as exc:
        pytest.fail(
            f"Model '{live_config.ANTHROPIC_MODEL}' rejected the request: {exc}"
        )

    assert reply.content, "Model returned no content blocks"


# 3. ChromaDB contents -------------------------------------------------------


def test_chroma_catalog_has_courses(live_store):
    titles = live_store.get_existing_course_titles()

    assert titles, (
        f"course_catalog is empty at {CHROMA_PATH}. Documents were not loaded; "
        "startup reads ../docs relative to the working directory."
    )


def test_chroma_content_collection_has_chunks(live_store):
    count = live_store.course_content.count()

    assert count > 0, "course_content collection has no chunks"


# 4. Vector search -----------------------------------------------------------


def test_unfiltered_search_returns_results_without_error(live_store):
    results = live_store.search(query="course overview")

    assert results.error is None, f"Search returned an error: {results.error}"
    assert not results.is_empty(), "Unfiltered search returned zero chunks"


def test_course_filtered_search_resolves_a_real_course(live_store):
    titles = live_store.get_existing_course_titles()
    if not titles:
        pytest.skip("No courses loaded; covered by test_chroma_catalog_has_courses")

    title = titles[0]
    results = live_store.search(query=title, course_name=title)

    assert results.error is None, f"Filtered search errored: {results.error}"
    assert not results.is_empty(), f"Filtered search for '{title}' returned nothing"
    assert all(m["course_title"] == title for m in results.metadata)


# 5. Full pipeline -----------------------------------------------------------


def test_full_query_on_content_question_returns_answer(live_config):
    from rag_system import RAGSystem

    rag = RAGSystem(live_config)
    if not rag.vector_store.get_existing_course_titles():
        pytest.skip("No courses loaded; cannot test a content question")

    answer, sources = rag.query("What topics do the courses cover?")

    assert answer and answer.strip(), "query() returned an empty answer"
    assert sources, (
        "Content question produced no sources, so the search tool was likely "
        "not called or returned nothing"
    )
