"""Shared pytest fixtures for the RAG system tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from test_app_factory import create_test_app


@pytest.fixture
def sample_course_analytics():
    """Analytics payload shaped like RAGSystem.get_course_analytics()."""
    return {
        "total_courses": 2,
        "course_titles": ["Intro to RAG", "Advanced Retrieval"],
    }


@pytest.fixture
def sample_query_result():
    """(answer, sources) tuple shaped like RAGSystem.query()."""
    return (
        "Retrieval-augmented generation combines search with an LLM.",
        ["Intro to RAG - Lesson 1"],
    )


@pytest.fixture
def mock_session_manager():
    manager = MagicMock(name="SessionManager")
    manager.create_session.return_value = "session-123"
    return manager


@pytest.fixture
def mock_rag_system(mock_session_manager, sample_course_analytics, sample_query_result):
    """A RAGSystem stand-in with no vector store, LLM, or disk access."""
    rag = MagicMock(name="RAGSystem")
    rag.session_manager = mock_session_manager
    rag.query.return_value = sample_query_result
    rag.get_course_analytics.return_value = sample_course_analytics
    return rag


@pytest.fixture
def frontend_dir(tmp_path):
    """A throwaway frontend directory so static serving does not depend on the repo."""
    root = tmp_path / "frontend"
    root.mkdir()
    (root / "index.html").write_text(
        "<!doctype html><html><body>RAG UI</body></html>", encoding="utf-8"
    )
    (root / "script.js").write_text("console.log('ok');", encoding="utf-8")
    return root


@pytest.fixture
def api_app(mock_rag_system, frontend_dir):
    return create_test_app(mock_rag_system, frontend_dir)


@pytest.fixture
def client(api_app):
    with TestClient(api_app) as test_client:
        yield test_client
