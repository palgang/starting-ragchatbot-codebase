"""API endpoint tests for /api/query, /api/courses, and the static root (/)."""

import pytest


class TestQueryEndpoint:
    def test_returns_answer_sources_and_session(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "answer": "Retrieval-augmented generation combines search with an LLM.",
            "sources": ["Intro to RAG - Lesson 1"],
            "session_id": "session-123",
        }

    def test_creates_session_when_none_provided(self, client, mock_session_manager):
        client.post("/api/query", json={"query": "hello"})

        mock_session_manager.create_session.assert_called_once_with()

    def test_reuses_provided_session_id(
        self, client, mock_rag_system, mock_session_manager
    ):
        response = client.post(
            "/api/query",
            json={"query": "follow-up", "session_id": "existing-session"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-session"
        mock_session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("follow-up", "existing-session")

    def test_passes_query_and_new_session_to_rag(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "explain chunking"})

        mock_rag_system.query.assert_called_once_with("explain chunking", "session-123")

    def test_empty_sources_are_returned_as_empty_list(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No relevant material found.", [])

        response = client.post("/api/query", json={"query": "unknown topic"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"session_id": "abc"},
            {"query": None},
        ],
    )
    def test_invalid_payload_returns_422(self, client, mock_rag_system, payload):
        response = client.post("/api/query", json=payload)

        assert response.status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_rag_failure_returns_500_with_detail(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "anything"})

        assert response.status_code == 500
        assert response.json() == {"detail": "vector store unavailable"}

    def test_session_creation_failure_returns_500(self, client, mock_session_manager):
        mock_session_manager.create_session.side_effect = RuntimeError(
            "session store down"
        )

        response = client.post("/api/query", json={"query": "anything"})

        assert response.status_code == 500
        assert response.json()["detail"] == "session store down"


class TestCoursesEndpoint:
    def test_returns_course_stats(self, client, sample_course_analytics):
        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": sample_course_analytics["total_courses"],
            "course_titles": sample_course_analytics["course_titles"],
        }

    def test_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {"total_courses": 0, "course_titles": []}

    def test_analytics_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError(
            "chroma offline"
        )

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json() == {"detail": "chroma offline"}

    def test_rejects_post(self, client):
        response = client.post("/api/courses", json={})

        assert response.status_code == 405


class TestRootEndpoint:
    def test_serves_index_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "RAG UI" in response.text
        assert response.headers["content-type"].startswith("text/html")

    def test_serves_static_assets(self, client):
        response = client.get("/script.js")

        assert response.status_code == 200
        assert "console.log" in response.text

    def test_unknown_static_path_returns_404(self, client):
        response = client.get("/does-not-exist.css")

        assert response.status_code == 404


class TestCors:
    def test_preflight_allows_any_origin_with_credentials(self, client):
        # With allow_credentials=True, Starlette echoes the request origin
        # instead of returning a literal "*".
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://example.com"
        assert response.headers["access-control-allow-credentials"] == "true"
