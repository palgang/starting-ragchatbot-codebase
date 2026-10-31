"""Tests for RAGSystem.query and the content-question path through it.

The vector store, document processor and Anthropic client are replaced with
test doubles. AIGenerator, ToolManager and CourseSearchTool are the real
classes, so the orchestration between them is exercised.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models import Course, Lesson
from rag_system import RAGSystem
from vector_store import SearchResults


@pytest.fixture
def config():
    return SimpleNamespace(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="test-model",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="unused",
        EMBEDDING_MODEL="unused",
    )


@pytest.fixture
def anthropic_client():
    client = MagicMock(name="anthropic.Anthropic")
    with patch("ai_generator.anthropic.Anthropic", return_value=client):
        yield client


@pytest.fixture
def rag(config, anthropic_client):
    """RAGSystem with a mocked vector store and processor; real generator/tools."""
    with patch("rag_system.VectorStore") as vs_cls, patch(
        "rag_system.DocumentProcessor"
    ) as dp_cls:
        system = RAGSystem(config)
        system.vector_store = vs_cls.return_value
        system.document_processor = dp_cls.return_value
        # Keep the real CourseSearchTool but point it at the mocked store.
        system.search_tool.store = system.vector_store
        yield system


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(input_, id_="tu_1"):
    return SimpleNamespace(
        type="tool_use", name="search_course_content", input=input_, id=id_
    )


def response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def hits(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


# --- Wiring -----------------------------------------------------------------


class TestWiring:
    def test_search_tool_is_registered_with_tool_manager(self, rag):
        names = [d["name"] for d in rag.tool_manager.get_tool_definitions()]
        assert names == ["search_course_content"]

    def test_generator_receives_tools_and_tool_manager(self, rag, anthropic_client):
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        rag.query("what is RAG?")

        kwargs = anthropic_client.messages.create.call_args.kwargs
        assert kwargs["tools"][0]["name"] == "search_course_content"
        assert kwargs["tool_choice"] == {"type": "auto"}


# --- Query behaviour without a session -----------------------------------------


class TestQueryNoSession:
    def test_prompt_wraps_user_question(self, rag, anthropic_client):
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("ok")]
        )

        rag.query("What is MCP?")

        message = anthropic_client.messages.create.call_args.kwargs["messages"][0]
        assert message == {
            "role": "user",
            "content": "Answer this question about course materials: What is MCP?",
        }

    def test_returns_text_and_empty_sources_for_direct_answer(
        self, rag, anthropic_client
    ):
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("general answer")]
        )

        answer, sources = rag.query("general question")

        assert answer == "general answer"
        assert sources == []

    def test_no_session_history_is_injected_without_session_id(
        self, rag, anthropic_client
    ):
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("ok")]
        )

        rag.query("q")

        assert "Previous conversation" not in anthropic_client.messages.create.call_args.kwargs[
            "system"
        ]


# --- Content-question path (the one that reports "query failed") -------------


class TestContentQuestionFlow:
    def _script_tool_then_answer(self, anthropic_client, tool_input, final_text):
        anthropic_client.messages.create.side_effect = [
            response("tool_use", [tool_use_block(tool_input)]),
            response("end_turn", [text_block(final_text)]),
        ]

    def test_content_question_returns_answer_and_sources(
        self, rag, anthropic_client
    ):
        rag.vector_store.search.return_value = hits(
            ["MCP servers expose tools and resources."],
            [{"course_title": "MCP Basics", "lesson_number": 2}],
        )
        self._script_tool_then_answer(
            anthropic_client,
            {"query": "What are MCP servers?"},
            "MCP servers expose tools and resources.",
        )

        answer, sources = rag.query("What are MCP servers?")

        assert answer == "MCP servers expose tools and resources."
        assert sources == ["MCP Basics - Lesson 2"]
        rag.vector_store.search.assert_called_once_with(
            query="What are MCP servers?", course_name=None, lesson_number=None
        )

    def test_course_filtered_question_passes_course_name_through(
        self, rag, anthropic_client
    ):
        rag.vector_store.search.return_value = hits(
            ["chunk"], [{"course_title": "MCP Basics", "lesson_number": 1}]
        )
        self._script_tool_then_answer(
            anthropic_client,
            {"query": "servers", "course_name": "MCP", "lesson_number": 1},
            "answer",
        )

        rag.query("servers in MCP lesson 1")

        rag.vector_store.search.assert_called_once_with(
            query="servers", course_name="MCP", lesson_number=1
        )

    def test_sources_are_reset_after_each_query(self, rag, anthropic_client):
        rag.vector_store.search.return_value = hits(
            ["chunk"], [{"course_title": "A", "lesson_number": 1}]
        )
        self._script_tool_then_answer(anthropic_client, {"query": "x"}, "first")
        rag.query("first question")

        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("second")]
        )
        _, sources = rag.query("general question")

        assert sources == []

    def test_store_error_is_passed_to_model_and_query_still_returns(
        self, rag, anthropic_client
    ):
        """A course-name miss must not crash query(); the model should get the
        error string and answer from it."""
        rag.vector_store.search.return_value = SearchResults.empty(
            "No course found matching 'Nope'"
        )
        self._script_tool_then_answer(
            anthropic_client,
            {"query": "x", "course_name": "Nope"},
            "I could not find that course.",
        )

        answer, sources = rag.query("question about Nope")

        assert answer == "I could not find that course."
        assert sources == []

    def test_empty_store_returns_answer_without_crashing(
        self, rag, anthropic_client
    ):
        """If the store is empty (e.g. docs were never loaded), the tool returns
        'No relevant content found.' and the model should still produce text."""
        rag.vector_store.search.return_value = hits([], [])
        self._script_tool_then_answer(
            anthropic_client, {"query": "x"}, "No relevant content."
        )

        answer, sources = rag.query("content question")

        assert answer == "No relevant content."
        assert sources == []

    def test_api_failure_during_content_query_propagates(
        self, rag, anthropic_client
    ):
        """Documents current behaviour: exceptions propagate out of query();
        app.py converts them to HTTP 500, which the frontend shows as 'query failed'."""
        anthropic_client.messages.create.side_effect = RuntimeError("model not found")

        with pytest.raises(RuntimeError, match="model not found"):
            rag.query("content question")

    def test_search_exception_during_content_query_propagates(
        self, rag, anthropic_client
    ):
        rag.vector_store.search.side_effect = RuntimeError("chroma exploded")
        anthropic_client.messages.create.side_effect = [
            response("tool_use", [tool_use_block({"query": "x"})]),
        ]

        with pytest.raises(RuntimeError, match="chroma exploded"):
            rag.query("content question")


# --- Session history ----------------------------------------------------------


class TestSessionHistory:
    def test_exchange_is_recorded_under_session(self, rag, anthropic_client):
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("first answer")]
        )
        session_id = rag.session_manager.create_session()

        rag.query("first question", session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history == "User: first question\nAssistant: first answer"

    def test_prior_history_is_sent_on_follow_up(self, rag, anthropic_client):
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "earlier q", "earlier a")
        anthropic_client.messages.create.return_value = response(
            "end_turn", [text_block("follow-up answer")]
        )

        rag.query("follow-up", session_id)

        system = anthropic_client.messages.create.call_args.kwargs["system"]
        assert "Previous conversation:\nUser: earlier q\nAssistant: earlier a" in system


# --- Document ingestion (add_course_folder) -----------------------------------


class TestAddCourseFolder:
    def test_missing_folder_returns_zero(self, rag, tmp_path):
        assert rag.add_course_folder(str(tmp_path / "missing")) == (0, 0)

    def test_skips_courses_already_in_store(self, rag, tmp_path):
        (tmp_path / "course.txt").write_text("x", encoding="utf-8")
        rag.vector_store.get_existing_course_titles.return_value = ["MCP Basics"]
        course = Course(title="MCP Basics", lessons=[Lesson(lesson_number=0, title="t")])
        rag.document_processor.process_course_document.return_value = (course, [])

        courses, chunks = rag.add_course_folder(str(tmp_path))

        assert (courses, chunks) == (0, 0)
        rag.vector_store.add_course_metadata.assert_not_called()

    def test_adds_new_course(self, rag, tmp_path):
        (tmp_path / "course.txt").write_text("x", encoding="utf-8")
        rag.vector_store.get_existing_course_titles.return_value = []
        course = Course(title="RAG 101", lessons=[Lesson(lesson_number=0, title="t")])
        chunks = [MagicMock(), MagicMock()]
        rag.document_processor.process_course_document.return_value = (course, chunks)

        courses, n_chunks = rag.add_course_folder(str(tmp_path))

        assert (courses, n_chunks) == (1, 2)
        rag.vector_store.add_course_metadata.assert_called_once_with(course)
        rag.vector_store.add_course_content.assert_called_once_with(chunks)
