"""Tests for CourseSearchTool.execute and the ToolManager that wraps it.

Unit tests use a mocked VectorStore. The integration tests at the bottom use a
real ChromaDB in a temp directory and are skipped if the embedding model cannot
be loaded (e.g. offline on first run).
"""

from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def _results(docs, metas):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


@pytest.fixture
def mock_store():
    store = MagicMock(name="VectorStore")
    store.search.return_value = _results([], [])
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# --- execute(): formatting ----------------------------------------------------


class TestExecuteFormatting:
    def test_formats_each_hit_with_course_and_lesson_header(self, tool, mock_store):
        mock_store.search.return_value = _results(
            ["MCP servers expose tools.", "Resources are read-only."],
            [
                {"course_title": "MCP Basics", "lesson_number": 2},
                {"course_title": "MCP Basics", "lesson_number": 3},
            ],
        )

        output = tool.execute(query="what are MCP servers")

        assert output == (
            "[MCP Basics - Lesson 2]\nMCP servers expose tools.\n\n"
            "[MCP Basics - Lesson 3]\nResources are read-only."
        )

    def test_header_omits_lesson_when_lesson_number_is_none(self, tool, mock_store):
        mock_store.search.return_value = _results(
            ["Course-level overview text."],
            [{"course_title": "MCP Basics", "lesson_number": None}],
        )

        output = tool.execute(query="overview")

        assert output == "[MCP Basics]\nCourse-level overview text."

    def test_missing_course_title_falls_back_to_unknown(self, tool, mock_store):
        mock_store.search.return_value = _results(
            ["orphan chunk"], [{"lesson_number": 1}]
        )

        output = tool.execute(query="x")

        assert output.startswith("[unknown - Lesson 1]")


# --- execute(): sources tracking ---------------------------------------------


class TestExecuteSources:
    def test_populates_last_sources_with_lesson_suffix(self, tool, mock_store):
        mock_store.search.return_value = _results(
            ["a", "b"],
            [
                {"course_title": "MCP Basics", "lesson_number": 2},
                {"course_title": "RAG 101", "lesson_number": None},
            ],
        )

        tool.execute(query="x")

        assert tool.last_sources == ["MCP Basics - Lesson 2", "RAG 101"]

    def test_last_sources_is_replaced_by_each_new_search(self, tool, mock_store):
        mock_store.search.return_value = _results(
            ["first"], [{"course_title": "A", "lesson_number": 1}]
        )
        tool.execute(query="first")

        mock_store.search.return_value = _results(
            ["second"], [{"course_title": "B", "lesson_number": 4}]
        )
        tool.execute(query="second")

        assert tool.last_sources == ["B - Lesson 4"]

    def test_duplicate_sources_are_collapsed_in_order(self, tool, mock_store):
        """Several chunks from one lesson should produce one source entry."""
        mock_store.search.return_value = _results(
            ["a", "b", "c"],
            [
                {"course_title": "MCP Basics", "lesson_number": 8},
                {"course_title": "RAG 101", "lesson_number": 1},
                {"course_title": "MCP Basics", "lesson_number": 8},
            ],
        )

        tool.execute(query="x")

        assert tool.last_sources == ["MCP Basics - Lesson 8", "RAG 101 - Lesson 1"]

    def test_empty_results_do_not_overwrite_previous_sources(self, tool, mock_store):
        """Documents current behaviour: an empty search returns early and leaves
        last_sources untouched, so stale sources from an earlier call survive."""
        mock_store.search.return_value = _results(
            ["hit"], [{"course_title": "A", "lesson_number": 1}]
        )
        tool.execute(query="hit")

        mock_store.search.return_value = _results([], [])
        tool.execute(query="miss")

        assert tool.last_sources == ["A - Lesson 1"]


# --- execute(): argument passing ---------------------------------------------


class TestExecuteArguments:
    def test_forwards_query_only(self, tool, mock_store):
        tool.execute(query="hello")

        mock_store.search.assert_called_once_with(
            query="hello", course_name=None, lesson_number=None
        )

    def test_forwards_course_name_and_lesson_number(self, tool, mock_store):
        tool.execute(query="hello", course_name="MCP", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="hello", course_name="MCP", lesson_number=3
        )

    def test_lesson_number_zero_is_forwarded_not_dropped(self, tool, mock_store):
        """Lesson 0 is a valid lesson in the document format ('Lesson 0: ...')."""
        tool.execute(query="intro", lesson_number=0)

        mock_store.search.assert_called_once_with(
            query="intro", course_name=None, lesson_number=0
        )


# --- execute(): errors and empty results -------------------------------------


class TestExecuteErrorsAndEmpty:
    def test_returns_store_error_verbatim(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty(
            "No course found matching 'Nope'"
        )

        assert tool.execute(query="x", course_name="Nope") == (
            "No course found matching 'Nope'"
        )
        assert tool.last_sources == []

    def test_empty_results_message_without_filters(self, tool, mock_store):
        assert tool.execute(query="x") == "No relevant content found."

    def test_empty_results_message_includes_course_filter(self, tool, mock_store):
        assert tool.execute(query="x", course_name="MCP") == (
            "No relevant content found in course 'MCP'."
        )

    def test_empty_results_message_includes_lesson_filter(self, tool, mock_store):
        assert tool.execute(query="x", lesson_number=5) == (
            "No relevant content found in lesson 5."
        )

    def test_empty_results_message_includes_both_filters(self, tool, mock_store):
        assert tool.execute(query="x", course_name="MCP", lesson_number=5) == (
            "No relevant content found in course 'MCP' in lesson 5."
        )


# --- tool definition ----------------------------------------------------------


class TestToolDefinition:
    def test_definition_name_matches_what_ai_generator_dispatches(self, tool):
        assert tool.get_tool_definition()["name"] == "search_course_content"

    def test_only_query_is_required(self, tool):
        schema = tool.get_tool_definition()["input_schema"]

        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query", "course_name", "lesson_number"}

    def test_lesson_number_is_declared_integer(self, tool):
        props = tool.get_tool_definition()["input_schema"]["properties"]

        assert props["lesson_number"]["type"] == "integer"


# --- ToolManager --------------------------------------------------------------


class TestToolManager:
    def test_registers_tool_under_its_definition_name(self, tool):
        manager = ToolManager()
        manager.register_tool(tool)

        names = [d["name"] for d in manager.get_tool_definitions()]
        assert names == ["search_course_content"]

    def test_execute_tool_dispatches_to_tool(self, tool, mock_store):
        manager = ToolManager()
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="q", lesson_number=1)

        mock_store.search.assert_called_once_with(
            query="q", course_name=None, lesson_number=1
        )

    def test_execute_unknown_tool_returns_message(self):
        assert ToolManager().execute_tool("nope", query="q") == (
            "Tool 'nope' not found"
        )

    def test_get_last_sources_reads_from_search_tool(self, tool, mock_store):
        manager = ToolManager()
        manager.register_tool(tool)
        mock_store.search.return_value = _results(
            ["x"], [{"course_title": "A", "lesson_number": 1}]
        )
        manager.execute_tool("search_course_content", query="x")

        assert manager.get_last_sources() == ["A - Lesson 1"]

    def test_reset_sources_clears_tool_state(self, tool, mock_store):
        manager = ToolManager()
        manager.register_tool(tool)
        mock_store.search.return_value = _results(
            ["x"], [{"course_title": "A", "lesson_number": 1}]
        )
        manager.execute_tool("search_course_content", query="x")

        manager.reset_sources()

        assert manager.get_last_sources() == []
        assert tool.last_sources == []


# --- integration: real ChromaDB + embeddings ---------------------------------


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """A real VectorStore seeded with a tiny course. Skips if embeddings fail."""
    from models import Course, CourseChunk, Lesson
    from vector_store import VectorStore

    path = tmp_path_factory.mktemp("chroma")
    try:
        store = VectorStore(str(path), "all-MiniLM-L6-v2", max_results=5)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Embedding model unavailable: {exc}")

    course = Course(
        title="MCP Basics",
        course_link="https://example.com/mcp",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Intro", lesson_link=None),
            Lesson(lesson_number=1, title="Servers", lesson_link=None),
        ],
    )
    chunks = [
        CourseChunk(
            content="An MCP server exposes tools and resources to a client.",
            course_title="MCP Basics",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Welcome to the course about the Model Context Protocol.",
            course_title="MCP Basics",
            lesson_number=0,
            chunk_index=1,
        ),
    ]
    store.add_course_metadata(course)
    store.add_course_content(chunks)
    return store


@pytest.mark.integration
class TestRealVectorStoreIntegration:
    def test_unfiltered_search_returns_content(self, real_store):
        output = CourseSearchTool(real_store).execute(query="MCP server tools")

        assert "No relevant content" not in output
        assert "[MCP Basics - Lesson 1]" in output

    def test_course_name_filter_resolves_partial_name(self, real_store):
        tool = CourseSearchTool(real_store)

        output = tool.execute(query="MCP server tools", course_name="MCP")

        assert "[MCP Basics" in output
        assert tool.last_sources

    def test_lesson_filter_restricts_results(self, real_store):
        tool = CourseSearchTool(real_store)

        tool.execute(query="MCP", lesson_number=0)

        assert tool.last_sources
        assert all(src.endswith("Lesson 0") for src in tool.last_sources)

    def test_course_and_lesson_filter_combined(self, real_store):
        """Exercises the $and filter path in VectorStore._build_filter."""
        tool = CourseSearchTool(real_store)

        output = tool.execute(query="MCP", course_name="MCP", lesson_number=1)

        assert "No relevant content" not in output and "Error" not in output
        assert all(src == "MCP Basics - Lesson 1" for src in tool.last_sources)
