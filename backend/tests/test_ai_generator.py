"""Tests for AIGenerator: does it offer the search tool, and does it run it?

The Anthropic client is replaced with a scripted fake so no network calls are
made. Each test scripts the sequence of model responses it expects.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import EMPTY_RESPONSE_FALLBACK, AIGenerator
from config import Config
from search_tools import CourseSearchTool, ToolManager

MODEL = "test-model"


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, id_="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


@pytest.fixture
def fake_client():
    """Patch anthropic.Anthropic so AIGenerator gets a MagicMock client."""
    client = MagicMock(name="anthropic.Anthropic")
    with patch("ai_generator.anthropic.Anthropic", return_value=client):
        yield client


@pytest.fixture
def generator(fake_client):
    return AIGenerator(api_key="test-key", model=MODEL)


@pytest.fixture
def search_manager():
    """A ToolManager with a CourseSearchTool backed by a mocked store."""
    store = MagicMock(name="VectorStore")
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(store))
    return manager, store


def tools_for(manager):
    return manager.get_tool_definitions()


# --- Tool offering -----------------------------------------------------------


class TestToolOffering:
    def test_tools_and_auto_tool_choice_are_sent_to_api(
        self, generator, fake_client, search_manager
    ):
        manager, _ = search_manager
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        generator.generate_response(
            query="q", tools=tools_for(manager), tool_manager=manager
        )

        kwargs = fake_client.messages.create.call_args.kwargs
        assert kwargs["tools"] == tools_for(manager)
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_sent_when_no_tools_passed(self, generator, fake_client):
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        generator.generate_response(query="general knowledge question")

        kwargs = fake_client.messages.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_configured_model_is_passed_through(self, fake_client):
        """Guards the model ID from Config; a retired ID shows up as an API error
        in the live diagnostics, but this confirms the value is wired correctly."""
        generator = AIGenerator(api_key="k", model=Config.ANTHROPIC_MODEL)
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        generator.generate_response(query="q")

        assert (
            fake_client.messages.create.call_args.kwargs["model"]
            == Config.ANTHROPIC_MODEL
        )

    def test_system_prompt_forbids_source_attribution_phrases(self):
        prompt = AIGenerator.SYSTEM_PROMPT.lower()

        assert "based on the search results" in prompt
        assert "based on the course materials" in prompt

    def test_system_prompt_tells_model_when_to_search(self, generator, fake_client):
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        generator.generate_response(query="q")

        system = fake_client.messages.create.call_args.kwargs["system"]
        assert "search tool" in system.lower()
        assert "course-specific" in system.lower()

    def test_conversation_history_is_appended_to_system_prompt(
        self, generator, fake_client
    ):
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("hi")]
        )

        generator.generate_response(
            query="q", conversation_history="User: earlier\nAssistant: reply"
        )

        system = fake_client.messages.create.call_args.kwargs["system"]
        assert system.endswith(
            "Previous conversation:\nUser: earlier\nAssistant: reply"
        )


# --- Tool-use decision ------------------------------------------------------


class TestToolUseDecision:
    def test_direct_answer_does_not_execute_tool(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager
        fake_client.messages.create.return_value = response(
            "end_turn", [text_block("Paris is the capital of France.")]
        )

        answer = generator.generate_response(
            query="capital of France?", tools=tools_for(manager), tool_manager=manager
        )

        assert answer == "Paris is the capital of France."
        store.search.assert_not_called()
        assert fake_client.messages.create.call_count == 1

    def test_tool_use_stop_reason_executes_search_with_model_arguments(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager
        store.search.return_value = MagicMock(error=None)
        store.search.return_value.is_empty.return_value = True
        fake_client.messages.create.side_effect = [
            response(
                "tool_use",
                [
                    tool_use_block(
                        "search_course_content",
                        {"query": "MCP servers", "course_name": "MCP", "lesson_number": 2},
                    )
                ],
            ),
            response("end_turn", [text_block("final")]),
        ]

        generator.generate_response(
            query="q", tools=tools_for(manager), tool_manager=manager
        )

        store.search.assert_called_once_with(
            query="MCP servers", course_name="MCP", lesson_number=2
        )

    def test_tool_use_stop_reason_without_tool_manager_returns_text_block(
        self, generator, fake_client
    ):
        """If the caller did not pass tool_manager, the tool_use response has no
        text block, so the fallback message is returned instead of an error."""
        fake_client.messages.create.return_value = response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "x"})],
        )

        answer = generator.generate_response(query="q", tools=[{"name": "x"}])

        assert answer == EMPTY_RESPONSE_FALLBACK


# --- Two-round message flow -------------------------------------------------


class TestToolRoundTrip:
    def _run_tool_round(self, generator, fake_client, manager, store, final_text):
        store.search.return_value = MagicMock(error=None)
        store.search.return_value.is_empty.return_value = False
        store.search.return_value.documents = ["MCP servers expose tools."]
        store.search.return_value.metadata = [
            {"course_title": "MCP Basics", "lesson_number": 2}
        ]
        fake_client.messages.create.side_effect = [
            response(
                "tool_use",
                [tool_use_block("search_course_content", {"query": "MCP servers"}, "tu_42")],
            ),
            response("end_turn", [text_block(final_text)]),
        ]
        return generator.generate_response(
            query="What are MCP servers?",
            tools=tools_for(manager),
            tool_manager=manager,
        )

    def test_returns_final_text_after_tool_round(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager

        answer = self._run_tool_round(
            generator, fake_client, manager, store, "They expose tools."
        )

        assert answer == "They expose tools."
        assert fake_client.messages.create.call_count == 2

    def test_follow_up_call_has_no_tools(self, generator, fake_client, search_manager):
        """Documents the single-round design: the second call strips tools."""
        manager, store = search_manager
        self._run_tool_round(generator, fake_client, manager, store, "done")

        follow_up = fake_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in follow_up
        assert "tool_choice" not in follow_up

    def test_follow_up_carries_tool_result_with_matching_id(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager
        self._run_tool_round(generator, fake_client, manager, store, "done")

        messages = fake_client.messages.create.call_args_list[1].kwargs["messages"]
        assert messages[0] == {"role": "user", "content": "What are MCP servers?"}
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        result = messages[2]["content"][0]
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu_42"
        assert "MCP servers expose tools." in result["content"]

    def test_follow_up_keeps_system_prompt(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager
        self._run_tool_round(generator, fake_client, manager, store, "done")

        first = fake_client.messages.create.call_args_list[0].kwargs["system"]
        second = fake_client.messages.create.call_args_list[1].kwargs["system"]
        assert first == second

    def test_tool_error_string_is_still_sent_back_to_model(
        self, generator, fake_client, search_manager
    ):
        manager, store = search_manager
        store.search.return_value = MagicMock(error="No course found matching 'X'")
        fake_client.messages.create.side_effect = [
            response(
                "tool_use",
                [tool_use_block("search_course_content", {"query": "q", "course_name": "X"})],
            ),
            response("end_turn", [text_block("I could not find that course.")]),
        ]

        generator.generate_response(
            query="q", tools=tools_for(manager), tool_manager=manager
        )

        result = fake_client.messages.create.call_args_list[1].kwargs["messages"][2][
            "content"
        ][0]
        assert result["content"] == "No course found matching 'X'"


# --- Robustness -------------------------------------------------------------


class TestRobustness:
    def test_empty_final_content_returns_fallback(
        self, generator, fake_client, search_manager
    ):
        """If the follow-up response has no content blocks, return a fallback
        message instead of raising IndexError (which surfaced as an HTTP 500)."""
        manager, store = search_manager
        store.search.return_value = MagicMock(error=None)
        store.search.return_value.is_empty.return_value = True
        fake_client.messages.create.side_effect = [
            response(
                "tool_use",
                [tool_use_block("search_course_content", {"query": "q"})],
            ),
            response("end_turn", []),
        ]

        answer = generator.generate_response(
            query="q", tools=tools_for(manager), tool_manager=manager
        )

        assert answer == EMPTY_RESPONSE_FALLBACK

    def test_empty_direct_content_returns_fallback(self, generator, fake_client):
        fake_client.messages.create.return_value = response("end_turn", [])

        assert generator.generate_response(query="q") == EMPTY_RESPONSE_FALLBACK

    def test_non_text_only_content_returns_fallback(self, generator, fake_client):
        fake_client.messages.create.return_value = response(
            "end_turn", [tool_use_block("search_course_content", {"query": "q"})]
        )

        assert generator.generate_response(query="q") == EMPTY_RESPONSE_FALLBACK

    def test_api_exception_propagates(self, generator, fake_client):
        fake_client.messages.create.side_effect = RuntimeError("401 invalid api key")

        with pytest.raises(RuntimeError, match="401"):
            generator.generate_response(query="q")
