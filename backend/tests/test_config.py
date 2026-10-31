"""Tests for Config: the API key must not appear in repr() output."""

from config import Config


def test_api_key_is_excluded_from_repr():
    config = Config(ANTHROPIC_API_KEY="sk-ant-secret-value")

    assert "sk-ant-secret-value" not in repr(config)


def test_default_model_is_a_supported_model_id():
    assert Config().ANTHROPIC_MODEL == "claude-sonnet-4-6"
