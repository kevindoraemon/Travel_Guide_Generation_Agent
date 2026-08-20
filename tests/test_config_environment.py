from pathlib import Path

from travel_planner import llm
from travel_planner.utils import load_config


def test_load_config_resolves_environment_references(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
stages:
  test:
    service:
      api_key: ${TEST_SERVICE_API_KEY}
      nested:
        - ${TEST_SECONDARY_KEY}
        - plain-value
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_SERVICE_API_KEY", "primary-secret")
    monkeypatch.setenv("TEST_SECONDARY_KEY", "secondary-secret")

    config = load_config(stage_name="test", config_path=str(config_path))

    assert config["service"]["api_key"] == "primary-secret"
    assert config["service"]["nested"] == ["secondary-secret", "plain-value"]


def test_missing_environment_reference_resolves_to_empty_string(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "stages:\n  test:\n    api_key: ${MISSING_TEST_API_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    config = load_config(stage_name="test", config_path=str(config_path))

    assert config["api_key"] == ""


def test_role_max_tokens_override_model_default(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
stages:
  test:
    cognition:
      openai:
        api_key: test
        models:
          model-a:
            max_tokens: 999
    roles:
      writer:
        backend: openai
        handle: model-a
        max_tokens: 123
""".strip(),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setattr(llm, "init_chat_model", lambda **kwargs: captured.update(kwargs) or kwargs)
    llm._CONFIG_CACHE.clear()

    llm.get_chat_model("writer", stage="test")

    assert captured["max_tokens"] == 123
