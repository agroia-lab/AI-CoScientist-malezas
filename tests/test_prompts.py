"""Tests for configurable prompt loading."""

import os

import pytest

from ai_coscientist.prompts import (
    get_evolution_prompt,
    get_generation_prompt,
    get_meta_review_prompt,
    get_proximity_prompt,
    get_ranking_prompt,
    get_reflection_prompt,
    get_supervisor_prompt,
    get_tournament_prompt,
    load_prompt,
)

# -- load_prompt helper -------------------------------------------


def test_load_prompt_none_returns_default():
    result = load_prompt(None, lambda: "default")
    assert result == "default"


def test_load_prompt_empty_string_returns_default():
    result = load_prompt("", lambda: "default")
    assert result == "default"


def test_load_prompt_whitespace_only_returns_default():
    result = load_prompt("   ", lambda: "default")
    assert result == "default"


def test_load_prompt_custom_string():
    result = load_prompt("my prompt", lambda: "default")
    assert result == "my prompt"


def test_load_prompt_file_path(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt from file", encoding="utf-8")
    result = load_prompt(str(prompt_file), lambda: "default")
    assert result == "prompt from file"


def test_load_prompt_nonexistent_file():
    """A path that doesn't exist is treated as a string."""
    path = "/tmp/nonexistent_prompt_file_xyz.txt"
    assert not os.path.isfile(path)
    result = load_prompt(path, lambda: "default")
    assert result == path


# -- get_*_prompt with defaults -----------------------------------

ALL_PROMPT_FNS = [
    get_generation_prompt,
    get_reflection_prompt,
    get_ranking_prompt,
    get_evolution_prompt,
    get_meta_review_prompt,
    get_proximity_prompt,
    get_tournament_prompt,
    get_supervisor_prompt,
]


@pytest.mark.parametrize("fn", ALL_PROMPT_FNS)
def test_default_prompt_non_empty(fn):
    result = fn()
    assert isinstance(result, str)
    assert len(result) > 50


@pytest.mark.parametrize("fn", ALL_PROMPT_FNS)
def test_custom_string_overrides_default(fn):
    result = fn(custom_prompt="Custom agent prompt")
    assert result == "Custom agent prompt"


@pytest.mark.parametrize("fn", ALL_PROMPT_FNS)
def test_none_returns_builtin_default(fn):
    assert fn(custom_prompt=None) == fn()


@pytest.mark.parametrize("fn", ALL_PROMPT_FNS)
def test_file_path_overrides_default(fn, tmp_path):
    prompt_file = tmp_path / "custom.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")
    result = fn(custom_prompt=str(prompt_file))
    assert result == "file prompt"


# -- Integration: custom_prompts in framework init ----------------


def test_framework_accepts_custom_prompts(framework):
    """Framework stores custom_prompts dict."""
    assert framework.custom_prompts == {}


def test_framework_custom_prompts_passed(
    mock_agent,
):
    """custom_prompts dict is stored on the instance."""
    from unittest.mock import patch, MagicMock

    with patch("ai_coscientist.main.DirectLLMAgent") as MA:
        MA.return_value = MagicMock()
        from ai_coscientist import AIScientistFramework

        fw = AIScientistFramework(
            model_name="test",
            max_iterations=1,
            custom_prompts={"generation": "custom gen"},
        )
        assert fw.custom_prompts == {"generation": "custom gen"}
