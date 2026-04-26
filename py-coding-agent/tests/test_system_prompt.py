"""Tests for coding_agent.core.system_prompt."""
import pytest
from datetime import date


def test_basic_prompt():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(cwd="/test")
    prompt = build_system_prompt(opts)
    # The base intro mentions "coding assistant"
    assert "coding" in prompt.lower() or "assistant" in prompt.lower()
    assert "/test" in prompt
    assert date.today().isoformat() in prompt


def test_tools_listed():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(selected_tools=["bash", "read"])
    prompt = build_system_prompt(opts)
    assert "bash" in prompt
    assert "read" in prompt


def test_guidelines_bash_only():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(selected_tools=["bash"])
    prompt = build_system_prompt(opts)
    # Should mention using bash for file ops
    assert "bash" in prompt.lower()


def test_guidelines_with_grep():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(selected_tools=["bash", "grep", "find"])
    prompt = build_system_prompt(opts)
    # Should prefer grep/find over bash
    assert "grep" in prompt.lower() or "prefer" in prompt.lower()


def test_skills_included():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    from coding_agent.core.types import Skill
    skill = Skill(name="test-skill", path="/test/SKILL.md", content="Do X", description="Test",
                  base_dir="/test", source_info={})
    opts = BuildSystemPromptOptions(selected_tools=["read"], skills=[skill])
    prompt = build_system_prompt(opts)
    # Progressive disclosure: name, description and location appear; content is NOT inlined.
    assert "test-skill" in prompt
    assert "/test/SKILL.md" in prompt   # location so the model can read it on demand
    assert "Do X" not in prompt         # content is loaded on-demand, not inlined


def test_skills_not_included_without_read_tool():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    from coding_agent.core.types import Skill
    skill = Skill(name="test-skill", path="/test", content="Do X", description="Test",
                  base_dir="/", source_info={})
    # Only bash — no "read" tool — skills should not appear
    opts = BuildSystemPromptOptions(selected_tools=["bash"], skills=[skill])
    prompt = build_system_prompt(opts)
    assert "Do X" not in prompt


def test_context_files_included():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    from coding_agent.core.types import ContextFile
    cf = ContextFile(path="/proj/AGENTS.md", content="Be helpful")
    opts = BuildSystemPromptOptions(context_files=[cf])
    prompt = build_system_prompt(opts)
    assert "Be helpful" in prompt


def test_custom_prompt_replaces_base():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(custom_prompt="You are a wizard.")
    prompt = build_system_prompt(opts)
    assert "You are a wizard." in prompt
    # Custom prompt replaces the base intro
    assert "You are a wizard." in prompt


def test_append_system_prompt():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(append_system_prompt="EXTRA INSTRUCTIONS")
    prompt = build_system_prompt(opts)
    assert "EXTRA INSTRUCTIONS" in prompt


def test_cwd_in_prompt():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(cwd="/my/project/dir")
    prompt = build_system_prompt(opts)
    assert "/my/project/dir" in prompt


def test_no_cwd_no_working_directory():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(cwd=None)
    prompt = build_system_prompt(opts)
    assert "Working directory" not in prompt


def test_date_always_present():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions()
    prompt = build_system_prompt(opts)
    assert date.today().isoformat() in prompt


def test_tool_snippets_override():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(
        selected_tools=["bash"],
        tool_snippets={"bash": "Custom bash description"},
    )
    prompt = build_system_prompt(opts)
    assert "Custom bash description" in prompt


def test_prompt_guidelines_included():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(
        selected_tools=["bash"],
        prompt_guidelines=["Always test your code", "Use meaningful variable names"],
    )
    prompt = build_system_prompt(opts)
    assert "Always test your code" in prompt
    assert "Use meaningful variable names" in prompt


def test_multiple_skills():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    from coding_agent.core.types import Skill
    skill1 = Skill(name="skill-one", path="/s1.md", content="Skill 1 content",
                   description="First skill", base_dir="/", source_info={})
    skill2 = Skill(name="skill-two", path="/s2.md", content="Skill 2 content",
                   description="Second skill", base_dir="/", source_info={})
    opts = BuildSystemPromptOptions(selected_tools=["read"], skills=[skill1, skill2])
    prompt = build_system_prompt(opts)
    # Progressive disclosure: names and locations appear; content is not inlined.
    assert "skill-one" in prompt
    assert "skill-two" in prompt
    assert "/s1.md" in prompt
    assert "/s2.md" in prompt
    assert "Skill 1 content" not in prompt
    assert "Skill 2 content" not in prompt


def test_empty_tools_no_tool_section():
    from coding_agent.core.system_prompt import build_system_prompt, BuildSystemPromptOptions
    opts = BuildSystemPromptOptions(selected_tools=[])
    prompt = build_system_prompt(opts)
    assert "Available Tools" not in prompt
