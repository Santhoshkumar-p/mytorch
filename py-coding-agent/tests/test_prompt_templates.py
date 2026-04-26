"""Tests for coding_agent.core.prompt_templates."""
import pytest
from pathlib import Path


def test_substitute_positional():
    from coding_agent.core.prompt_templates import substitute_args
    assert substitute_args("Hello $1 and $2", ["world", "python"]) == "Hello world and python"


def test_substitute_all():
    from coding_agent.core.prompt_templates import substitute_args
    assert substitute_args("Args: $@", ["a", "b", "c"]) == "Args: a b c"


def test_substitute_arguments_alias():
    from coding_agent.core.prompt_templates import substitute_args
    assert substitute_args("Args: $ARGUMENTS", ["x", "y"]) == "Args: x y"


def test_substitute_slice():
    from coding_agent.core.prompt_templates import substitute_args
    # ${@:2} = args from index 2 (1-indexed) = args[1:]
    result = substitute_args("Rest: ${@:2}", ["first", "second", "third"])
    assert result == "Rest: second third"


def test_substitute_slice_with_length():
    from coding_agent.core.prompt_templates import substitute_args
    # ${@:1:2} = 2 args starting at index 1 (1-indexed) = args[0:2]
    result = substitute_args("Two: ${@:1:2}", ["a", "b", "c", "d"])
    assert result == "Two: a b"


def test_substitute_missing_positional():
    from coding_agent.core.prompt_templates import substitute_args
    # $3 with only 2 args → empty string
    result = substitute_args("$1 $2 $3", ["a", "b"])
    assert result == "a b "


def test_substitute_empty_args():
    from coding_agent.core.prompt_templates import substitute_args
    result = substitute_args("Hello $@", [])
    assert result == "Hello "


def test_substitute_no_placeholders():
    from coding_agent.core.prompt_templates import substitute_args
    result = substitute_args("No placeholders here", ["a", "b"])
    assert result == "No placeholders here"


def test_parse_command_args():
    from coding_agent.core.prompt_templates import parse_command_args
    assert parse_command_args('hello "world foo" bar') == ["hello", "world foo", "bar"]


def test_parse_command_args_simple():
    from coding_agent.core.prompt_templates import parse_command_args
    assert parse_command_args("one two three") == ["one", "two", "three"]


def test_parse_command_args_single_quotes():
    from coding_agent.core.prompt_templates import parse_command_args
    result = parse_command_args("hello 'world foo'")
    assert result == ["hello", "world foo"]


def test_parse_command_args_empty():
    from coding_agent.core.prompt_templates import parse_command_args
    result = parse_command_args("")
    assert result == []


def test_load_prompt_templates(tmp_path):
    from coding_agent.core.prompt_templates import load_prompt_templates
    f = tmp_path / "my-prompt.md"
    f.write_text("---\nname: my-prompt\ndescription: A test prompt\n---\nDo this task")
    templates = load_prompt_templates([str(f)])
    assert len(templates) == 1
    t = templates[0]
    assert t.name == "my-prompt"
    assert t.description == "A test prompt"
    assert "Do this task" in t.content


def test_load_prompt_templates_no_frontmatter(tmp_path):
    from coding_agent.core.prompt_templates import load_prompt_templates
    f = tmp_path / "simple.md"
    f.write_text("# Simple prompt\nDo something")
    templates = load_prompt_templates([str(f)])
    assert len(templates) == 1
    t = templates[0]
    # Name falls back to file stem
    assert t.name == "simple"
    # Description falls back to first non-empty non-header line
    assert t.description == "Simple prompt"


def test_load_prompt_templates_missing_file(tmp_path):
    from coding_agent.core.prompt_templates import load_prompt_templates
    templates = load_prompt_templates([str(tmp_path / "nonexistent.md")])
    assert templates == []


def test_load_prompt_templates_argument_hint(tmp_path):
    from coding_agent.core.prompt_templates import load_prompt_templates
    f = tmp_path / "argprompt.md"
    f.write_text("---\nname: argprompt\ndescription: test\nargument_hint: <filename>\n---\ncontent")
    templates = load_prompt_templates([str(f)])
    assert len(templates) == 1
    assert templates[0].argument_hint == "<filename>"


def test_load_prompt_templates_multiple(tmp_path):
    from coding_agent.core.prompt_templates import load_prompt_templates
    f1 = tmp_path / "p1.md"
    f2 = tmp_path / "p2.md"
    f1.write_text("---\nname: p1\ndescription: first\n---\nfirst content")
    f2.write_text("---\nname: p2\ndescription: second\n---\nsecond content")
    templates = load_prompt_templates([str(f1), str(f2)])
    assert len(templates) == 2
    names = [t.name for t in templates]
    assert "p1" in names
    assert "p2" in names
