"""Tests for coding_agent.core.skills."""
import pytest


def test_load_skill_valid(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "my-skill.md"
    f.write_text("---\nname: my-skill\ndescription: Does things\ncommands: [/my-skill]\n---\n# Content\nHello")
    skill = load_skill(str(f))
    assert skill is not None
    assert skill.name == "my-skill"
    assert skill.description == "Does things"
    assert "my-skill" in skill.commands
    assert "Hello" in skill.content


def test_load_skill_invalid_name(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "bad.md"
    f.write_text("---\nname: BAD NAME WITH SPACES\ndescription: test\n---\ncontent")
    result = load_skill(str(f))
    assert result is None


def test_load_skill_name_starts_with_uppercase(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "Bad.md"
    f.write_text("---\nname: Bad\ndescription: test\n---\ncontent")
    result = load_skill(str(f))
    assert result is None


def test_load_skill_no_frontmatter(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "no-fm.md"
    f.write_text("# Just content\nNo frontmatter")
    # Should use filename as name — "no-fm" is valid (matches [a-z0-9][a-z0-9-]{0,62})
    # Either loads or returns None — just no crash
    result = load_skill(str(f))
    # If loaded, name should match file stem
    if result is not None:
        assert result.name == "no-fm"


def test_load_skill_missing_file(tmp_path):
    from coding_agent.core.skills import load_skill
    result = load_skill(str(tmp_path / "nonexistent.md"))
    assert result is None


def test_load_skill_path_stored(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "valid-skill.md"
    f.write_text("---\nname: valid-skill\ndescription: test\n---\ncontent")
    skill = load_skill(str(f))
    assert skill is not None
    assert skill.path == str(f)
    assert skill.base_dir == str(tmp_path)


def test_load_skill_tags_parsed(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "tagged-skill.md"
    f.write_text("---\nname: tagged-skill\ndescription: test\ntags: [python, code]\n---\ncontent")
    skill = load_skill(str(f))
    assert skill is not None
    assert "python" in skill.tags
    assert "code" in skill.tags


def test_load_skill_disable_model_invocation(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "no-llm-skill.md"
    f.write_text("---\nname: no-llm-skill\ndescription: test\ndisable_model_invocation: true\n---\ncontent")
    skill = load_skill(str(f))
    assert skill is not None
    assert skill.disable_model_invocation is True


def test_load_skills_from_dir_with_skill_md(tmp_path):
    from coding_agent.core.skills import load_skills_from_dir
    # Create SKILL.md marker + skills
    (tmp_path / "SKILL.md").write_text("# Index")
    for name in ["skill-a", "skill-b"]:
        (tmp_path / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: test\n---\ncontent"
        )
    # Create subdir with more skills
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.md").write_text("---\nname: nested\ndescription: test\n---\ncontent")

    skills = load_skills_from_dir(str(tmp_path))
    names = [s.name for s in skills]
    assert "skill-a" in names
    assert "skill-b" in names
    # nested should not be here since SKILL.md is in root → only direct children
    assert "nested" not in names


def test_load_skills_from_dir_without_skill_md(tmp_path):
    from coding_agent.core.skills import load_skills_from_dir
    # No SKILL.md — should load recursively
    (tmp_path / "top-skill.md").write_text("---\nname: top-skill\ndescription: test\n---\ncontent")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "sub-skill.md").write_text("---\nname: sub-skill\ndescription: test\n---\ncontent")

    skills = load_skills_from_dir(str(tmp_path))
    names = [s.name for s in skills]
    assert "top-skill" in names
    assert "sub-skill" in names


def test_load_skills_from_dir_empty(tmp_path):
    from coding_agent.core.skills import load_skills_from_dir
    skills = load_skills_from_dir(str(tmp_path))
    assert skills == []


def test_load_skills_from_dir_nonexistent():
    from coding_agent.core.skills import load_skills_from_dir
    skills = load_skills_from_dir("/nonexistent/path/xyz")
    assert skills == []


def test_load_skills_respects_nested_skill_md(tmp_path):
    """A file deeply nested inside a subdirectory that has SKILL.md should be skipped.

    The ancestor check in the implementation covers directories *between* root and the file,
    which requires a depth of at least 3 (root → sub → deep → file.md).
    """
    from coding_agent.core.skills import load_skills_from_dir
    # Root has NO SKILL.md (recursive mode)
    (tmp_path / "root-skill.md").write_text("---\nname: root-skill\ndescription: test\n---\ncontent")
    # sub directory has SKILL.md
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "SKILL.md").write_text("# Sub Index")
    # deep directory under sub — for this file, sub is an ancestor checked by the loop
    deep = sub / "deep"
    deep.mkdir()
    (deep / "deep-skill.md").write_text("---\nname: deep-skill\ndescription: test\n---\ncontent")

    skills = load_skills_from_dir(str(tmp_path))
    names = [s.name for s in skills]
    assert "root-skill" in names
    # deep-skill should be skipped: its ancestor 'sub' has SKILL.md
    assert "deep-skill" not in names


def test_load_skill_commands_without_leading_slash(tmp_path):
    from coding_agent.core.skills import load_skill
    f = tmp_path / "cmd-skill.md"
    f.write_text("---\nname: cmd-skill\ndescription: test\ncommands: [/cmd-skill, /alias]\n---\ncontent")
    skill = load_skill(str(f))
    assert skill is not None
    # The parser strips leading /
    assert "cmd-skill" in skill.commands
    assert "alias" in skill.commands
