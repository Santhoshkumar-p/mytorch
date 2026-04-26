"""Tests for coding_agent.cli.file_processor."""
import asyncio
import pytest
import base64
from pathlib import Path


async def test_process_at_file(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    f = tmp_path / "prompt.txt"
    f.write_text("Do this task")
    result = await process_file_arguments([f"@{f}"], str(tmp_path))
    assert result.text is not None
    assert "Do this task" in result.text


async def test_process_plain_text(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    result = await process_file_arguments(["Hello there"], str(tmp_path))
    assert result.text == "Hello there"


async def test_process_multiple_plain_texts(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    result = await process_file_arguments(["first", "second"], str(tmp_path))
    assert result.text is not None
    assert "first" in result.text
    assert "second" in result.text


async def test_process_image_file(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    # Create a minimal valid PNG (1x1 pixel)
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    f = tmp_path / "image.png"
    f.write_bytes(png_1x1)
    result = await process_file_arguments([f"@{f}"], str(tmp_path))
    assert len(result.images) == 1
    img = result.images[0]
    # Either an ImageContent object or a dict
    if hasattr(img, "mime_type"):
        assert img.mime_type == "image/png"
    else:
        assert img["mime_type"] == "image/png"


async def test_process_nonexistent_file(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    # Nonexistent @file should be silently skipped
    result = await process_file_arguments([f"@{tmp_path}/nonexistent.txt"], str(tmp_path))
    assert result.text is None
    assert result.images == []


async def test_process_empty_text_file(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    f = tmp_path / "empty.txt"
    f.write_text("   \n   ")
    # Empty/whitespace-only files should not add to text
    result = await process_file_arguments([f"@{f}"], str(tmp_path))
    # text should be None or not include the file content
    assert result.text is None or result.text.strip() == ""


async def test_process_at_file_wraps_in_file_tag(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    f = tmp_path / "myfile.txt"
    f.write_text("content inside")
    result = await process_file_arguments([f"@{f}"], str(tmp_path))
    assert result.text is not None
    assert "content inside" in result.text
    assert "<file" in result.text


async def test_process_mixed_text_and_file(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    f = tmp_path / "task.txt"
    f.write_text("Do the task")
    result = await process_file_arguments(["Please", f"@{f}"], str(tmp_path))
    assert result.text is not None
    assert "Please" in result.text
    assert "Do the task" in result.text


async def test_process_no_args(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    result = await process_file_arguments([], str(tmp_path))
    assert result.text is None
    assert result.images == []


async def test_process_relative_file_path(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    f = tmp_path / "rel.txt"
    f.write_text("relative content")
    # Pass relative path — should resolve against cwd
    result = await process_file_arguments(["@rel.txt"], str(tmp_path))
    assert result.text is not None
    assert "relative content" in result.text


async def test_process_jpeg_image(tmp_path):
    from coding_agent.cli.file_processor import process_file_arguments
    # Minimal JPEG magic bytes
    jpeg_header = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46])
    jpeg_footer = bytes([0xFF, 0xD9])
    jpeg_data = jpeg_header + b"\x00" * 20 + jpeg_footer
    f = tmp_path / "test.jpg"
    f.write_bytes(jpeg_data)
    # Should not raise even with minimal JPEG
    result = await process_file_arguments([f"@{f}"], str(tmp_path))
    # Either image or empty — depends on PIL availability, but shouldn't crash
    assert isinstance(result.images, list)
