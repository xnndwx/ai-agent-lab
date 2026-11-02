"""week02 注册表与工具测试；不访问网络。"""

from __future__ import annotations

from pathlib import Path

import pytest

from week02 import observation as obs
from week02.registry import ToolRegistry, ToolSpec, TruncateError
from week02.tools import build_default_registry, calc, length


def test_observation_prefixes() -> None:
    assert obs.ok("42").startswith("[ok]")
    assert "bad_args" in obs.error_bad_args("calc", "缺参数")
    assert "unknown_tool" in obs.error_unknown_tool("foo", ["calc"])
    assert "truncated" in obs.error_truncated("read_note", "太长")


def test_registry_schemas_and_length() -> None:
    registry = build_default_registry()
    names = registry.names()
    assert "length" in names
    assert "calc" in names
    assert "list_notes" in names
    assert "read_note" in names
    assert "code_explain" in names
    schemas = registry.get_schemas()
    schema_names = [item["function"]["name"] for item in schemas]
    assert schema_names == names
    assert registry.run("length", {"text": "你好"}) == "[ok] 2"


def test_calc_safe_ast() -> None:
    assert calc({"expression": "(3+5)*2"}) == "16"
    assert calc({"expression": "19+23"}) == "42"
    with pytest.raises(ValueError):
        calc({"expression": "__import__('os')"})
    message = build_default_registry().run("calc", {"expression": "1+"})
    assert message.startswith("[error:bad_args]")


def test_unknown_tool_and_bad_json() -> None:
    registry = build_default_registry()
    unknown = registry.run("rm", {})
    assert unknown.startswith("[error:unknown_tool]")
    assert "calc" in unknown
    bad = registry.run("length", "{")
    assert bad.startswith("[error:bad_args]")


def test_notes_whitelist(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "hello.txt").write_text("hello agent", encoding="utf-8")
    registry = build_default_registry(notes_dir=notes)
    listed = registry.run("list_notes", {})
    assert listed.startswith("[ok]")
    assert "hello.txt" in listed
    content = registry.run("read_note", {"name": "hello.txt"})
    assert content == "[ok] hello agent"
    traverse = registry.run("read_note", {"name": "../secret.txt"})
    assert traverse.startswith("[error:bad_args]")
    missing = registry.run("read_note", {"name": "nope.txt"})
    assert missing.startswith("[error:bad_args]")


def test_read_note_truncate(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "big.txt").write_text("x" * 600, encoding="utf-8")
    registry = build_default_registry(notes_dir=notes)
    message = registry.run("read_note", {"name": "big.txt"})
    assert message.startswith("[error:truncated]")


def test_register_new_tool_without_touching_loop() -> None:
    """新增工具只改注册表。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="double",
            description="把整数翻倍",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            func=lambda arguments: str(int(arguments["n"]) * 2),
        )
    )
    assert registry.run("double", {"n": 21}) == "[ok] 42"
    assert registry.get_schemas()[0]["function"]["name"] == "double"


def test_truncate_error_helper() -> None:
    registry = ToolRegistry()

    def boom(_: dict) -> str:
        raise TruncateError("已截断")

    registry.register(
        ToolSpec(
            name="boom",
            description="截断",
            parameters={"type": "object", "properties": {}},
            func=boom,
        )
    )
    assert registry.run("boom", {}).startswith("[error:truncated]")


def test_length_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        length({"text": 1})
