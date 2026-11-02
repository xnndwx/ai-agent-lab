"""工具桩测试。不访问网络，也不读取真实时钟以外的外部状态。"""

from __future__ import annotations

import re

from week01.tools import TOOL_FUNCS, TOOL_SCHEMAS, add, echo, now, run_tool


def test_echo_returns_text() -> None:
    """echo 原样返回 text，缺省时为空字符串。"""
    assert echo({"text": "你好"}) == "你好"
    assert echo({}) == ""
    # 非字符串也会被转成文本，方便观察
    assert run_tool("echo", {"text": 7}) == "7"


def test_add_numbers() -> None:
    """整数结果不带小数点，小数字符串也可以相加。"""
    assert add({"a": 1, "b": 2}) == "3"
    assert add({"a": "1.5", "b": 2}) == "3.5"
    assert run_tool("add", {"a": -4, "b": 1.0}) == "-3"


def test_add_rejects_non_numbers() -> None:
    """非法参数变成观察文本，而不是抛出异常。"""
    message = run_tool("add", {"a": "x", "b": 1})
    assert "参数" in message
    # 布尔值不是合法数字
    assert "参数" in run_tool("add", {"a": True, "b": 1})


def test_now_is_utc_iso() -> None:
    """当前时间是去掉微秒的 UTC ISO 字符串。"""
    text = now({})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", text)


def test_unknown_tool_lists_known_names() -> None:
    """未知工具要列出目前注册的名字。"""
    message = run_tool("delete_file", {})
    assert message.startswith("未知工具：delete_file")
    assert "echo" in message
    assert "add" in message
    assert "now" in message


def test_arguments_json_string_and_errors() -> None:
    """参数可以是 JSON 字符串；非法 JSON 返回说明。"""
    assert run_tool("echo", '{"text": "hi"}') == "hi"
    assert "合法 JSON" in run_tool("echo", "{")
    assert "对象" in run_tool("echo", "[]")
    assert "对象" in run_tool("echo", 1)


def test_schemas_match_implementations() -> None:
    """发给模型的 schema 与真实函数表一致，并且是 function 对象。"""
    schema_names = [item["function"]["name"] for item in TOOL_SCHEMAS]
    assert schema_names == ["echo", "add", "now"]
    assert set(schema_names) == set(TOOL_FUNCS)
    for item in TOOL_SCHEMAS:
        assert item["type"] == "function"
        assert item["function"]["parameters"]["type"] == "object"
