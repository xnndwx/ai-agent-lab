"""week02 ReAct 循环测试；模型回复全部由脚本客户端提供。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from week02.agent_loop import AgentResult, build_system_prompt, run_agent
from week02.tools import build_default_registry


class ScriptedClient:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        snapshot = [dict(item) for item in messages]
        self.calls.append({"messages": snapshot, "tools": tools})
        if not self.replies:
            raise AssertionError("脚本回复已用尽")
        return self.replies.pop(0)


def test_system_prompt_lists_registered_tools(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    prompt = build_system_prompt(registry)
    assert "calc" in prompt
    assert "read_note" in prompt
    assert "Final Answer:" in prompt


def test_text_multi_tool_chain(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "tip.txt").write_text("多工具串联", encoding="utf-8")
    registry = build_default_registry(notes_dir=notes)
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": "Thought: 先列表\nAction: list_notes\nAction Input: {}"
                }
            },
            {
                "message": {
                    "content": (
                        'Thought: 再读\nAction: read_note\n'
                        'Action Input: {"name": "tip.txt"}'
                    )
                }
            },
            {"message": {"content": "Thought: 好了\nFinal Answer: 笔记写着多工具串联"}},
        ]
    )
    result = run_agent("请列出并读取 tip.txt", client, registry=registry)
    assert result.answer == "笔记写着多工具串联"
    assert result.steps[0].actions == ["list_notes"]
    assert "tip.txt" in result.steps[0].observations[0]
    assert result.steps[1].actions == ["read_note"]
    assert "多工具串联" in result.steps[1].observations[0]
    assert result.steps[0].observations[0].startswith("[ok]")


def test_bad_args_then_recover(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": "Action: calc\nAction Input: {坏}"
                }
            },
            {
                "message": {
                    "content": 'Action: calc\nAction Input: {"expression": "2+3"}'
                }
            },
            {"message": {"content": "Final Answer: 5"}},
        ]
    )
    result = run_agent("算一下", client, registry=registry, max_tool_retries=2)
    assert result.steps[0].observations[0].startswith("[error:bad_args]")
    assert result.steps[1].observations[0] == "[ok] 5"
    assert result.answer == "5"
    assert result.stop_reason == "final_answer"


def test_tool_retries_exhausted(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    bad = {"message": {"content": "Action: calc\nAction Input: {坏}"}}
    client = ScriptedClient([bad, dict(bad), dict(bad)])
    result = run_agent("算", client, registry=registry, max_tool_retries=1)
    assert result.stop_reason == "tool_retries"
    assert "参数多次失败" in result.answer


def test_native_multi_tool_calls(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "length", "arguments": {"text": "ab"}}},
                        {
                            "function": {
                                "name": "calc",
                                "arguments": '{"expression": "10/2"}',
                            }
                        },
                    ],
                }
            },
            {"message": {"content": "Final Answer: 2 和 5"}},
        ]
    )
    result = run_agent("长度和除法", client, registry=registry)
    assert result.steps[0].actions == ["length", "calc"]
    assert result.steps[0].observations == ["[ok] 2", "[ok] 5"]
    tool_msgs = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == ["[ok] 2", "[ok] 5"]


def test_max_steps(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    reply = {
        "message": {
            "content": 'Action: length\nAction Input: {"text": "x"}'
        }
    }
    client = ScriptedClient([reply, dict(reply)])
    result = run_agent("循环", client, registry=registry, max_steps=2)
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 2


def test_prior_messages_session_resume(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    prior = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "记住了"},
    ]
    client = ScriptedClient(
        [{"message": {"content": "Final Answer: 你叫小明"}}]
    )
    result = run_agent(
        "我叫什么？",
        client,
        registry=registry,
        prior_messages=prior,
    )
    assert result.answer == "你叫小明"
    roles = [m["role"] for m in client.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert isinstance(result, AgentResult)


def test_missing_message_raises(tmp_path: Path) -> None:
    registry = build_default_registry(notes_dir=tmp_path)
    client = ScriptedClient([{"done": True}])
    with pytest.raises(ValueError, match="message"):
        run_agent("hi", client, registry=registry)
