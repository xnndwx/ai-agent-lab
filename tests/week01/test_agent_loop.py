"""ReAct 循环测试。模型回复全部由脚本客户端提供，不连接 Ollama。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from week01.agent_loop import AgentResult, build_system_prompt, run_agent
from week01.main import main
from week01.tools import TOOL_SCHEMAS


class ScriptedClient:
    """按给定顺序返回响应，并记下每次收到的消息。"""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        # 待返回的响应队列
        self.replies = list(replies)
        # 每次 chat 的入参
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """弹出下一条脚本响应。用尽时立刻失败，避免测试假绿。"""
        # 复制当时的消息，避免循环继续追加后，历史记录被同一份列表带偏
        snapshot = [dict(item) for item in messages]
        self.calls.append({"messages": snapshot, "tools": tools})
        if not self.replies:
            raise AssertionError("脚本回复已用完")
        return self.replies.pop(0)


def test_system_prompt_lists_tools() -> None:
    """系统提示包含全部工具名，并且保留文本协议示例。"""
    prompt = build_system_prompt()
    assert "echo" in prompt
    assert "add" in prompt
    assert "now" in prompt
    assert "Final Answer:" in prompt
    assert "Action Input:" in prompt


def test_plain_final_answer_stops() -> None:
    """没有工具调用时，Final Answer 就是结果，且只请求一次模型。"""
    client = ScriptedClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Thought: 可以直接回答\nFinal Answer: 4\n补充",
                }
            }
        ]
    )
    result = run_agent("1+3 是多少？", client)
    assert result.answer == "4\n补充"
    assert result.stop_reason == "final_answer"
    assert result.steps[0].thought == "可以直接回答"
    assert result.steps[0].actions == []
    assert client.calls[0]["tools"] == TOOL_SCHEMAS
    # 首轮消息包含系统提示和用户问题
    roles = [item["role"] for item in client.calls[0]["messages"]]
    assert roles == ["system", "user"]


def test_plain_text_without_protocol_is_answer() -> None:
    """3B 模型若直接说话、不写协议，也把正文当作最终回答。"""
    client = ScriptedClient([{"message": {"content": "你好，我在。"}}])
    result = run_agent("在吗", client)
    assert result.answer == "你好，我在。"
    assert result.stop_reason == "final_answer"


def test_text_action_then_final_answer() -> None:
    """文本 Action 执行后，观察结果会进入下一轮，再取出最终回答。"""
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": 'Thought: 交给工具\nAction: add\nAction Input: {"a": 19, "b": 23}'
                }
            },
            {"message": {"content": "Thought: 已经拿到和\nFinal Answer: 和是 42"}},
        ]
    )
    result = run_agent("19 加 23", client)
    assert result.answer == "和是 42"
    assert result.steps[0].actions == ["add"]
    assert result.steps[0].observations == ["42"]
    # 第二轮能看见观察，而不是再猜一遍
    followup = client.calls[1]["messages"][-1]
    assert followup["role"] == "user"
    assert followup["content"] == "Observation: 42"


def test_action_beats_final_answer_in_same_message() -> None:
    """同一条消息里如果写了 Action，就先执行工具，不采纳猜出来的答案。"""
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": (
                        'Thought: 先算\nAction: add\nAction Input: {"a": 1, "b": 2}\n'
                        "Final Answer: 100"
                    )
                }
            },
            {"message": {"content": "Final Answer: 3"}},
        ]
    )
    result = run_agent("1+2", client)
    assert result.answer == "3"
    assert result.steps[0].observations == ["3"]


def test_bad_action_input_is_observable() -> None:
    """非法 JSON 成为观察，模型可以在下一轮改正。"""
    client = ScriptedClient(
        [
            {"message": {"content": "Action: echo\nAction Input: {bad}"}},
            {"message": {"content": "Final Answer: 参数写错了"}},
        ]
    )
    result = run_agent("回显", client)
    assert "合法 JSON" in result.steps[0].observations[0]
    assert result.answer == "参数写错了"


def test_unknown_text_tool_does_not_crash() -> None:
    """未知工具把错误观察交回模型。"""
    client = ScriptedClient(
        [
            {"message": {"content": "Action: rm\nAction Input: {}"}},
            {"message": {"content": "Final Answer: 没有这个工具"}},
        ]
    )
    result = run_agent("删掉文件", client)
    assert "未知工具" in result.steps[0].observations[0]
    assert result.stop_reason == "final_answer"


def test_native_tool_calls() -> None:
    """Ollama 原生 tool_calls 会执行，并用 role=tool 写回。"""
    client = ScriptedClient(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "echo", "arguments": {"text": "pong"}}},
                        {"function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}},
                    ],
                }
            },
            {"message": {"content": "Final Answer: pong / 4"}},
        ]
    )
    result = run_agent("请回显并相加", client)
    assert result.steps[0].actions == ["echo", "add"]
    assert result.steps[0].observations == ["pong", "4"]
    assert result.answer == "pong / 4"
    tool_messages = [item for item in client.calls[1]["messages"] if item["role"] == "tool"]
    assert [item["content"] for item in tool_messages] == ["pong", "4"]
    assert [item["tool_name"] for item in tool_messages] == ["echo", "add"]


def test_max_steps_stops_tool_loop() -> None:
    """模型一直调用工具时，到达上限就停止。"""
    reply = {"message": {"content": 'Action: echo\nAction Input: {"text": "loop"}'}}
    client = ScriptedClient([reply, dict(reply)])
    result = run_agent("循环", client, max_steps=2)
    assert result.stop_reason == "max_steps"
    assert "最大步数" in result.answer
    assert len(result.steps) == 2
    assert len(client.calls) == 2


def test_empty_model_content() -> None:
    """空回复给出明确说明，而不是空字符串。"""
    client = ScriptedClient([{"message": {"content": "   "}}])
    result = run_agent("你好", client)
    assert result.answer == "模型没有给出回答。"


def test_missing_message_raises() -> None:
    """客户端没有返回 message 对象时，循环拒绝继续。"""
    client = ScriptedClient([{"done": True}])
    with pytest.raises(ValueError, match="message"):
        run_agent("你好", client)


def test_max_steps_must_be_positive() -> None:
    """步数下限在发请求前检查。"""
    client = ScriptedClient([])
    with pytest.raises(ValueError, match="max_steps"):
        run_agent("你好", client, max_steps=0)
    assert client.calls == []


def test_main_prints_answer_and_loads_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """命令行读取 .env、打印回答，并且不覆盖已有环境变量。"""
    # 临时环境文件，只含本周允许的两个变量
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# 注释应被忽略",
                "export OLLAMA_HOST=http://env-host:11434",
                'OLLAMA_MODEL="demo:3b"',
                "OTHER=no",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    # 已存在的模型名必须保留
    monkeypatch.setenv("OLLAMA_MODEL", "keep:3b")
    # 记录客户端初始化时能看见的环境
    seen: dict[str, str | None] = {}

    class _CaptureClient:
        """替身客户端，只读取环境变量。"""

        def __init__(self) -> None:
            seen["host"] = os.environ.get("OLLAMA_HOST")
            seen["model"] = os.environ.get("OLLAMA_MODEL")

    def _fake_run(question: str, client: object, max_steps: int = 6) -> AgentResult:
        assert question == "你好 世界"
        assert max_steps == 3
        assert isinstance(client, _CaptureClient)
        return AgentResult(answer="好的", steps=[], stop_reason="final_answer")

    monkeypatch.setattr("week01.main.OllamaClient", _CaptureClient)
    monkeypatch.setattr("week01.main.run_agent", _fake_run)
    code = main(["你好", "世界", "--max-steps", "3", "--env-file", str(env_file)])
    assert code == 0
    assert capsys.readouterr().out.strip() == "好的"
    assert seen["host"] == "http://env-host:11434"
    assert seen["model"] == "keep:3b"


def test_main_empty_question_and_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """空问题或运行错误返回 1，且不打印假回答。"""
    monkeypatch.setattr("builtins.input", lambda _prompt: "  ")
    assert main([]) == 1
    assert "请提供一个问题" in capsys.readouterr().err

    def _boom(question: str, client: object, max_steps: int = 6) -> AgentResult:
        raise ValueError("步数非法")

    monkeypatch.setattr("week01.main.OllamaClient", lambda: object())
    monkeypatch.setattr("week01.main.run_agent", _boom)
    assert main(["还有问题吗"]) == 1
    assert "运行失败" in capsys.readouterr().err
