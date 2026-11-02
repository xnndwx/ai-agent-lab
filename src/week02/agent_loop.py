"""第 2 周 ReAct 循环：注册表多工具 + 结构化 Observation + 工具错误重试预算。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from week02.registry import ToolRegistry
from week02.tools import build_default_registry

DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_TOOL_RETRIES = 2

_THOUGHT_RE = re.compile(r"(?m)^Thought:\s*(.*)$")
_ACTION_RE = re.compile(r"(?m)^Action:\s*(.*)$")
_INPUT_RE = re.compile(r"(?m)^Action Input:\s*(.*)$")
_FINAL_RE = re.compile(r"(?m)^Final Answer:\s*(.*)$")

SYSTEM_PROMPT = """你是一个正在学习中的本地 Agent，运行在约 3B 参数的模型上。
回答问题时可以先调用工具。请优先使用函数调用（tool call）。
如果当前模型不支持函数调用，就严格使用下面的纯文本格式，每次只做一件事。

需要工具时：
Thought: 你为什么需要工具
Action: 工具名
Action Input: {"参数": "值"}

可以直接回答时：
Thought: 你的简短理由
Final Answer: 给用户的最终回答

同一轮里如果写了 Action，就不要写 Final Answer。
不要编造 Observation。工具返回后，你会看到以 Observation: 开头的消息。
若 Observation 以 [error:bad_args] 或 [error:unknown_tool] 开头，请改正参数或换工具，不要立刻 Final Answer。
可用工具：
{tool_list}
"""


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class StepRecord:
    raw: str
    thought: str
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[StepRecord]
    stop_reason: str
    messages: list[dict[str, Any]] = field(default_factory=list)


def build_system_prompt(registry: ToolRegistry) -> str:
    lines = [
        f"- {schema['function']['name']}: {schema['function']['description']}"
        for schema in registry.get_schemas()
    ]
    return SYSTEM_PROMPT.replace("{tool_list}", "\n".join(lines))


def run_agent(
    user_text: str,
    client: ChatClient,
    *,
    registry: ToolRegistry | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES,
    prior_messages: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """运行 ReAct 循环；可注入已有 messages（会话恢复时不含最后一轮用户话可另拼）。"""
    if max_steps < 1:
        raise ValueError("max_steps 必须 >= 1")
    tools_registry = registry or build_default_registry()
    schemas = tools_registry.get_schemas()

    if prior_messages:
        messages = [dict(item) for item in prior_messages]
        # 确保有系统提示
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": build_system_prompt(tools_registry)})
        messages.append({"role": "user", "content": user_text})
    else:
        messages = [
            {"role": "system", "content": build_system_prompt(tools_registry)},
            {"role": "user", "content": user_text},
        ]

    steps: list[StepRecord] = []
    tool_retries = 0

    for _ in range(max_steps):
        response = client.chat(messages, tools=schemas)
        message = _assistant_message(response)
        messages.append(message)
        raw = str(message.get("content") or "")
        thought = _extract_thought(raw)
        native_calls = _native_tool_calls(message)
        step = StepRecord(raw=raw, thought=thought)

        if native_calls:
            recoverable = _apply_native_calls(messages, step, native_calls, tools_registry)
            steps.append(step)
            if recoverable:
                tool_retries += 1
                if tool_retries > max_tool_retries:
                    return AgentResult(
                        answer="工具参数多次失败，已停止。",
                        steps=steps,
                        stop_reason="tool_retries",
                        messages=messages,
                    )
            else:
                tool_retries = 0
            continue

        action_name = _extract_action(raw)
        if action_name:
            observation = _run_text_action(raw, action_name, tools_registry)
            step.actions.append(action_name)
            step.observations.append(observation)
            steps.append(step)
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            if observation.startswith("[error:bad_args]") or observation.startswith(
                "[error:unknown_tool]"
            ):
                tool_retries += 1
                if tool_retries > max_tool_retries:
                    return AgentResult(
                        answer="工具参数多次失败，已停止。",
                        steps=steps,
                        stop_reason="tool_retries",
                        messages=messages,
                    )
            else:
                tool_retries = 0
            continue

        answer = _final_or_plain_text(raw)
        steps.append(step)
        return AgentResult(
            answer=answer,
            steps=steps,
            stop_reason="final_answer",
            messages=messages,
        )

    return AgentResult(
        answer="已达到最大步数，停止继续调用工具。",
        steps=steps,
        stop_reason="max_steps",
        messages=messages,
    )


def _apply_native_calls(
    messages: list[dict[str, Any]],
    step: StepRecord,
    native_calls: list[tuple[str, Any]],
    registry: ToolRegistry,
) -> bool:
    """执行原生 tool_calls。若任一观察为可恢复错误，返回 True。"""
    recoverable = False
    for name, arguments in native_calls:
        observation = registry.run(name, arguments)
        step.actions.append(name)
        step.observations.append(observation)
        messages.append({"role": "tool", "content": observation, "tool_name": name})
        if observation.startswith("[error:bad_args]") or observation.startswith(
            "[error:unknown_tool]"
        ):
            recoverable = True
    return recoverable


def _run_text_action(raw: str, action_name: str, registry: ToolRegistry) -> str:
    action_input = _extract_action_input(raw)
    if isinstance(action_input, str):
        # 解析失败：伪装成 bad_args 观察
        from week02 import observation as obs

        return obs.error_bad_args(action_name, action_input)
    return registry.run(action_name, action_input)


def _final_or_plain_text(raw: str) -> str:
    final_answer = _extract_final_answer(raw)
    answer = raw.strip() if final_answer is None else final_answer
    if not answer:
        return "模型没有给出回答。"
    return answer


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    message = response.get("message")
    if not isinstance(message, dict):
        raise ValueError("客户端响应缺少 message 对象")
    copied = dict(message)
    copied.setdefault("role", "assistant")
    return copied


def _native_tool_calls(message: dict[str, Any]) -> list[tuple[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    calls: list[tuple[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        payload = function if isinstance(function, dict) else item
        name = payload.get("name")
        if not name:
            continue
        calls.append((str(name), payload.get("arguments")))
    return calls


def _extract_thought(text: str) -> str:
    matched = _THOUGHT_RE.search(text)
    return matched.group(1).strip() if matched else ""


def _extract_action(text: str) -> str | None:
    matched = _ACTION_RE.search(text)
    if not matched:
        return None
    name = matched.group(1).strip()
    return name or None


def _extract_action_input(text: str) -> dict[str, Any] | str:
    matched = _INPUT_RE.search(text)
    if not matched or not matched.group(1).strip():
        return {}
    blob = matched.group(1).strip()
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        return f"Action Input 不是合法 JSON：{blob}"
    if not isinstance(loaded, dict):
        return "Action Input 必须是 JSON 对象"
    return loaded


def _extract_final_answer(text: str) -> str | None:
    matched = _FINAL_RE.search(text)
    if not matched:
        return None
    same_line = matched.group(1).strip()
    trailing = text[matched.end() :].strip()
    if same_line and trailing:
        return f"{same_line}\n{trailing}"
    return same_line or trailing
