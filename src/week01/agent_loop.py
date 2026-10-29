"""最小 ReAct 循环。

每一轮做三件事：
1. 把目前的对话发给模型（推理，Thought）
2. 如果模型给出工具调用或 Action，就执行工具（行动，Action）
3. 把工具结果写回对话（观察，Observation）

直到模型给出最终回答，或达到最大步数。
第 1 周不使用 LangChain / LangGraph，这个文件就是循环本身。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from week01.tools import TOOL_SCHEMAS, run_tool

# 防止约 3B 的模型反复调用工具
DEFAULT_MAX_STEPS = 6

# 只匹配行首标记，避免正文里偶然出现的单词被当成协议
_THOUGHT_RE = re.compile(r"(?m)^Thought:\s*(.*)$")
_ACTION_RE = re.compile(r"(?m)^Action:\s*(.*)$")
_INPUT_RE = re.compile(r"(?m)^Action Input:\s*(.*)$")
_FINAL_RE = re.compile(r"(?m)^Final Answer:\s*(.*)$")

# 同时说明原生函数调用和纯文本协议。{tool_list} 由代码替换，避免 format 误伤 JSON 花括号。
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
可用工具：
{tool_list}
"""


class ChatClient(Protocol):
    """任何实现了 chat 的对象都能驱动循环，测试里可以换成脚本客户端。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """返回至少包含 message 的响应字典。"""


@dataclass
class StepRecord:
    """一轮模型调用的轨迹，方便测试和以后打印调试日志。"""

    # 模型原文；只做原生工具调用时可能是空字符串
    raw: str
    # Thought 行的内容；没有则为空字符串
    thought: str
    # 本轮执行过的工具名，顺序与 observations 一致
    actions: list[str] = field(default_factory=list)
    # 与 actions 一一对应的工具结果
    observations: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """一次用户提问的完整结果。"""

    # 最终展示给用户的文本
    answer: str
    # 从第一轮到停止的轨迹
    steps: list[StepRecord]
    # final_answer：模型给出了回答；max_steps：步数用尽
    stop_reason: str


def build_system_prompt() -> str:
    """把已注册工具的名字和说明填进系统提示。"""
    # 每个工具一行，避免提示词和 schema 各写一套描述
    lines = [
        f"- {schema['function']['name']}: {schema['function']['description']}"
        for schema in TOOL_SCHEMAS
    ]
    tool_list = "\n".join(lines)
    return SYSTEM_PROMPT.replace("{tool_list}", tool_list)


def run_agent(
    user_text: str,
    client: ChatClient,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """运行最小 ReAct 循环。"""
    if max_steps < 1:
        raise ValueError("max_steps 必须 >= 1")
    # 对话历史：系统提示在最前，用户问题紧随其后
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_text},
    ]
    # 每轮都带上工具清单，支持原生函数调用的模型可以直接用
    tools = TOOL_SCHEMAS
    # 已完成的轮次
    steps: list[StepRecord] = []

    for _ in range(max_steps):
        response = client.chat(messages, tools=tools)
        # 写回历史的助手消息（含可能的 tool_calls）
        message = _assistant_message(response)
        messages.append(message)
        # 模型正文，原生工具调用时允许为空
        raw = str(message.get("content") or "")
        thought = _extract_thought(raw)
        native_calls = _native_tool_calls(message)
        step = StepRecord(raw=raw, thought=thought)

        if native_calls:
            _apply_native_calls(messages, step, native_calls)
            steps.append(step)
            continue

        # 文本协议里只要出现 Action，就先执行，不把同一条消息当成最终回答
        action_name = _extract_action(raw)
        if action_name:
            observation = _run_text_action(raw, action_name)
            step.actions.append(action_name)
            step.observations.append(observation)
            steps.append(step)
            # 3B 模型对 role=tool 支持不稳定，文本协议改用用户消息送达观察
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        answer = _final_or_plain_text(raw)
        steps.append(step)
        return AgentResult(answer=answer, steps=steps, stop_reason="final_answer")

    return AgentResult(
        answer="已达到最大步数，停止继续调用工具。",
        steps=steps,
        stop_reason="max_steps",
    )


def _apply_native_calls(
    messages: list[dict[str, Any]],
    step: StepRecord,
    native_calls: list[tuple[str, Any]],
) -> None:
    """执行原生 tool_calls，并把每条结果按 Ollama 习惯追加为 role=tool。"""
    for name, arguments in native_calls:
        observation = run_tool(name, arguments)
        step.actions.append(name)
        step.observations.append(observation)
        messages.append({"role": "tool", "content": observation, "tool_name": name})


def _run_text_action(raw: str, action_name: str) -> str:
    """解析文本协议中的 Action Input 并执行工具。"""
    action_input = _extract_action_input(raw)
    if isinstance(action_input, str):
        # 字符串表示解析失败，直接作为观察交回模型
        return action_input
    return run_tool(action_name, action_input)


def _final_or_plain_text(raw: str) -> str:
    """没有动作时：优先使用 Final Answer，否则把整段正文当作回答。"""
    final_answer = _extract_final_answer(raw)
    # None 表示没有 Final Answer 标记
    answer = raw.strip() if final_answer is None else final_answer
    if not answer:
        return "模型没有给出回答。"
    return answer


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """从客户端响应中取出 assistant message，并保证带有 role。"""
    message = response.get("message")
    if not isinstance(message, dict):
        raise ValueError("客户端响应缺少 message 对象")
    # 拷贝一份，避免调用方随后修改影响已保存的历史
    copied = dict(message)
    copied.setdefault("role", "assistant")
    return copied


def _native_tool_calls(message: dict[str, Any]) -> list[tuple[str, Any]]:
    """读取 Ollama 风格的 tool_calls。没有时返回空列表。"""
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    # (工具名, 参数) 列表，参数保持原样交给 run_tool
    calls: list[tuple[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        # 兼容直接把 name 放在 tool_call 上的写法
        payload = function if isinstance(function, dict) else item
        name = payload.get("name")
        if not name:
            continue
        calls.append((str(name), payload.get("arguments")))
    return calls


def _extract_thought(text: str) -> str:
    """取出第一条 Thought 的内容。"""
    matched = _THOUGHT_RE.search(text)
    if not matched:
        return ""
    return matched.group(1).strip()


def _extract_action(text: str) -> str | None:
    """取出 Action 后的工具名。没有或为空时返回 None。"""
    matched = _ACTION_RE.search(text)
    if not matched:
        return None
    name = matched.group(1).strip()
    return name or None


def _extract_action_input(text: str) -> dict[str, Any] | str:
    """解析单行 Action Input。成功返回字典，失败返回错误说明。"""
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
    """取出 Final Answer 标记之后的正文，允许回答跨行。"""
    matched = _FINAL_RE.search(text)
    if not matched:
        return None
    # 标记同一行上的文字
    same_line = matched.group(1).strip()
    # 标记所在行之后的剩余正文
    trailing = text[matched.end() :].strip()
    if same_line and trailing:
        return f"{same_line}\n{trailing}"
    return same_line or trailing
