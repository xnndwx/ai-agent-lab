"""第 1 周的工具桩。

这些函数不访问网络、不执行命令、也不读写任意文件。
它们只为 ReAct 循环提供可以观察的动作。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

# 工具实现的统一签名：接收参数字典，返回给模型看的字符串
ToolFunc = Callable[[dict[str, Any]], str]


def echo(arguments: dict[str, Any]) -> str:
    """原样返回 text，用来确认工具调用已经打通。"""
    # 缺省时回显空字符串，而不是抛异常
    text = arguments.get("text", "")
    return str(text)


def add(arguments: dict[str, Any]) -> str:
    """返回 a 与 b 的和。两个参数都必须是数字。"""
    # 左右加数先单独校验，错误信息能指出是哪一个
    left = _as_number(arguments.get("a"), "a")
    right = _as_number(arguments.get("b"), "b")
    # 用 float 统一整数和小数
    total = left + right
    if total.is_integer():
        # 整数值不带 .0，方便初学者阅读
        return str(int(total))
    return str(total)


def now(arguments: dict[str, Any]) -> str:
    """返回当前 UTC 时间的 ISO 字符串。参数字典仅为了签名一致。"""
    # 这个工具不读取参数
    _ = arguments
    # 去掉微秒，观察结果更短
    current = datetime.now(timezone.utc).replace(microsecond=0)
    return current.isoformat()


def _as_number(value: Any, name: str) -> float:
    """把工具参数转成 float。布尔值不算数字。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"参数 {name} 必须是数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参数 {name} 必须是数字") from exc
    return number


# 名称到实现的注册表。新增工具时同时补下面的 schema。
TOOL_FUNCS: dict[str, ToolFunc] = {
    "echo": echo,
    "add": add,
    "now": now,
}

# 发给模型的工具描述，格式对齐 Ollama / OpenAI 的 function tools。
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "原样返回一段文本，用来确认工具调用是否成功。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要回显的文本"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数的和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now",
            "description": "返回当前 UTC 时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(name: str, arguments: Any) -> str:
    """按名称执行工具。失败时返回可放进 Observation 的说明，不向外抛出。"""
    try:
        # 字典或 JSON 字符串都收成字典
        parsed = _coerce_arguments(arguments)
    except json.JSONDecodeError:
        return "工具参数不是合法 JSON"
    except ValueError as exc:
        return f"工具参数错误：{exc}"
    # 按名称查找实现
    func = TOOL_FUNCS.get(name)
    if func is None:
        known = ", ".join(sorted(TOOL_FUNCS))
        return f"未知工具：{name}。可用工具：{known}"
    try:
        return func(parsed)
    except ValueError as exc:
        return f"工具 {name} 参数错误：{exc}"


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """把缺失值、字典或 JSON 字符串统一成字典。"""
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        loaded = json.loads(arguments)
        if not isinstance(loaded, dict):
            raise ValueError("工具参数 JSON 必须是对象")
        return loaded
    raise ValueError("工具参数必须是对象")
