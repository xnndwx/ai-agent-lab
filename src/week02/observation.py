"""结构化 Observation 字符串，方便模型改正错误。"""

from __future__ import annotations


def ok(text: str) -> str:
    """成功观察。"""
    return f"[ok] {text}"


def error_bad_args(tool: str, detail: str) -> str:
    """参数错误。"""
    return f"[error:bad_args] {tool} → {detail}（请改参数后重试）"


def error_unknown_tool(name: str, known: list[str]) -> str:
    """未知工具。"""
    listed = ", ".join(known) if known else "(无)"
    return f"[error:unknown_tool] {name} → 可用工具：{listed}"


def error_truncated(tool: str, detail: str) -> str:
    """内容被截断。"""
    return f"[error:truncated] {tool} → {detail}"


def error_generic(code: str, detail: str) -> str:
    """其它错误。"""
    return f"[error:{code}] {detail}"
