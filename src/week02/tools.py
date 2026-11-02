"""第 2 周工具实现：注册进 ToolRegistry。"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from week02.registry import ToolRegistry, ToolSpec, TruncateError

# 默认笔记目录：仓库根下的 notes/
DEFAULT_NOTES_DIR = Path(__file__).resolve().parents[2] / "notes"
# 单文件读取上限，避免 Observation 过长
MAX_NOTE_CHARS = 500

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def length(arguments: dict[str, Any]) -> str:
    """返回 text 的字符数。"""
    text = arguments.get("text")
    if not isinstance(text, str):
        raise ValueError("参数 text 必须是字符串")
    return str(len(text))


def calc(arguments: dict[str, Any]) -> str:
    """安全计算四则表达式（禁止任意代码 / eval）。"""
    expr = arguments.get("expression")
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("参数 expression 必须是非空字符串")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc}") from exc
    value = _eval_ast(tree.body)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def list_notes(arguments: dict[str, Any], notes_dir: Path) -> str:
    """列出 notes/ 下的文件名。"""
    _ = arguments
    root = notes_dir.resolve()
    if not root.is_dir():
        return "(notes 目录不存在)"
    names = sorted(p.name for p in root.iterdir() if p.is_file())
    return "\n".join(names) if names else "(空)"


def read_note(arguments: dict[str, Any], notes_dir: Path) -> str:
    """读取 notes/ 下白名单文件；禁止路径穿越。"""
    name = arguments.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("参数 name 必须是非空文件名")
    root = notes_dir.resolve()
    target = _resolve_note_path(root, name.strip())
    if not target.is_file():
        raise ValueError(f"找不到笔记：{name}")
    text = target.read_text(encoding="utf-8")
    if len(text) > MAX_NOTE_CHARS:
        raise TruncateError(f"已截断到 {MAX_NOTE_CHARS} 字：\n{text[:MAX_NOTE_CHARS]}")
    return text


def code_explain(arguments: dict[str, Any]) -> str:
    """对短代码片段做极简说明（不执行代码）。"""
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("参数 code 必须是非空字符串")
    snippet = code.strip()
    lines = snippet.count("\n") + 1
    first = snippet.splitlines()[0][:80]
    return f"共 {lines} 行、{len(snippet)} 字符的代码片段（未执行）。首行：{first}"


def _resolve_note_path(root: Path, name: str) -> Path:
    if Path(name).name != name or ".." in name or "/" in name or "\\" in name:
        raise ValueError("只允许 notes/ 下的简单文件名")
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("禁止读取 notes/ 以外的路径") from exc
    return target


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_eval_ast(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_BINOPS[type(node.op)](left, right))
    raise ValueError("只允许数字与 + - * / // % **")


def build_default_registry(notes_dir: Path | None = None) -> ToolRegistry:
    """创建并注册本周默认工具。"""
    root = notes_dir if notes_dir is not None else DEFAULT_NOTES_DIR
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="length",
            description="返回一段文本的字符数。",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "要计数的文本"}},
                "required": ["text"],
            },
            func=length,
        )
    )
    registry.register(
        ToolSpec(
            name="calc",
            description="安全计算四则运算表达式，例如 19+23 或 (3+5)*2。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算术表达式"},
                },
                "required": ["expression"],
            },
            func=calc,
        )
    )
    registry.register(
        ToolSpec(
            name="list_notes",
            description="列出 notes/ 目录下的笔记文件名。",
            parameters={"type": "object", "properties": {}},
            func=lambda arguments: list_notes(arguments, root),
        )
    )
    registry.register(
        ToolSpec(
            name="read_note",
            description="读取 notes/ 下某个笔记文件的内容（仅文件名，禁止路径穿越）。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "文件名，如 hello.txt"},
                },
                "required": ["name"],
            },
            func=lambda arguments: read_note(arguments, root),
        )
    )
    registry.register(
        ToolSpec(
            name="code_explain",
            description="对短代码做极简说明，不执行代码。",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "代码文本"}},
                "required": ["code"],
            },
            func=code_explain,
        )
    )
    return registry
