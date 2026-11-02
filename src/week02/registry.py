"""工具注册表（类似 Java 的服务注册 + Bean 查找）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from week02 import observation as obs


class TruncateError(Exception):
    """内容过长，需要 truncated 观察。"""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


ToolFunc = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的名称、说明、参数 schema 与实现。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc


class ToolRegistry:
    """按名字注册 / 查找 / 执行工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """注册工具；同名会覆盖。"""
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        """已注册工具名，排序后返回。"""
        return sorted(self._tools)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Ollama / OpenAI 风格的 function tools 列表。"""
        schemas: list[dict[str, Any]] = []
        for name in self.names():
            spec = self._tools[name]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
            )
        return schemas

    def run(self, name: str, arguments: Any) -> str:
        """执行工具；失败返回结构化 Observation，不向外抛。"""
        import json

        if name not in self._tools:
            return obs.error_unknown_tool(name, self.names())
        try:
            parsed = _coerce_arguments(arguments)
        except json.JSONDecodeError:
            return obs.error_bad_args(name, "工具参数不是合法 JSON")
        except ValueError as exc:
            return obs.error_bad_args(name, str(exc))
        try:
            result = self._tools[name].func(parsed)
        except ValueError as exc:
            return obs.error_bad_args(name, str(exc))
        except TruncateError as exc:
            return obs.error_truncated(name, exc.detail)
        except Exception as exc:  # noqa: BLE001 — 观察给模型看
            return obs.error_generic("runtime", f"{name}: {exc}")
        return obs.ok(result)


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """把缺失值、字典或 JSON 字符串统一成字典。"""
    import json

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
