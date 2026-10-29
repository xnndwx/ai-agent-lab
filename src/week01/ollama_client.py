"""本地 Ollama 的最小聊天客户端。

只封装 `POST /api/chat`。第 1 周不引入 LangChain / LangGraph。
单元测试可以注入 transport，因此不需要真正启动 Ollama。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

# 未配置环境变量时使用的本机地址
DEFAULT_HOST = "http://127.0.0.1:11434"
# 面向个人电脑的约 3B 模型，可在 .env 中改成 llama3.2:3b 等
DEFAULT_MODEL = "qwen2.5:3b"
# 单次对话的超时秒数，避免学习时请求一直挂起
DEFAULT_TIMEOUT_SECONDS = 120.0

# 可替换的发送函数：入参是 URL 与 JSON 对象，返回已解析的响应字典
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class OllamaError(RuntimeError):
    """调用 Ollama 失败时抛出，便于命令行打印原因。"""


class OllamaClient:
    """把消息发给本地 Ollama，并返回完整 JSON 响应。"""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Transport | None = None,
    ) -> None:
        # 显式参数优先，其次环境变量，最后使用默认值
        raw_host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        raw_model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        # 去掉末尾斜杠，避免拼出 //api/chat
        self.host = _normalize_host(raw_host)
        # 模型名去掉空白，空字符串视为非法
        self.model = raw_model.strip()
        # 传给传输层的超时
        self.timeout = timeout
        # 测试注入的假传输；为 None 时走 urllib
        self.transport = transport
        if not self.model:
            raise OllamaError("OLLAMA_MODEL 不能为空")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用 /api/chat。返回值至少包含 message 字段。"""
        # 关闭流式输出，初学者可以一次读完整段回复
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            # 只有确实提供工具时才带上，避免给模型空列表
            payload["tools"] = tools
        # 聊天接口的完整地址
        url = f"{self.host}/api/chat"
        # 生产环境用 urllib；测试可换成假函数
        send = self.transport or _urllib_transport(self.timeout)
        try:
            response = send(url, payload)
        except OllamaError:
            raise
        except Exception as exc:
            # 传输层抛出的其它异常统一包成 OllamaError
            raise OllamaError(f"请求 Ollama 失败：{exc}") from exc
        if not isinstance(response, dict) or "message" not in response:
            raise OllamaError("Ollama 响应缺少 message 字段")
        return response


def _normalize_host(host: str) -> str:
    """整理主机地址：去空白、去末尾斜杠。"""
    cleaned = host.strip().rstrip("/")
    if not cleaned:
        raise OllamaError("OLLAMA_HOST 不能为空")
    return cleaned


def _urllib_transport(timeout: float) -> Transport:
    """返回一个用标准库 urllib 发送 JSON 的函数。"""

    def send(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 请求体使用 UTF-8，保证中文问题不会乱码
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # 标准库的 POST 请求对象
        request = urllib.request.Request(
            url,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                # 响应正文
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # HTTP 错误页的文本，截断后放进异常信息
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OllamaError(f"Ollama HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"无法连接 Ollama（{url}）：{exc.reason}") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama 返回的不是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Ollama 返回的 JSON 不是对象")
        return parsed

    return send
