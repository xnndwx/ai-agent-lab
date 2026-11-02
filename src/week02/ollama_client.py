"""本地 Ollama 聊天客户端（与 week01 同形，供 week02 独立导入）。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT_SECONDS = 120.0

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class OllamaError(RuntimeError):
    """调用 Ollama 失败时抛出。"""


class OllamaClient:
    """把消息发给本地 Ollama，并返回完整 JSON 响应。"""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Transport | None = None,
        http_retries: int = 1,
    ) -> None:
        raw_host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        raw_model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.host = _normalize_host(raw_host)
        self.model = raw_model.strip()
        self.timeout = timeout
        self.transport = transport
        self.http_retries = max(0, http_retries)
        if not self.model:
            raise OllamaError("OLLAMA_MODEL 不能为空")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用 /api/chat；连接失败时可按 http_retries 重试一次。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        url = f"{self.host}/api/chat"
        send = self.transport or _urllib_transport(self.timeout)
        attempts = self.http_retries + 1
        last_error: Exception | None = None
        for index in range(attempts):
            try:
                response = send(url, payload)
            except OllamaError as exc:
                last_error = exc
                if index + 1 >= attempts:
                    raise
                time.sleep(1.0)
                continue
            except Exception as exc:
                raise OllamaError(f"请求 Ollama 失败：{exc}") from exc
            if not isinstance(response, dict) or "message" not in response:
                raise OllamaError("Ollama 响应缺少 message 字段")
            return response
        assert last_error is not None
        raise last_error


def _normalize_host(host: str) -> str:
    cleaned = host.strip().rstrip("/")
    if not cleaned:
        raise OllamaError("OLLAMA_HOST 不能为空")
    return cleaned


def _urllib_transport(timeout: float) -> Transport:
    def send(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
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
