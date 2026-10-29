"""Ollama 客户端测试。全部使用假传输，不连接真实服务。"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from week01.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaClient,
    OllamaError,
)


def test_defaults_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有环境变量时使用本机地址和 3B 默认模型。"""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    # 不发请求，只检查配置
    client = OllamaClient(transport=lambda url, payload: {"message": {}})
    assert client.host == DEFAULT_HOST
    assert client.model == DEFAULT_MODEL


def test_env_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量生效，但构造参数优先于环境变量。"""
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    # 只看环境变量
    from_env = OllamaClient()
    assert from_env.host == "http://env-host:11434"
    assert from_env.model == "llama3.2:3b"
    # 显式参数覆盖环境变量，并去掉末尾斜杠
    explicit = OllamaClient(host="http://explicit:9/", model="demo:3b")
    assert explicit.host == "http://explicit:9"
    assert explicit.model == "demo:3b"


def test_empty_host_or_model_rejected() -> None:
    """空地址或空模型名要在发请求前失败。"""
    with pytest.raises(OllamaError, match="OLLAMA_HOST"):
        OllamaClient(host="   ", model="qwen2.5:3b")
    with pytest.raises(OllamaError, match="OLLAMA_MODEL"):
        OllamaClient(host="http://127.0.0.1:11434", model="  ")


def test_chat_uses_injected_transport() -> None:
    """chat 应把模型、消息和工具交给传输层，并检查 message 字段。"""
    # 记录最后一次请求，供断言使用
    captured: dict[str, object] = {}

    def transport(url: str, payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"role": "assistant", "content": "pong"}, "done": True}

    client = OllamaClient(host="http://127.0.0.1:11434", model="qwen2.5:3b", transport=transport)
    # 一条用户消息和一个工具清单
    messages = [{"role": "user", "content": "ping"}]
    tools = [{"type": "function", "function": {"name": "echo"}}]
    result = client.chat(messages, tools=tools)
    assert result["message"]["content"] == "pong"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen2.5:3b"
    assert payload["stream"] is False
    assert payload["messages"] == messages
    assert payload["tools"] == tools


def test_chat_rejects_bad_payload() -> None:
    """传输层返回了无法使用的内容时要抛出 OllamaError。"""
    client = OllamaClient(transport=lambda url, payload: {"done": True})
    with pytest.raises(OllamaError, match="message"):
        client.chat([{"role": "user", "content": "hi"}])
    broken = OllamaClient(transport=lambda url, payload: ["not-a-dict"])  # type: ignore[arg-type]
    with pytest.raises(OllamaError, match="message"):
        broken.chat([{"role": "user", "content": "hi"}])


def test_transport_exception_is_wrapped() -> None:
    """传输层的普通异常会被包成 OllamaError。"""

    def transport(url: str, payload: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("slow")

    client = OllamaClient(transport=transport)
    with pytest.raises(OllamaError, match="slow"):
        client.chat([{"role": "user", "content": "hi"}])


def test_urllib_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """标准库路径会 POST JSON，并解析响应。"""
    # 捕获请求，确认没有访问网络
    captured: dict[str, object] = {}

    class _FakeResponse:
        """模拟 urlopen 的上下文管理器。"""

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"message":{"role":"assistant","content":"hi"},"done":true}'

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("week01.ollama_client.urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(host="http://127.0.0.1:11434", model="qwen2.5:3b", timeout=3)
    result = client.chat([{"role": "user", "content": "ping"}])
    assert result["message"]["content"] == "hi"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 3
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen2.5:3b"
    assert body["stream"] is False


def test_urllib_http_and_url_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 错误和连接错误都变成 OllamaError，且不会继续抛出原始异常。"""

    def raise_http(request: urllib.request.Request, timeout: float = 0) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"boom"),
        )

    monkeypatch.setattr("week01.ollama_client.urllib.request.urlopen", raise_http)
    client = OllamaClient(host="http://127.0.0.1:9", model="qwen2.5:3b", timeout=1)
    with pytest.raises(OllamaError, match="HTTP 500"):
        client.chat([{"role": "user", "content": "x"}])

    def raise_url(request: urllib.request.Request, timeout: float = 0) -> None:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("week01.ollama_client.urllib.request.urlopen", raise_url)
    with pytest.raises(OllamaError, match="无法连接"):
        client.chat([{"role": "user", "content": "x"}])


def test_urllib_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """响应体不是 JSON 对象时失败。"""

    class _FakeResponse:
        """返回一段非法 JSON。"""

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(
        "week01.ollama_client.urllib.request.urlopen",
        lambda request, timeout=0: _FakeResponse(),
    )
    client = OllamaClient(host="http://127.0.0.1:11434", model="m", timeout=1)
    with pytest.raises(OllamaError, match="不是合法 JSON"):
        client.chat([{"role": "user", "content": "x"}])
