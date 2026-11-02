"""OllamaClient 假 transport 测试。"""

from __future__ import annotations

from typing import Any

import pytest

from week02.ollama_client import OllamaClient, OllamaError


def test_chat_success() -> None:
    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert url.endswith("/api/chat")
        assert payload["model"] == "demo:3b"
        assert payload["stream"] is False
        return {"message": {"role": "assistant", "content": "ok"}}

    client = OllamaClient(host="http://127.0.0.1:11434", model="demo:3b", transport=transport)
    response = client.chat([{"role": "user", "content": "hi"}])
    assert response["message"]["content"] == "ok"


def test_http_retry_once() -> None:
    calls = {"n": 0}

    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OllamaError("连接失败")
        return {"message": {"role": "assistant", "content": "重试成功"}}

    client = OllamaClient(
        host="http://x",
        model="m",
        transport=transport,
        http_retries=1,
    )
    response = client.chat([{"role": "user", "content": "hi"}])
    assert response["message"]["content"] == "重试成功"
    assert calls["n"] == 2


def test_http_retry_exhausted() -> None:
    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise OllamaError("一直失败")

    client = OllamaClient(host="http://x", model="m", transport=transport, http_retries=1)
    with pytest.raises(OllamaError, match="一直失败"):
        client.chat([{"role": "user", "content": "hi"}])


def test_empty_model_rejected() -> None:
    with pytest.raises(OllamaError):
        OllamaClient(host="http://x", model="  ")
