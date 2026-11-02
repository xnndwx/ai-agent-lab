"""会话落盘测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from week02.memory import load_messages, save_messages, session_path


def test_roundtrip(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "好的"},
    ]
    path = save_messages("demo", messages, tmp_path)
    assert path == tmp_path / "demo.json"
    assert load_messages("demo", tmp_path) == messages


def test_missing_session_is_empty(tmp_path: Path) -> None:
    assert load_messages("none", tmp_path) == []


def test_rejects_unsafe_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        session_path("../evil", tmp_path)
    with pytest.raises(ValueError):
        session_path("a" * 65, tmp_path)
