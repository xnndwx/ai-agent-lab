"""简单会话落盘：sessions/<id>.json 只存 messages。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def session_path(session_id: str, sessions_dir: Path | None = None) -> Path:
    """校验会话 id，并返回 JSON 路径。"""
    if not _SAFE_ID.match(session_id):
        raise ValueError("session id 只允许字母数字、_、-，最长 64")
    root = (sessions_dir or DEFAULT_SESSIONS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{session_id}.json"


def load_messages(session_id: str, sessions_dir: Path | None = None) -> list[dict[str, Any]]:
    """读取历史消息；不存在则返回空列表。"""
    path = session_path(session_id, sessions_dir)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError("会话文件格式错误")
    return list(data["messages"])


def save_messages(
    session_id: str,
    messages: list[dict[str, Any]],
    sessions_dir: Path | None = None,
) -> Path:
    """覆盖写入会话。"""
    path = session_path(session_id, sessions_dir)
    payload = {"messages": messages}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
