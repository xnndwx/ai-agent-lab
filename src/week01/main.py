"""命令行入口：读取 .env，对一个问题运行第 1 周 Agent。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from week01.agent_loop import DEFAULT_MAX_STEPS, run_agent
from week01.ollama_client import OllamaClient, OllamaError


def load_env_file(path: Path) -> None:
    """读取 KEY=VALUE 形式的 .env。已经存在的环境变量不会被覆盖。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        # 去掉空白，并允许写成 export KEY=VALUE
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # 键名与去掉引号后的值
        name = key.strip()
        cleaned = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = cleaned


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。问题放在位置参数中。"""
    parser = argparse.ArgumentParser(description="第 1 周本地 Ollama Agent")
    parser.add_argument("question", nargs="*", help="要问 Agent 的问题")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="ReAct 循环的最大步数",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="环境变量文件路径，默认是当前目录的 .env",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """程序入口。成功打印回答并返回 0，失败返回 1。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    # 先装入 .env，OllamaClient 随后才能读到主机和模型
    load_env_file(Path(args.env_file))
    question = " ".join(args.question).strip()
    if not question:
        question = input("问题> ").strip()
    if not question:
        print("请提供一个问题。", file=sys.stderr)
        return 1
    # 使用环境变量或默认值创建客户端
    client = OllamaClient()
    try:
        result = run_agent(question, client, max_steps=args.max_steps)
    except (OllamaError, ValueError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
