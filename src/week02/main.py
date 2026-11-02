"""week02 CLI：多工具 ReAct Agent。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from week02.agent_loop import AgentResult, run_agent
from week02.memory import load_messages, save_messages
from week02.ollama_client import OllamaClient, OllamaError
from week02.tools import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地多工具 ReAct Agent（week02）")
    parser.add_argument("prompt", nargs="?", help="一次性提问；省略则进入交互")
    parser.add_argument("--model", default=None, help="覆盖 OLLAMA_MODEL")
    parser.add_argument("--host", default=None, help="覆盖 OLLAMA_HOST")
    parser.add_argument("--max-steps", type=int, default=6, help="最大工具步数")
    parser.add_argument(
        "--max-tool-retries",
        type=int,
        default=2,
        help="可恢复工具错误（bad_args / unknown_tool）最多重试次数",
    )
    parser.add_argument("--session", default=None, help="会话 id，启用落盘记忆")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="会话目录，默认仓库根下 sessions/",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印中间步骤")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = OllamaClient(host=args.host, model=args.model)
    except OllamaError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    registry = build_default_registry()
    if args.prompt is not None:
        return _one_shot(args, client, registry)
    return _repl(args, client, registry)


def _one_shot(args: argparse.Namespace, client: OllamaClient, registry) -> int:
    prior = _load_prior(args)
    try:
        result = run_agent(
            args.prompt,
            client,
            registry=registry,
            max_steps=args.max_steps,
            max_tool_retries=args.max_tool_retries,
            prior_messages=prior,
        )
    except OllamaError as exc:
        print(f"调用失败：{exc}", file=sys.stderr)
        return 1
    _print_result(result, quiet=args.quiet)
    _maybe_save(args, result)
    return 0


def _repl(args: argparse.Namespace, client: OllamaClient, registry) -> int:
    print("week02 Agent。空行或 exit 退出。")
    prior = _load_prior(args)
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line.lower() in {"exit", "quit", "q"}:
            return 0
        try:
            result = run_agent(
                line,
                client,
                registry=registry,
                max_steps=args.max_steps,
                max_tool_retries=args.max_tool_retries,
                prior_messages=prior,
            )
        except OllamaError as exc:
            print(f"调用失败：{exc}", file=sys.stderr)
            continue
        _print_result(result, quiet=args.quiet)
        prior = result.messages
        _maybe_save(args, result, messages=prior)
    return 0


def _load_prior(args: argparse.Namespace) -> list | None:
    if not args.session:
        return None
    return load_messages(args.session, args.sessions_dir) or None


def _maybe_save(
    args: argparse.Namespace,
    result: AgentResult,
    messages: list | None = None,
) -> None:
    if not args.session:
        return
    save_messages(args.session, messages or result.messages, args.sessions_dir)


def _print_result(result: AgentResult, *, quiet: bool) -> None:
    if not quiet:
        for index, step in enumerate(result.steps, start=1):
            if step.thought:
                print(f"[{index}] Thought: {step.thought}")
            for action, observation in zip(step.actions, step.observations):
                print(f"[{index}] Action: {action}")
                print(f"[{index}] Observation: {observation}")
            if not step.actions and step.raw and "Final Answer:" not in step.raw:
                # 纯文本最终步已在 answer 里
                pass
    print(f"回答: {result.answer}")
    if not quiet and result.stop_reason != "final_answer":
        print(f"(停止原因: {result.stop_reason})")


if __name__ == "__main__":
    raise SystemExit(main())
