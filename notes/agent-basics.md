# Agent 基础

- ReAct = Thought → Action → Observation → 再 Thought
- 工具失败时把错误写成 Observation，让模型改正
- 本周用 ToolRegistry 注册多工具，而不是 if/elif
