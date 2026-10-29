# ai-agent-lab

第 1 周的本地 Agent 学习骨架：在自己的电脑上用 Ollama 跑一个约 **3B** 参数的模型，亲手写完一个最小 **ReAct** 循环。

这一周不使用 LangChain，也不使用 LangGraph。模型、工具和循环都在 `src/week01/` 里，方便逐行阅读。

## 你会得到什么

- 一个只认识 `OLLAMA_HOST` 和 `OLLAMA_MODEL` 的配置
- 一个用标准库调用 Ollama `/api/chat` 的客户端
- 三个无副作用的工具桩：`echo`、`add`、`now`
- 一个最多若干步的 ReAct 循环：思考、行动、观察，然后给出最终回答
- 不连接真实 Ollama 也能通过的测试

更细的说明见 [docs/week-01.md](docs/week-01.md)。

## 环境

- Python 3.10 及以上
- 可选：本机 [Ollama](https://ollama.com/)。只跑测试时不需要安装或启动它

拉取一个约 3B 的模型（二选一即可）：

```bash
ollama pull qwen2.5:3b
# 或者
ollama pull llama3.2:3b
```

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

`.env` 里只需要这两项：

```bash
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
```

提问：

```bash
python -m week01 "19 加 23 是多少？请使用工具。"
```

不带参数时，程序会提示你输入一个问题。Ollama 没启动时，命令行会打印连接错误并返回退出码 1。

## 测试

```bash
pytest
```

测试用假的模型回复和假的 HTTP 传输，不会访问 `127.0.0.1:11434`。

## 目录

```text
.env.example                 只包含 OLLAMA_HOST 与 OLLAMA_MODEL
src/week01/ollama_client.py  调用本地 /api/chat
src/week01/tools.py          工具桩与 JSON schema
src/week01/agent_loop.py     最小 ReAct 循环
src/week01/main.py           命令行入口
docs/week-01.md              本周阅读材料
tests/                       不依赖真实模型的测试
```

## 文本协议

支持函数调用的模型会直接收到工具 schema。若模型不会原生 tool call，请让它按下面的格式回答：

```text
Thought: 为什么要调用工具
Action: add
Action Input: {"a": 19, "b": 23}
```

拿到观察结果后：

```text
Thought: 工具已经给出结果
Final Answer: 19 加 23 等于 42。
```

同一条消息里如果出现 `Action`，循环会先执行工具，不采用这条消息里的 `Final Answer`。这样可以减少 3B 模型把答案提前编出来。

## 本周不做的事

- 不接入 LangChain / LangGraph
- 工具不能读文件、不能跑命令、不能访问网络
- 不保存长期记忆，进程结束对话就结束
