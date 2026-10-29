# 第 1 周：最小 ReAct Agent

本周目标是看清一件事：**Agent 不是一个神秘框架，而是「模型 + 循环 + 工具」。**

模型只负责生成下一段文字（或一次函数调用）。循环负责决定还要不要再问它。工具负责做模型自己做不好、或不该编造的事。第 1 周把这三块都写在本地 Ollama 上，参数量控制在约 3B，这样普通笔记本也能跑。

## 为什么先不使用 LangChain / LangGraph

框架会把提示词、消息格式和循环藏进库里。第一次学习时，隐藏的部分正好是要理解的部分。所以本周：

- `ollama_client.py` 只用 Python 标准库发 HTTP
- `tools.py` 是几只手写的函数
- `agent_loop.py` 是一个普通的 `for` 循环

后面几周如果要比较框架，也有这份手写版本可以对照。本仓库的第 1 周代码不导入 LangChain 或 LangGraph。

## ReAct 在这一周是什么

ReAct 把一轮对话拆成：

1. **Thought**：模型说明为什么需要信息
2. **Action**：选择一个工具，并给出参数
3. **Observation**：程序执行工具，把结果交回模型

可以回答时，模型写出 **Final Answer**，循环停止。为了防止小模型不停调用工具，默认最多 6 步。

一次加法的轨迹如下。

用户：

```text
19 加 23 是多少？
```

模型：

```text
Thought: 这是加法，应该调用 add，而不是心算。
Action: add
Action Input: {"a": 19, "b": 23}
```

程序执行 `add`，再追加一条消息：

```text
Observation: 42
```

模型：

```text
Thought: 工具已经给出和。
Final Answer: 19 加 23 等于 42。
```

`Action Input` 必须是**同一行**上的 JSON 对象。这是为了让解析器保持在几十行以内。

如果模型支持 Ollama 的原生 `tool_calls`，循环会优先执行那些调用，并把结果写成 `role` 为 `tool` 的消息。文本协议和原生调用走的是同一个循环，工具实现也是同一套。

小模型经常不遵守格式，直接输出一句中文。此时只要正文里没有 `Action`，循环就把这段正文当作最终回答，避免空转。

## 代码怎么串起来

```text
main.py
  读取 .env 里的 OLLAMA_HOST、OLLAMA_MODEL
  创建 OllamaClient
  调用 run_agent

agent_loop.py
  组装 system + user 消息
  反复调用 client.chat(messages, tools)
  有工具调用就 run_tool，把观察写回 messages
  没有工具调用就返回回答

tools.py
  echo / add / now
  TOOL_SCHEMAS 负责告诉模型每个工具的参数

ollama_client.py
  POST {OLLAMA_HOST}/api/chat
  请求体包含 model、messages、stream=false，以及可选的 tools
```

系统提示里的工具名单来自 `TOOL_SCHEMAS`，这样改工具说明时不用改两处文案。

## 三个工具桩

| 工具 | 作用 | 故意不做的事 |
| --- | --- | --- |
| `echo` | 原样返回 `text`，用来确认调用打通 | 不记录日志文件 |
| `add` | 计算两个数的和 | 不执行任意表达式 |
| `now` | 返回当前 UTC 时间 | 不查询时区数据库以外的信息 |

它们叫「桩」，是因为后面几周会换成搜索、读文件等真正有副作用的工具。第 1 周先保证循环是对的。未知工具或坏参数不会让进程崩溃，而是变成一段 Observation，让模型有机会改口。

## 建议的模型

`.env.example` 默认是 `qwen2.5:3b`，中文指令通常比同尺寸的英文模型稳一些。也可以换成 `llama3.2:3b`。不需要为了本周去下载 7B 以上的模型；小模型不听话，反而更容易看见协议和解析器在做什么。

确认服务：

```bash
curl "$OLLAMA_HOST/api/tags"
```

## 练习

1. 运行 `python -m week01 "请用 echo 回显：学习 Agent"`，看它是否真的调用了工具。
2. 把 `DEFAULT_MAX_STEPS` 改成 1，再问一个必须使用 `add` 的问题，观察停止原因 `max_steps`。
3. 在 `tools.py` 增加一个只做字符串长度的工具 `length`，同时补上 schema 和测试。
4. 故意把 `Action Input` 写成非法 JSON，看下一轮模型能不能根据 Observation 改正。

做练习时保持工具无副作用。不要在第 1 周把 shell 暴露给模型。

## 测试为什么不连接 Ollama

`OllamaClient` 接受一个 `transport` 函数。测试用脚本客户端代替模型，按顺序交还写好的回复。因此 `pytest` 检查的是：

- 请求体里有没有模型名、消息和工具
- 文本 `Action` 与原生 `tool_calls` 会不会执行
- 观察结果有没有进入下一轮
- 步数用尽时会不会停

这和「模型答得聪不聪明」是两件事。答得聪不聪明，需要你本机的 Ollama；循环写得对不对，测试就可以证明。

## 常见问题

**`无法连接 Ollama`**  
先启动 Ollama，并确认 `.env` 的 `OLLAMA_HOST` 与 `ollama serve` 的地址一致。测试失败通常不是这个原因。

**模型一直不调用工具**  
3B 模型可能忽略 schema。在问题里写明「请使用 add」，或在系统提示里保留文本协议示例。循环已经能接受不带协议的直接回答。

**模型在调用工具前就写出了 Final Answer**  
同一条消息里只要有 `Action`，本周的循环会忽略这条 `Final Answer`，先拿观察结果。这样可以减少一次心算幻觉。
