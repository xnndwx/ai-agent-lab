# Week01 学习总结：本地 Ollama + 最小 ReAct Agent

> 基于仓库 `xnndwx/ai-agent-lab` 第 1 周材料（`docs/week-01.md`、`src/week01/`）与本机实践整理。  
> 机器：note250138；模型路径：`F:\AI-Box\ollama\models`；已拉取并跑通 `qwen2.5:3b`。  
> 整理时间：2026-11-02（UTC+8）

---

## 1. 本周目标与收获

### 目标

看清一件事：**Agent 不是神秘框架，而是「模型 + 循环 + 工具」。**

- 模型：只负责生成下一段文字（或一次函数调用）
- 循环：决定还要不要再问模型（有没有工具可调、是否已可回答）
- 工具：做模型自己做不好、或不该编造的事（本周仅无副作用桩）

本周把这三块都写在**本地 Ollama** 上，参数量约 **3B**，普通笔记本 / iGPU 也能跑。

### 收获（可对照自检）

| 维度 | 本周实际掌握点 |
| --- | --- |
| 运行时 | Ollama 把模型变成 HTTP API；`pull` / `run` / `serve` 的职责不同 |
| 工程 | Python 项目用 `pyproject.toml` + venv + `pip install -e ".[dev]"` + pytest |
| 客户端 | 手写 `OllamaClient`，标准库 `urllib` 调 `POST /api/chat` |
| Agent | 手写 ReAct：`Thought → Action → Observation → Final Answer`，默认最多 6 步 |
| 框架边界 | **week01 不使用 LangChain / LangGraph**，便于逐行理解隐藏逻辑 |
| 硬件约束 | 3B 优先；7B 在弱 iGPU / 内存紧张时更难稳定跑 |

Java 背景对照：`pyproject.toml` ≈ `pom.xml`；`OllamaClient` ≈ 封装好的 HTTP Client（类似 OkHttp/RestTemplate 薄封装）。

---

## 2. 环境与工具

### Python / 工程

- **Python**：≥ 3.10
- **虚拟环境**：`python -m venv .venv`，激活后 `pip install -e ".[dev]"`
- **配置**：`.env` 来自 `.env.example`，本周只需：
  - `OLLAMA_HOST=http://127.0.0.1:11434`
  - `OLLAMA_MODEL=qwen2.5:3b`
- **测试**：`pytest`（**不连接真实 Ollama**；注入假 `transport` / 脚本回复）
- **运行**：`python -m week01 "问题……"`

### Ollama（本机 note250138）

| 项 | 实践情况 |
| --- | --- |
| 版本 | Ollama **0.40.13** |
| 模型目录 | `OLLAMA_MODELS=F:\AI-Box\ollama\models` |
| 模型 | 已 `ollama pull qwen2.5:3b`，可通过 API 调用 |
| 常用命令 | `ollama serve`（开服务）、`ollama pull`（拉模型）、`ollama run`（交互试聊）、`curl $OLLAMA_HOST/api/tags`（探活） |

说明：改 `OLLAMA_MODELS` 等环境变量后，通常需要**重启 Ollama 服务**才生效；关掉某个 `ollama run` 终端 **≠** 关掉后台 `serve`。

### 建议模型与硬件

- 默认 / 推荐：`qwen2.5:3b`（中文指令相对稳）；可选 `llama3.2:3b`
- 本周**不必**强上 7B：小模型「不听话」反而更容易暴露协议与解析器问题；iGPU / 内存紧张时 7B 更吃紧

---

## 3. 核心概念

### 3.1 Ollama vs 模型

| 概念 | 是什么 | 不是什么 |
| --- | --- | --- |
| **Ollama** | 本机运行时 / 部署框架：拉模型、常驻服务、暴露 HTTP API | 某一个具体大模型本身 |
| **模型**（如 `qwen2.5:3b`） | 权重 + 推理能力；由 Ollama 加载并对外提供 chat/generate | 「装了 Ollama 就等于装了所有模型」 |

一句话：Ollama 把「模型」变成「可被客户端调用的本地 API 服务」。

### 3.2 API

- 本周客户端只封装：`POST {OLLAMA_HOST}/api/chat`
- 请求体要点：`model`、`messages`、`stream=false`，可选 `tools`
- 默认地址：`http://127.0.0.1:11434`
- 探活：`GET /api/tags`（列出本地已有模型）

`OllamaClient` = 薄 HTTP 客户端；测试可注入 `transport`，因此单元测试不依赖真服务。

### 3.3 ReAct 闭环（本周定义）

一轮典型轨迹：

1. **Thought**：模型说明为什么需要信息 / 工具  
2. **Action** + **Action Input**：选择工具名 + **同一行** JSON 参数  
3. **Observation**：程序执行工具，把结果写回对话  
4. 可回答时 → **Final Answer**，循环停止  

防护：默认 `DEFAULT_MAX_STEPS = 6`，防止小模型空转。

两种调用方式走**同一个循环、同一套工具**：

- 原生 `tool_calls`（优先）
- 纯文本协议（`Thought` / `Action` / `Action Input` / `Final Answer`）

容错：若正文没有 `Action`，循环可把整段正文当作最终回答，避免 3B 不守格式时空转。

### 3.4 与 LangChain 的关系（重要）

- **week01 明确不用 LangChain / LangGraph**
- 原因：框架会把提示词、消息格式、循环藏进库里；初学时隐藏的部分正是要理解的部分
- 手写三件套：
  - `ollama_client.py`：标准库 HTTP
  - `tools.py`：手写函数 + schema
  - `agent_loop.py`：普通 `for` 循环
- 后续若对比框架，可用这份手写版做对照基线

### 3.5 开源模型许可（概念层）

使用开源模型前要看清许可证用途（商用 / 研究 / 改权重再分发等）。本周本地学习实验通常风险低，但就业项目 / 对外产品需单独核对所选模型的 license。

---

## 4. 代码结构对应关系

仓库路径（本机）：`F:\AI-Box\ai-agent-lab`；源码布局如下。

```text
ai-agent-lab/
├── pyproject.toml              # ≈ pom.xml：项目元数据、依赖、入口、pytest 配置
├── .env.example                # OLLAMA_HOST / OLLAMA_MODEL
├── README.md
├── docs/week-01.md             # 本周讲义
├── src/week01/
│   ├── main.py                 # CLI 入口：读环境 → 建 Client → run_agent
│   ├── ollama_client.py        # HTTP 客户端：POST /api/chat
│   ├── tools.py                # echo / add / now + TOOL_SCHEMAS + run_tool
│   ├── agent_loop.py           # 最小 ReAct 循环（核心）
│   └── __main__.py             # python -m week01
└── tests/                      # 假 transport，不连真 Ollama
    ├── test_ollama_client.py
    ├── test_agent_loop.py
    └── test_tools.py
```

### 调用链（Java 式理解）

```text
main.py
  → 读 .env（OLLAMA_HOST、OLLAMA_MODEL）
  → new OllamaClient(...)
  → run_agent(user_text, client)

agent_loop.py
  → 组装 system + user messages
  → for step in 1..max_steps:
        client.chat(messages, tools)
        有原生 tool_calls 或文本 Action → run_tool → 写回 Observation
        否则 → Final Answer / 纯文本 → 返回
  → 步数用尽 → stop_reason = max_steps

tools.py
  → TOOL_FUNCS 注册表 + TOOL_SCHEMAS（告诉模型参数形状）
  → 桩工具：echo（回显）、add（加法）、now（UTC 时间）
```

### 与 Java 心智模型对照

| Python / 本周 | Java 近似 |
| --- | --- |
| `pyproject.toml` | `pom.xml` / Gradle 构建描述 |
| `pip install -e ".[dev]"` | 本地 install + 测试依赖 |
| `OllamaClient` | 薄 RestClient / Feign，只打一个接口 |
| `transport` 注入 | 接口抽象 + mock，便于单测 |
| `agent_loop.run_agent` | 状态机 / while 循环编排服务 |
| `TOOL_SCHEMAS` | API 契约 / OpenAPI 片段给模型看 |
| `pytest` | JUnit（本周测的是循环与协议，不是「模型聪不聪明」） |

---

## 5. 易错点

1. **PATH / 命令找不到**  
   安装 Ollama 或 Python 后，新开终端或检查 PATH；旧会话可能仍找不到 `ollama` / `python`。

2. **`OLLAMA_MODELS` 改了不生效**  
   环境变量改完后需**重启 Ollama 服务**（不仅是新开一个 shell）。本机模型目录：`F:\AI-Box\ollama\models`。

3. **关掉 `ollama run` 终端 ≠ 关掉服务**  
   `run` 是交互客户端；后台真正对外提供 API 的是 **`serve`（或系统服务）**。API 连不上时先确认 serve 是否仍在、端口 `11434` 是否监听。

4. **`.env` 与 serve 地址不一致**  
   `OLLAMA_HOST` 必须和实际 serve 地址一致，否则客户端报「无法连接 Ollama」。

5. **`Action Input` 必须是同一行合法 JSON 对象**  
   非法 JSON 会变成 Observation 错误说明，交给模型下一轮改；解析器刻意保持简单。

6. **同一条消息里既有 `Action` 又有 `Final Answer`**  
   本周循环**优先执行 Action**，忽略同条里的 Final Answer，减少「心算幻觉」。

7. **3B 不调用工具**  
   问题里写明「请使用 add / echo」；或依赖系统提示中的文本协议示例。循环已能接受无协议的直接回答。

8. **测试失败 ≠ Ollama 挂了**  
   pytest 用假传输，一般不访问 `127.0.0.1:11434`；测的是请求体与循环逻辑。

9. **本周不要给模型 shell / 读任意文件**  
   工具保持无副作用；先保证循环正确。

---

## 6. 对照你写的两点总结：纠正与扩充

### 你的原文（意译）

1. Ollama 是本地部署框架，把模型变成 API 服务。  
2. 客户端提问后走 thought → action → action_input，ReAct 有最大步数，最后 Final Answer。

### 评价

方向正确，已抓住主干。下面做**纠正粒度**与**必要扩充**。

### 对第 (1) 点：扩充

| 已对 | 建议补上 |
| --- | --- |
| Ollama ≈ 本地部署 / 运行时 | **模型**是另一层：要 `pull` 具体 tag（如 `qwen2.5:3b`） |
| 变成 API 服务 | 本周用的是 **`/api/chat`**；客户端只是 HTTP 调用方 |
| | `serve` 才是服务进程；`run` 是试聊客户端 |
| | `OLLAMA_MODELS` 决定权重落盘位置；改完要重启服务 |

更完整一句：  
**Ollama = 本机模型运行时；某个模型（如 qwen2.5:3b）= 被加载的权重；API = 二者对外的统一访问面。**

### 对第 (2) 点：纠正 + 扩充

| 已对 | 建议纠正 / 补上 |
| --- | --- |
| Thought / Action / Action Input / Final Answer | 完整闭环还应有程序侧的 **Observation**（工具执行结果写回对话） |
| 有 max steps | 默认 **6**；用尽则 `stop_reason=max_steps`，不是无限问 |
| | **优先原生 `tool_calls`**，文本协议是兼容路径 |
| | 有 `Action` 时先执行工具，**同条 Final Answer 会被忽略** |
| | 无 `Action` 时，可把整段正文当最终回答（容错 3B 不守格式） |
| 「客户端 ask → …」 | 更准确：`main` 调 `run_agent`；**循环在 `agent_loop`，工具在 `tools`，HTTP 在 `ollama_client`** |

更完整一句：  
**用户问题进入 ReAct 循环 → 模型 Thought →（可选）Action/Input 或原生 tool_calls → 程序 Observation → 重复直至 Final Answer 或达到 max_steps。**

### 一句话总校正版（可背）

> Agent = 本地模型（经 Ollama API）+ 手写循环（ReAct，含 Observation 与步数上限）+ 手写工具；week01 不引入 LangChain，是为了把这三块看清楚。

---

## 7. 建议下周可学什么（1–3 条）

1. **加一个有意义但仍可控的工具**（如本周练习的 `length`，或只读指定目录的小文件工具），并补 schema + pytest。  
2. **对比一次「手写循环 vs 极薄封装」**：仍可不整库上 LangChain，但弄清消息列表、tool schema、stop 条件分别对应框架里哪一层。  
3. **在固定硬件上做一次 3B vs（若资源够）更大一点模型的行为对比**：同一问题看工具调用率、格式遵守、延迟与显存/内存占用，形成选型直觉。

---

## 附：本周刻意不做的事

- 不接入 LangChain / LangGraph  
- 工具不读任意文件、不跑 shell、不访问网络  
- 不保存长期记忆（进程结束对话即结束）

