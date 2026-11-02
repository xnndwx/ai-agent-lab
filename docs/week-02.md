# Week02 学习计划：多工具注册、Observation 加固、会话记忆

> 仓库：`xnndwx/ai-agent-lab`（本机 `F:\AI-Box\ai-agent-lab`）  
> 硬件：16GB RAM / i5-1335U / 弱 iGPU → **模型继续 ≤3B**（主用 `qwen2.5:3b`）  
> 节奏：约 **1–2 小时/天 × 5–7 天**；Java 背景、Python 初学，计划里会对照 Java 心智模型  
> 原则：**仍尽量不引入 LangChain / LangGraph**；本周只在「手写 Agent 骨架」上升级  
> 整理时间：2026-11-02（UTC+8）

---

## 0. 相对 Week1 要升级什么

| Week1 已有 | Week2 升级目标 |
| --- | --- |
| 3 个硬编码桩工具（echo / add / now） | **工具注册表**：装饰器或注册函数一次登记「实现 + schema」；支持 **多工具串联** |
| Observation 多为「工具返回字符串」 | **结构化 Observation**（成功/失败、错误码、截断、可重试标记） |
| 工具失败 → 写回一句说明，靠模型自觉改 | **错误重试策略**（参数错 / 未知工具 / 超时；循环侧可再给模型 1–2 次机会） |
| 单轮问答，进程结束即丢历史 | **简单会话记忆**：同一进程多轮对话；可选落盘 JSON 会话文件 |
| 只跑 `qwen2.5:3b` | **可选** `qwen2.5-coder:1.5b` 或 `3b`，做「读一小段代码 / 解释」类工具演示（非必须） |
| 无 LangChain | **继续手写**；若好奇框架，只做「对照阅读」，不把业务绑死在框架上 |

一句话：Week1 证明「Agent = 模型 + 循环 + 工具」能跑通；Week2 让它更像**可演示的迷你产品骨架**（多工具、可恢复错误、有短期记忆）。

---

## 1. 本周总目标（验收导向）

做到下面几件事即可算 Week2 过关：

1. 用**注册机制**新增 ≥2 个有意义工具（见 §3），并能在一次提问里 **连续调用 ≥2 个不同工具**。  
2. Observation 带统一格式；非法参数 / 未知工具时，Agent **能重试并最终给出正确 Final Answer**（演示 + pytest）。  
3. CLI 支持 **多轮会话**（`--session` 或交互式 REPL），退出后再进可读回同一会话（简单 JSON 即可）。  
4. `pytest` 全绿，且**仍不连真实 Ollama**（假 transport / 脚本回复）。  
5. 文档：仓库内准备 `docs/week-02.md`（讲义）+ 本计划可对照执行；代码落在 `src/week02/`（实现时再建，本文件不改 git 仓库）。

---

## 2. 建议技术点（做什么 / 不做什么）

### 2.1 工具注册与多工具

- 从 Week1 的「两个全局 dict + 手改两处」升级为：
  - `@tool(name=..., description=..., parameters=...)` 装饰器，或  
  - `registry.register(func, schema)`  
- Java 对照：≈ Spring 里「接口实现 + Bean 注册」；schema ≈ OpenAPI 片段给模型看。  
- 循环侧：`get_schemas()` / `run(name, args)` 只依赖注册表，**新增工具不改 `agent_loop`**。  
- 多工具：同一轮可多个 `tool_calls`；文本协议仍「每轮一个 Action」也可接受，但要能在多步里串起来（例如先 `now` 再 `format_time`，或先 `read_note` 再 `word_count`）。

### 2.2 更好的 Observation 处理

建议统一成短字符串（模型好读）+ 内部结构（测试好断言），例如：

```text
[ok] add → 42
[error:bad_args] add → 参数 a 必须是数字（可重试）
[error:unknown_tool] foo → 未知工具；可用：add, echo, ...
[error:truncated] read_note → （内容过长，已截断到 500 字）
```

- 截断：防止小文件工具把上下文撑爆（3B 上下文宝贵）。  
- 禁止把 Python traceback 原文整段塞给模型；**翻译成可操作说明**。

### 2.3 错误重试

两层即可，不要上复杂状态机：

| 层 | 行为 |
| --- | --- |
| 工具层 | `run_tool` 永不向外抛未捕获异常；返回 `[error:...]` |
| 循环层 | 对 `bad_args` / `unknown_tool`：**不立刻停**；Observation 写回后继续下一轮，让模型改参数或换工具；可用 `max_tool_retries`（如 2）防止死循环 |
| HTTP 层（可选） | `OllamaClient.chat` 对连接失败做 **1 次有限重试**（指数退避或固定 1s），超时要有清晰退出码 |

### 2.4 简单记忆 / 会话

- **短期**：`messages` 列表跨用户轮次保留（REPL / `chat` 模式）。  
- **持久化（推荐本周做）**：`sessions/<id>.json` 存 messages（可含 system）；启动时 `--session demo` 加载。  
- **不做**：向量库、RAG、跨机器同步、用户画像。  
- 注意：持久化时要考虑 **截断过旧的 Observation**，避免文件和上下文无限涨。

### 2.5 模型选型（硬件友好）

| 模型 | 用途 | 说明 |
| --- | --- | --- |
| `qwen2.5:3b`（已有） | 主循环、中文指令 | 默认保持 |
| `qwen2.5-coder:1.5b`（优先可选） | 代码解释 / 小 diff 说明类工具 | 更省内存 |
| `qwen2.5-coder:3b` | 同上，效果略好 | 内存紧时别和主模型同时常驻；用完可 `ollama stop` |

路径仍建议：`OLLAMA_MODELS=F:\AI-Box\ollama\models`。改环境变量后重启 Ollama 服务。

### 2.6 与 LangChain 的边界（本周纪律）

- **默认：整周不 pip 安装 LangChain / LangGraph。**  
- 若某天「只想理解就业 JD 在说什么」：允许 **只读官方概念文档 30–45 分钟**，对照表写在笔记里，例如：
  - 你的 `TOOL_SCHEMAS` ≈ Tool / StructuredTool 的 schema  
  - 你的 `run_agent` ≈ AgentExecutor / 图里的一步循环  
  - 你的 `messages` ≈ ChatMessageHistory  
- **不要**为了「用上框架」而重写本周作业；手写基线是求职时讲得清原理的资本。

### 2.7 本周刻意不做

- 任意路径读盘、shell、任意 URL 抓取（安全与调试成本过高）  
- 真·长期记忆 / 向量检索  
- 多 Agent 协作、复杂 Planner  
- 7B+ 模型硬上  

允许的「有意义但仍可控」工具示例见 §3。

---

## 3. 建议工具清单（从中选 2–4 个实现）

| 工具 | 作用 | 约束（必须写进 description） |
| --- | --- | --- |
| `length` | 字符串长度 | Week1 练习延伸，可作第一个注册表示例 |
| `calc` | 仅支持 `+ - * /` 与括号的安全表达式 | **禁止** `eval` 任意代码；用手写解析或 `ast` 白名单 |
| `read_note` | 读取 `notes/` 下**指定文件名** | 白名单目录 + 禁止 `..`；超长截断 |
| `list_notes` | 列出 `notes/` 文件名 | 只读、限数量 |
| `word_count` | 统计一段文本词/字数 | 可与 `read_note` 串联演示多工具 |
| `code_explain`（可选，配 coder 模型） | 对传入的短代码字符串做说明 | 输入长度上限；**不执行代码** |

安全底线：所有路径类工具只碰仓库内 `notes/`（或 `data/notes/`），拒绝绝对路径与 `..`。

---

## 4. 每日 / 分块任务（约 6 天，可压缩成 5 或拉到 7）

> 每天 1–2h。打勾即过；写不动代码的那天就只做「读讲义 + 画调用链」。

### Day 1（1–1.5h）— 设计与目录骨架

- [ ] 通读本计划 + 回顾 `docs/week-01.md` / Week1 总结里「下周建议」三条。  
- [ ] 在纸上或笔记画出：`registry → schemas → agent_loop → observation → session`。  
- [ ] 约定仓库落点（实现周再建，本计划不改仓库）：
  - `docs/week-02.md`  
  - `src/week02/`（可从 week01 **复制再改**，避免破坏 week01 可运行基线）  
  - `tests/test_week02_*.py` 或 `tests/week02/`  
  - `notes/` 样例文本 2–3 个  
- [ ] Java 对照笔记：注册表 ≈ 服务定位器；Session ≈ 有状态的 Conversation scope。

### Day 2（1.5–2h）— 工具注册表

- [ ] 实现 `ToolRegistry`（或装饰器）：注册时同时写入 func + schema。  
- [ ] 把 `echo` / `add` / `now` 迁到新注册方式；行为与 Week1 兼容。  
- [ ] 新增 `length`（或 `word_count`），**只改注册、不改循环核心**。  
- [ ] pytest：注册后 schemas 含新工具；`run("length", ...)` 正确。

### Day 3（1.5–2h）— Observation 格式 + 多工具串联

- [ ] 统一 Observation 前缀（`[ok]` / `[error:...]`）。  
- [ ] 增加 `list_notes` + `read_note`（白名单目录）。  
- [ ] 写一条脚本化测试：假模型先 `list_notes` 再 `read_note`，断言两步 Observation 都进了 messages。  
- [ ] 真机试跑（可选）：`python -m week02 "请列出 notes 并读 xxx.txt 摘要"`。

### Day 4（1–2h）— 错误重试

- [ ] 循环侧：`bad_args` / `unknown_tool` 时继续跑，直到成功或耗尽 `max_steps` / `max_tool_retries`。  
- [ ] 假模型剧本：第一轮坏 JSON → Observation 报错 → 第二轮正确 Action → Final Answer。  
- [ ] （可选）`OllamaClient` 连接失败重试 1 次；测试用假 transport 模拟第一次失败。  
- [ ] 记录易错点：3B 可能重复相同错误 → 在 Observation 里写清「请换参数，不要重复同一错误」。

### Day 5（1.5–2h）— 会话记忆

- [ ] CLI：`python -m week02` 进入多轮；或 `--session mychat`。  
- [ ] 保存 / 加载 `sessions/<id>.json`（仅 messages；可剥掉过大字段）。  
- [ ] 演示：第一轮「记住我叫小明」；第二轮「我叫什么？」（可无工具，纯 messages 记忆）。  
- [ ] 再演示：跨轮引用上次 `read_note` 的内容（短期记忆价值）。  
- [ ] pytest：序列化往返不丢 role/content。

### Day 6（1–2h）— 可选 coder 模型 + 验收彩排

- [ ] （可选）`ollama pull qwen2.5-coder:1.5b`；`.env` 增加 `OLLAMA_CODER_MODEL` 或临时切换。  
- [ ] （可选）`code_explain` 工具：只吃字符串，不执行；用 coder 模型或主模型均可。  
- [ ] 跑通 §5 验收清单；补 `docs/week-02.md` 讲义提纲（目标 / 协议 / 目录 / 练习）。  
- [ ] 整理 5 分钟口述：如何向面试官解释「我为什么还不急着上 LangChain」。

### Day 7（缓冲，0–1.5h）

- 还债：测试红的、文档没写的、截断策略没做的。  
- 或提前预习 Week3 方向（自行决定，例如：受控 HTTP 工具、简单 RAG、或薄封装对比）——**不纳入本周必验**。

---

## 5. 验收标准（能演示什么）

准备一段 **5–10 分钟** 本地演示（或录屏），覆盖：

1. **多工具**：一问之内（或多步 ReAct）调用 ≥2 个工具，终端能看到 Thought / Action / Observation 轨迹。  
2. **注册表**：现场加一个小工具（或展示已加的 `length`），证明无需改循环核心。  
3. **错误恢复**：故意错误参数 → 模型根据 Observation 纠正 → 正确 Final Answer。  
4. **会话**：同一 `--session` 下两轮对话，第二轮用到第一轮信息；重启进程后仍能加载。  
5. **测试**：`pytest` 全绿，且测试过程无真实 Ollama 依赖。  
6. **硬件纪律**：演示模型仍为 ≤3B；能说清为何不盲目上 7B。

口头加分项（可选）：对照一句「LangChain 里对应哪一层」，但代码仍是手写。

---

## 6. 与仓库目录建议

实现本周内容时建议布局（**仅建议；本计划文件写在 box 的 `/workspace/docs/`，不直接改 git 仓库**）：

```text
ai-agent-lab/
├── docs/
│   ├── week-01.md              # 已有
│   ├── week01-学习总结.md      # 已有（可选）
│   └── week-02.md              # 本周讲义（Day6 补齐）
├── notes/                      # 只读白名单示例文本
│   ├── hello.txt
│   └── agent-tips.txt
├── sessions/                   # 本地会话 JSON（建议 gitignore）
├── src/
│   ├── week01/                 # 保留不动，作为对照基线
│   └── week02/
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py             # CLI：单问 / REPL / --session
│       ├── ollama_client.py    # 可复用 week01，或薄封装 + 重试
│       ├── registry.py         # 工具注册
│       ├── tools.py            # 具体工具实现
│       ├── observation.py      # 格式化 Observation（可与 tools 合并）
│       ├── memory.py           # 会话加载/保存
│       └── agent_loop.py       # 多工具 + 重试后的 ReAct
└── tests/
    ├── test_week02_registry.py
    ├── test_week02_agent_loop.py
    ├── test_week02_memory.py
    └── ...
```

`pyproject.toml` 后续可增加：

- script：`week02 = "week02.main:main"`  
- 描述更新为覆盖 week01+week02  
- **dependencies 仍建议为空**（标准库优先）；dev 继续 `pytest`

`.gitignore` 建议加入：`sessions/`、`.env`（若尚未忽略）。

---

## 7. 学习方法提示（Java → Python）

| 卡住点 | 建议 |
| --- | --- |
| 装饰器 / 类型注解看不懂 | 先写 `register(name, func, schema)` 普通函数，跑通再换成装饰器 |
| `dict[str, Any]` 太多 | 先用 `@dataclass` 表示 ToolSpec / AgentResult，减少裸 dict |
| 不知道测什么 | 测注册表、测 Observation 字符串、测「坏参数→好参数」剧本；**不要测模型智商** |
| 3B 不听话 | 问题里点名工具；系统提示保留文本协议；Observation 写清楚下一步该怎么改 |
| 想引入框架 | 先完成本周验收；框架对比放到笔记，不阻塞演示 |

---

## 8. 本周结束时的「一句话」

> Week2：在本地 Ollama（≤3B）上，用手写 ReAct 做出「可注册的多工具 + 结构化 Observation/重试 + 可持久化短会话」的 Agent 骨架；仍不绑定 LangChain，以便求职时讲清每一层。

---

## 附：与 Week1 总结「下周建议」的对应

1. 有意义仍可控的工具 → §3 `read_note` / `list_notes` / `calc` 等  
2. 手写 vs 极薄封装对照 → §2.6 只读概念、不进依赖  
3. 3B 行为直觉 → 主模型保持 3B；可选 coder 1.5b/3b 做代码向工具，不做 7B 强上  
