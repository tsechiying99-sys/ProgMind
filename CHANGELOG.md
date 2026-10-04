# 修改日志

## 2026-09-22

### 本次目标与实施边界

按“工具拆分与统一入口 → 消除重复定义 → 接入真实能力”的推荐顺序实施到第三阶段。本节内容取代 2026-09-21 日志中关于 `agents/tools.py`、`AgentToolSpec` 和 `knowledge_search` 适配器的现状说明；结构化 Agent 角色设计继续保留。

本次修改仍只保留在本地工作区，**没有提交或推送到 GitHub**。修改前的远端基线仍为 `main` 分支提交 `2ab732d Establish learning assistant baseline`。未修改或删除 `output/` 中的用户生成文件。

### 第一阶段：拆分工具并建立统一注册入口

- 删除集中式 `agents/tools.py`，新增根目录 `tools/` 包。
- 最终只保留 5 个非重复的 Agent 工具，每个工具使用独立模块：
  - `check_prerequisites.py`
  - `build_hint_plan.py`
  - `search_question_bank.py`
  - `grade_answer.py`
  - `record_learning_result.py`
- 新增 `tools/contracts.py`：
  - `ToolExecutionContext` 统一传递 `request_id`、`user_id`、`conv_id`、Agent 类型、意图、紧急程度和脱敏后的请求上下文。
  - `ToolDependencies` 显式注入知识库和记忆管理器，工具模块不读取 API 全局变量。
  - `object_schema()` 统一生成 `additionalProperties: false` 的严格对象 Schema。
- 新增 `tools/registry.py`，由 `register_learning_tools()` 在应用启动时一次性显式注册工具；导入模块不会产生注册副作用。

### 第二阶段：由 MCPToolManager 唯一管理定义、权限校验和执行

- 移除 `AgentToolSpec` 及 Agent 内部的 `_role_tools()`、`set_shared_tools()` 和直接 Handler 调用。
- `mcp/tool_manager.Tool` 现在是唯一工具定义，新增：
  - `to_model_schema()`：生成发送给模型的工具定义。
  - `read_only`：标识工具是否会写数据，便于审计及后续策略扩展。
- `MCPToolManager` 新增公开查询接口：
  - `get_tool()`
  - `get_tools()`
  - `get_tool_schemas()`
  - `registered_names`
- 重复注册默认抛出错误，必须显式传入 `replace=True` 才能覆盖，避免同名工具静默替换。
- 工具管理器统一执行同步和异步 Handler；同步 Handler 放入线程运行，避免阻塞事件循环。
- 参数校验统一收口到工具管理器，支持：
  - 必填参数和未知参数检查
  - string、number、integer、boolean、array、object 类型检查
  - 阻止 `bool` 冒充 integer/number
  - enum、minimum、maximum、minLength、maxLength
  - 数组元素的递归类型检查
- 无效参数被视为调用方错误，不触发工具 fallback、失败计数或熔断器。
- `BaseAgent` 只从管理器读取 `AgentProfile.tool_scope` 声明的工具；模型伪造未授权工具名时仍在服务端拒绝。
- 所有实际工具调用统一经过 `MCPToolManager.call()`，因此超时、缓存、熔断、降级、统计和参数校验只有一套实现。

### 最终架构：固定预处理与 Agent 工具分离

- 删除 `knowledge_search` 和 `search_course_material` 两个工具注册名，不再把普通课程 RAG 暴露给 Agent。
- `/chat` 和 `/chat/stream` 在 Agent 执行前，通过 `_search_course_knowledge()` 直接调用 `KnowledgeBase.search_handler()`；Memory 仍由 `_prepare_chat_request()` 统一预加载。
- 快速 RAG 使用一次 Hybrid 检索；增强 RAG 使用查询改写、并行召回、文档去重和按融合分数排序。
- `/search` 复用相同的增强 RAG 服务函数。
- `MCPToolManager` 只管理模型可以自主选择的 5 个 Agent 工具，消除预加载 Memory/RAG 后再次调用同类工具的重复路径。
- `search_question_bank` 仍复用 KnowledgeBase，但用途是按题目标记筛选练习材料，不等同于普通课程 RAG。
- API 使用内部 `_knowledge_base` 引用处理课程检索、导入和统计，不再访问 `_tool_manager._tools` 或 `handler.__self__`。

### 第三阶段：保留非重复的真实能力

- `check_prerequisites`：为 Explain Agent 提供前置知识检查规则。
- `build_hint_plan`：为 QA Agent 提供渐进提示规则。
- `search_question_bank`：只在现有课程知识库检索结果中筛选明确带有“题目、习题、练习、quiz、question”标记的内容。项目当前没有独立题库，因此没有伪造题库服务；无匹配时明确返回空结果和 `configured=false`。
- `grade_answer`：使用确定性规则评分：
  - 提供 `key_points` 时计算关键点覆盖率。
  - 未提供关键点时计算规范化文本相似度。
  - 返回分数、是否通过、命中点、缺失点和评分方法，并声明开放题仍需教师复核。
- `record_learning_result`：把已完成练习的知识点、题目、学生答案、分数、正确性和反馈写入 Memory。
- `MemoryManager` 新增公开方法：
  - `record_learning_result()`：Redis 中按用户保存最近 100 条，TTL 为 30 天。
- 删除 `inspect_learning_context`、`build_quiz_constraints`、`search_course_material`、`get_learning_profile`、`get_course_progress`、`get_recent_mistakes` 六个工具模块。
- 画像、进度、最近对话、会话摘要和课程资料继续由固定预处理流程一次性注入 Agent 上下文。

### Agent 工具权限调整

- Explain Agent：仅 `check_prerequisites`。
- QA Agent：仅 `build_hint_plan`。
- Quiz Agent：仅 `search_question_bank`、`grade_answer`、`record_learning_result`。
- 写工具 `record_learning_result` 仅对 Quiz Agent 开放，其余角色不可见且无法通过伪造名称执行。

### 启动顺序调整

`api/main.py` 的生命周期初始化顺序改为：

1. 加载 Skills。
2. 初始化 MemoryManager。
3. 初始化 MCPToolManager 和 KnowledgeBase。
4. 使用依赖对象注册全部学习工具。
5. 创建 AgentOrchestrator，并注入同一个 MCPToolManager。
6. 启动 Monitor 和 Evaluator。

这样可以保证 Agent 创建时已拥有完整、唯一的工具注册表。

### 测试与验证

- 更新 `tests/test_agent_profiles.py`：验证三类角色契约、工具范围和越权工具隔离。
- 更新 `tests/test_agent_tools.py`：验证模型可见白名单、服务端越权拒绝、参数错误拒绝、工具轮数限制，以及已删除上下文工具不再注册。
- 新增 `tests/test_tool_manager.py`：验证注册表恰好包含 5 个工具、重复注册保护、同步/异步统一调用、评分与学习结果记录、递归参数校验和数值边界。
- 新增 `tests/test_chat_rag.py`：验证普通课程 RAG 脱离 Agent 工具后，快速检索和增强检索仍可直接调用 KnowledgeBase，并正确合并去重结果。
- 验证命令：

```powershell
.\.venv\Scripts\python.exe -m compileall agents api mcp memory tools tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "import api.main; from tools import register_learning_tools; print('imports-ok')"
```

- 验证结果：Python 编译通过，API 与工具包导入通过，15 个单元测试全部通过。
- 测试使用 Fake KnowledgeBase、Fake Memory 和 Fake LLM Client，不需要真实 Anthropic、Redis 或 ChromaDB 服务。

### 当前明确边界

- 流式 `/chat/stream` 仍沿用文本流，不在流中执行工具；普通 `/chat` 支持完整工具调用循环。
- `search_question_bank` 复用课程知识库并诚实筛选题目标记，不等同于独立题库。以后接入真实题库时只需替换该工具 Handler，不需要改 Agent。
- `grade_answer` 是可复现的初步评分规则，不是语义评阅模型，不适合单独决定复杂开放题成绩。
- 本次没有新增第三方依赖，也没有更改前端接口 Schema。

## 2026-09-21

### 修改前 GitHub 基线

- 在本次 Agent 改造开始前，将当前学习助手代码提交为 `2ab732d Establish learning assistant baseline`。
- 基线已推送到 `https://github.com/tsechiying99-sys/EchoMind` 的 `main` 分支。
- `.env`、`.venv`、`node_modules`、前端构建产物和 `output/` 生成文件未包含在基线提交中。
- 本节以下修改仅保留在本地工作区，暂未提交或推送到 GitHub。

### 修改范围

本次严格围绕两项目标实施：

1. 将 Explain、QA、Quiz Agent 从“不同 System Prompt”升级为结构化角色。
2. 引入模型可见范围、服务端执行范围和参数 Schema 三层一致的 Agent 工具白名单。

业务范围外的 Memory、混合 RAG 算法、Evaluator、Monitor、Skills 和前端均未修改。涉及文件：

- 修改 `agents/agent_orchestrator.py`
- 新增 `agents/tools.py`
- 小改 `api/main.py`
- 新增 `tests/test_agent_profiles.py`
- 新增 `tests/test_agent_tools.py`
- 更新本修改日志 `CHANGELOG.md`

### 1. Agent 结构化角色契约

- 新增不可变数据结构 `AgentProfile`，包含：
  - `role`：角色定位
  - `responsibilities`：职责列表
  - `workflow`：确定性处理流程
  - `input_contract`：允许使用的输入
  - `output_contract`：最终回答必须满足的输出约束
  - `escalation_conditions`：需要教师或外部确认的条件
  - `tool_scope`：角色允许使用的工具名
  - `temperature`、`max_tokens`：角色级生成参数
- `BaseAgent._build_system_prompt()` 现在统一注入结构化角色契约，再追加已有动态 Skills。
- `ExplainAgent` 的处理契约固定为“识别知识点 → 检查前置知识 → 概念讲解 → 示例 → 理解检查”。
- `QaAgent` 的处理契约固定为“识别已尝试步骤 → 定位首个卡点 → 渐进提示 → 必要时提出一个澄清问题”。
- `QuizAgent` 的处理契约固定为“确定知识点 → 规范难度和题型 → 只生成一道题 → 学生作答前隐藏答案与解法”。
- 三类 Agent 使用不同的温度、最大输出和工具范围，不再只依靠 Prompt 文案体现角色差异。

### 2. Agent 工具定义与白名单

- 新增 `AgentToolSpec`，将工具名、说明、严格 JSON Schema 和可信 Handler 绑定为一个不可变定义。
- 所有工具 Schema 默认设置 `additionalProperties: false`，禁止模型添加未声明参数。
- `BaseAgent.get_tools()` 只返回“实际已注册工具”和 `AgentProfile.tool_scope` 的交集。
- 模型请求的 `tools` 参数仅包含当前 Agent 可见工具，其他角色工具不会发送给模型。
- 模型即使伪造未授权工具名，服务端仍会拒绝执行，并以失败的 `tool_result` 返回模型。
- 执行前校验：
  - 参数必须为 JSON 对象
  - 所有必填字段必须存在
  - 不允许未知字段
  - 校验 string、number、integer、boolean、array、object 基础类型
  - 防止 Python `bool` 被错误当作 number/integer
- 工具调用循环受 `AGENT_MAX_TOOL_ROUNDS` 限制，默认最多 3 轮，防止模型无限调用工具。
- `AgentResponse` 和 `OrchestratorResult` 新增内部 `tools_used` 与 `tool_traces`，但暂未改变公开 `/chat` 响应结构。

### 3. 当前工具清单

Explain Agent：

- `inspect_learning_context`：只返回上下文是否存在、历史轮数等脱敏状态。
- `check_prerequisites`：生成保守的前置知识检查清单，不断言学生已掌握。
- `search_course_material`：只读检索现有课程知识库。

QA Agent：

- `inspect_learning_context`
- `build_hint_plan`：生成由弱到强的提示方案，默认不泄露完整答案。
- `search_course_material`

Quiz Agent：

- `inspect_learning_context`
- `build_quiz_constraints`：规范知识点、难度和题型，并强制隐藏答案和解法。
- `search_course_material`

### 4. 复用现有课程知识库

- `api/main.py` 在现有 `knowledge_search` 注册完成后创建 `search_course_material` 适配器，并注入 Orchestrator。
- 新工具继续调用现有 `MCPToolManager.call()`，因此保留已有参数校验、超时、熔断、缓存和降级机制。
- 未修改 `mcp/knowledge_base.py`，BGE、BM25 和混合融合排序保持原样。
- 未移除 API 层已有的课程资料预检索。普通 `/chat` 中 Agent 可在现有上下文不足时追加调用课程检索；相同查询继续受 ToolManager 缓存保护。

### 5. 兼容性与明确边界

- `/chat` 请求和响应 Schema 保持不变，现有前端不需要同步修改。
- `/chat` 普通调用已支持真实模型 tool-use 循环。
- `/chat/stream` 继续使用原有直接文本流和结构化角色 Prompt，但本次没有加入流中工具调用；这是为了把修改严格限制在指定两项并避免改变 SSE 协议。流式工具循环可作为独立后续任务处理。
- 工具均为只读分析或检索工具，不提供知识库写入、画像写入、文件操作或任意代码执行能力。
- 本次没有引入新的第三方依赖。

### 6. 自动化验证

- 新增 10 个基于标准库 `unittest` 的测试，不依赖 pytest：
  - 三类 Agent 具有不同的结构化角色契约。
  - 每个 Profile 均包含职责、流程、输入、输出和升级条件。
  - 角色级温度和最大输出参数确实不同。
  - System Prompt 包含结构化契约与工具权限。
  - 三类 Agent 的本地工具范围相互隔离。
  - 共享课程检索只有在声明并注入后才可见。
  - 模型只收到当前 Agent 白名单内的工具定义。
  - 伪造的越权工具调用会被服务端拒绝。
  - 未声明字段和错误参数类型会被拒绝。
  - 工具调用达到配置轮数后会终止，不会无限循环。
  - 课程检索适配器能够复用现有 `knowledge_search`。
- 验证命令：

```powershell
.\.venv\Scripts\python.exe -m py_compile agents\agent_orchestrator.py agents\tools.py api\main.py tests\test_agent_profiles.py tests\test_agent_tools.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

- 验证结果：10 个测试全部通过。

## 2026-08-22

### 前端流式输出

- 聊天页优先请求 `POST /chat/stream`，使用 `fetch + ReadableStream + TextDecoder` 持续解析 SSE 数据。
- 支持 `delta`、`token`、`meta`、`done`、`error` 事件，以及 `data: [DONE]` 结束标记。
- 收到文本增量时立即更新同一条助手消息，并显示闪烁光标、自动滚动和输出模式提示。
- 后端未提供流式接口（404、405、501）时，自动回退现有 `POST /chat`，并渐进显示完整回答，保证原接口仍可使用。
- 取消操作同时支持中止 SSE 请求、普通 Axios 请求和兼容模式的渐进展示。
- 空回答不会作为成功消息保留；等待动画只在首个文本片段到达前显示。

### 后端 SSE 流式接口

- 新增 `POST /chat/stream`，响应类型为 `text/event-stream`，与前端已经实现的流式客户端直接匹配。
- 新增 `meta`、`delta`、`done`、`error` 四类 SSE 事件；中文数据使用 UTF-8 JSON 传输。
- Agent 使用 Anthropic SDK 的 `messages.stream()` 直接转发模型文本增量，不再等待完整回答后才返回。
- 普通调用和流式调用共用模型参数、System Prompt、动态 Skills、Memory 与 RAG 上下文。
- `/chat` 和 `/chat/stream` 共用请求准备函数；Memory 与 RAG 仍保持并行读取。
- 单 Agent 请求真正逐 token 输出；复合意图保持原有多 Agent 并行合并逻辑，合并完成后作为一个增量发送，避免多个回答相互穿插。
- 流式生成成功后写入用户消息、助手回答并异步更新学习画像；客户端中断时取消上游生成，不写入不完整回答。
- 回答已经输出但记忆写入失败时只记录服务端错误，不把有效回答改判为失败。
- 保留原有 `POST /chat`，旧客户端与 PowerShell 测试命令不受影响。

### 流式接口验证

- `api/main.py`、`agents/agent_orchestrator.py` 及项目 Python 文件语法检查通过。
- `git diff --check` 未发现本次流式接口新增代码的空白错误；输出中的既有提示来自此前修改的其他文件。
- 前端生产构建通过，可识别后端 SSE 响应并显示“实时流式输出”。

### 流式输出卡顿修复

- 修复 Vue 响应式对象使用错误：助手占位消息现在从响应式数组中重新获取，SSE 文本到达时会立即触发页面更新。
- 聊天主链路的 RAG 默认改为一次向量检索，不再为每条消息额外调用 LLM 做查询改写和结果重排。
- `/search` 等检索接口保留原有增强检索能力；设置 `CHAT_RAG_MODE=enhanced` 也可以恢复聊天链路的查询改写和重排。
- 增加“流式首段已发送”和“流式回答完成”日志，记录耗时，便于区分前置检索慢、模型首字慢和前端显示问题。
- 根据 2026-08-22 15:47 的日志，原请求约 12.2 秒用于增强 RAG，Agent 完整生成约 24.7 秒；后端实际上已经完成回答并更新画像，未显示回答的直接原因是前端没有响应式刷新。

本次修改针对 `/chat` 同步链路减少不必要的等待，不引入 LangGraph。

### 1. 意图识别改为规则优先、LLM 兜底

- 明确命中单一学习意图关键词时直接返回规则结果，不再调用 LLM。
- 增加“出/来/生成……题或练习”的短句规则，覆盖“给我出一道函数参数的题”等非连续关键词表达。
- 调整规则置信度算法，单一类别命中时置信度从原来的低比例分数改为至少 `0.88`。
- 多个意图类别命中数相同时视为模糊输入，继续使用 LLM 和轻量 Embedding 判断。
- 最终置信度改为取命中意图对应来源的最高置信度。

### 2. 暂停独立实体提取

- `recognize()` 不再调用 `_extract_entities()`，避免每轮对话额外产生一次 LLM 请求。
- `IntentResult.entities` 暂时返回空字典，保留原实体提取方法供后续需要时恢复。

### 3. Memory 与 RAG 并行读取

- `/chat` 使用 `asyncio.create_task()` 同时启动 Memory 上下文读取和课程资料检索。
- 使用 `asyncio.gather()` 合并两个结果，然后构建 Agent 上下文。
- 保持原有接口请求体和响应体不变，Vue 前端不需要同步修改。

### 预期效果

- 讲解、答疑、出题等明确问题减少意图识别 LLM 调用。
- 所有聊天请求暂停实体提取 LLM 调用。
- Memory 与 RAG 的前置耗时由串行累加改为取两者较慢值。

### 验证结果

- “给我讲一下 Python 函数”“为什么这个变量访问不到？”“给我出一道函数参数的题”均由规则正确识别，置信度为 `0.88`，LLM 调用数为 `0`。
- 同分复合意图进入 LLM 兜底，LLM 调用数为 `1`。
- 实体提取调用数为 `0`，`IntentResult.entities` 返回空字典。
- Memory 与 RAG 各模拟耗时 `0.2s` 时，预热后的 `/chat` 前置阶段总耗时约 `0.201s`；若串行执行应约为 `0.4s`。
- 项目 Python 文件语法检查通过。

## 2026-08-19

本次修改围绕 EchoMind 学习助手的 RAG 主链路、长期学习画像、LLM 空回答处理和默认多轮评测数据展开。未新增 `learning_level` 字段。

### 1. 修复 RAG 主链路

- 在 `api/main.py` 的 `_should_use_knowledge()` 中为非空、非寒暄消息补充 `return True`，避免 `/chat` 永远跳过课程资料检索。
- 构建知识库上下文时跳过 fallback 结果，避免把降级提示误标记为有效知识。
- 根据文档标题和内容对多路检索结果去重，并限制最终注入的片段数量。
- 更新 RAG 提示，要求 Agent 优先依据课程资料，并在资料不足时区分课程内容与通用知识。

### 2. 调整长期薄弱点画像存储

- 在 `core/llm_utils.py` 中增加 `extract_json_value()`，用于从 LLM 输出中提取合法 JSON。
- 用户画像文档 ID 从“用户 + 会话”调整为稳定的用户级 ID：`{user_id}_profile`。
- 将 ChromaDB 的“先删除再新增”改为 `upsert()`，支持覆盖更新同一用户画像。
- 读取画像时按稳定文档 ID 查询，不再使用 `where + limit=1` 获取不确定记录。
- 画像字段保持为：
  - `learning_progress`
  - `weak_points`
  - `recent_mistakes`
  - `preferred_style`
- 本次未新增 `learning_level` 字段。

### 3. 防止空回答被记录为成功

- Agent 从 LLM 响应中提取文本后执行 `strip()` 和非空检查。
- 当 LLM 返回空内容时，记录 `stop_reason` 和内容块类型，并抛出异常进入失败处理。
- 空回答不再增加 Agent 成功次数，而是返回统一的失败提示。
- Agent 回答的 `max_tokens` 从 1024 调整为 2048，降低模型在生成正文前耗尽 token 的概率。

### 4. 修复默认多轮评测数据解析

- `EndToEndEvaluator._dialog_turns()` 现在同时支持字符串 turn 和字典 turn。
- 字典 turn 会读取 `message` 字段，不再把整个 Python 字典转换成字符串后发送给 Agent。
- 默认意图和对话用例已替换为讲解、答疑、出题、复习和寒暄等学习场景。

### 验证结果

- `GET /health`：通过。
- “给我讲一下 Python 函数”：路由到 `explain` Agent，回答非空，`knowledge_used=true`。
- “为什么这个变量访问不到？”：路由到 `qa` Agent，回答非空，`knowledge_used=true`。
- “给我出一道函数参数的题”：路由到 `quiz` Agent，回答非空，`knowledge_used=true`。
- 空 LLM 响应模拟测试：通过；结果被标记为失败，Agent 成功计数不增加。
- 默认多轮评测解析测试：通过；两个默认用例均被解析为正常的问题字符串列表。
- 固定画像 ID 与 ChromaDB `upsert`：写入链路通过。
- 薄弱点内容提取：未通过。DeepSeek 在画像更新调用中仍可能返回空内容，当前测试画像仍为四个空数组。

### 当前已知问题

- DeepSeek 的查询改写、意图识别和画像提取偶尔返回空文本或非 JSON 内容；系统能降级，但会影响画像更新和额外增加延迟。
- 当前端到端请求耗时约 29～48 秒，RAG 查询改写、意图识别、Agent 回答和画像更新包含多次 LLM 调用。
- `AgentType` 已移除 `ESCALATION`，但紧急路由仍引用 `AgentType.ESCALATION`；命中 CRITICAL 紧急度时可能抛出 `AttributeError`。
- ChromaDB telemetry 仍会输出 PostHog 参数不兼容错误，但不影响当前检索请求返回。
- 性能监控阈值仍为 3000ms，与当前模型的实际响应时间不匹配，会产生重复延迟告警。
