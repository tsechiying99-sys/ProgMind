# Jev 决策与 GeneralAgent 改造日志

## 一、版本基线与同步状态

- 修改前已将旧版本提交为 `574795fc96fb32869eff27e4bd01d78ac1d5a406`，提交说明为 `Unify structured agent tools before Jev rollout`。
- 旧版本已推送到 `https://github.com/tsechiying99-sys/ProgMind.git` 的 `main` 分支。
- 已核对远端 `main` 与本地修改前提交一致。
- 本日志所列 Jev 与 GeneralAgent 修改仅保留在本地工作区，尚未提交或推送到 GitHub。
- 未修改或加入现有未跟踪目录 `output/`、`tmp/`。

## 二、实施边界

本次严格实施到第五阶段：

1. 引入统一决策数据结构，并移除回答置信度的 `1.0` 默认值。
2. 接入 Jev 意图路由影子模式。
3. 增加意图置信度门控和澄清路径。
4. 将普通聊天 RAG 改为条件调用。
5. 增加 RAG 证据准入影子模式。

同时新增 GeneralAgent，并把原来指向 ExplainAgent 的通用路由和失败降级统一改为 GeneralAgent。未实施第六阶段“正式过滤 RAG 证据”、第七阶段“多维答案评分”和第八阶段“评分凭证与可信写入”。

## 三、GeneralAgent 与降级链路

### 3.1 新增通用角色

- 在 `AgentType` 中新增 `GENERAL`。
- 新增 `GeneralAgent`，使用完整 `AgentProfile` 声明角色、职责、处理流程、输入输出契约和升级条件。
- GeneralAgent 的 `tool_scope` 为空，不获得 Explain、QA、Quiz 的任何专用工具权限。
- GeneralAgent 负责：
  - 问候请求；
  - 复习概述请求；
  - 无法可靠分类的请求；
  - 低置信度澄清；
  - 专用 Agent 失败后的有限通用回应。

### 3.2 路由与降级变化

- `REVIEW`、`GREETING`、`OTHER` 统一路由到 GeneralAgent。
- 未命中路由表、目标 Agent 缺失、严重请求的安全兜底均改为 GeneralAgent。
- Explain、QA、Quiz 执行失败时改为降级到 GeneralAgent。
- 流式响应在尚未输出正文时允许降级到 GeneralAgent；已经输出正文后不切换 Agent，避免混合响应。
- GeneralAgent 自身失败时不再递归降级，防止循环。
- 低置信度请求强制进入 GeneralAgent，不再根据关键词误触发多 Agent 并行协作。

## 四、阶段一：统一决策数据结构

- 新增 `core/decision_models.py`：
  - `RoutingDecision`：保存生产意图、意图置信度、紧急程度、RAG 概率、是否澄清、来源和耗时。
  - `JevShadowDecision`：保存 Jev 影子意图、概率、置信度和调用状态。
  - `EvidenceShadowDecision`：保存单个 RAG 片段的影子准入指标。
- `AgentResponse.confidence` 从固定默认 `1.0` 改为 `Optional[float] = None`，没有真实计算时不再伪装为满置信度。
- `Request` 新增 `intent_confidence`、`urgency_confidence`、`routing_source`、`needs_clarification`、`needs_rag` 字段。

## 五、阶段二：Jev 意图路由影子模式

- 新增 `core/jev_client.py`，使用项目已有 `httpx` 调用 TypeSafe System One API。
- 未配置 `TYPESAFE_API_KEY` 时不创建 HTTP 客户端，也不会发送网络请求。
- 新增 `core/decision_engine.py`，继续使用现有 `IntentRecognizer` 产生正式路由，同时可并行请求 Jev。
- Jev 一次评估四项结构化问题：
  - `intent`：Choice；
  - `urgency`：Score；
  - `needs_rag`：Noul；
  - `needs_clarification`：Noul。
- Jev 结果只写入影子决策和日志，不覆盖现有识别器的正式结果。
- 日志只记录请求编号、决策类型和置信度，不记录用户原始消息。
- 已依据 TypeSafe 官方 OpenAPI 0.2.0 核对 `/v1/systemone`、Bearer 鉴权和三类问题结构。

## 六、阶段三：置信度门控与澄清路径

- 默认正式意图置信度阈值为 `0.55`，可通过 `INTENT_MIN_CONFIDENCE` 调整。
- 正式意图为 `OTHER` 或置信度低于阈值时，设置 `needs_clarification=True`。
- 澄清请求不执行 RAG、不触发多个专用 Agent，并路由到 GeneralAgent。
- 系统提示中加入明确约束：不猜测用户目标，只提出一个最关键的澄清问题。

## 七、阶段四：条件 RAG

- `_prepare_chat_request()` 改为先并行读取 Memory 和统一路由决策。
- 只有 `RoutingDecision.should_use_rag()` 通过门控后才调用课程知识库。
- 问候和低置信度澄清默认跳过 RAG，减少无意义检索及延迟。
- Explain、QA、Quiz、Review 的生产 RAG 概率暂为 `1.0`；Greeting 为 `0.0`；Other 为 `0.25`。
- 默认 RAG 决策阈值为 `0.55`，可通过 `RAG_DECISION_THRESHOLD` 调整。
- Memory 仍是固定预处理，不作为 Agent 工具重复调用。

## 八、阶段五：RAG 证据准入影子模式

- 新增 `core/evidence_gate.py`，对候选片段并发评估：
  - 与问题是否相关；
  - 是否包含直接回答证据；
  - 是否与问题前提冲突；
  - 是否包含提示注入迹象。
- 影子分类包括 `evidence`、`conflict`、`drop_injection`、`drop_irrelevant` 和 `unavailable`。
- 当前只记录分类数量、请求编号和查询哈希；不记录查询正文。
- 当前影子结果不会过滤、重排或改写正式 RAG 返回值，符合第五阶段边界。
- `RAG_EVIDENCE_SHADOW_TOP_K` 限制影子评估片段数，信号量限制并发调用数。

## 九、配置项

`.env.example` 新增以下配置：

```dotenv
TYPESAFE_API_KEY=
JEV_MODEL=jev-latest
JEV_BASE_URL=https://api.typesafe.ai
JEV_TIMEOUT_S=3
JEV_SHADOW_ENABLED=false
RAG_EVIDENCE_SHADOW_ENABLED=false
RAG_EVIDENCE_SHADOW_TOP_K=6
RAG_EVIDENCE_SHADOW_CONCURRENCY=4
INTENT_MIN_CONFIDENCE=0.55
RAG_DECISION_THRESHOLD=0.55
```

两个 Jev 影子功能默认关闭。启用时还必须提供有效的 `TYPESAFE_API_KEY`。

## 十、涉及文件

### 修改

- `.env.example`
- `agents/agent_orchestrator.py`
- `api/main.py`
- `tests/test_agent_profiles.py`
- `tests/test_chat_rag.py`

### 新增

- `core/decision_models.py`
- `core/decision_engine.py`
- `core/jev_client.py`
- `core/evidence_gate.py`
- `tests/test_decision_engine.py`
- `tests/test_evidence_gate.py`
- `tests/test_general_agent.py`
- `CHANGELOG_Jev_2026-10-05.md`

## 十一、验证结果

执行命令：

```powershell
.\.venv\Scripts\python.exe -m compileall agents api core tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

验证结果：

- Python 编译检查通过。
- 25 个单元测试全部通过。
- 已覆盖结构化角色、GeneralAgent 空工具白名单、专用 Agent 失败降级、低置信度澄清、条件 RAG、Jev 不改变生产路由以及证据影子模式不影响正式检索结果。

## 十二、当前风险与后续边界

- Jev 属于外部服务；本地单元测试使用 Fake Client 验证协议处理，没有使用真实密钥产生外部调用或费用。
- 影子模式启用后会把当前消息及最多三条最近历史发送给 TypeSafe，生产启用前应完成数据合规确认。
- 阈值 `0.55`、证据分类阈值和 RAG 概率是首轮工程默认值，需要通过离线评测校准后才能进入正式过滤。
- 第六至第八阶段尚未实施，因此 Jev 当前不会决定正式证据去留、正式评分或学习结果写入。
