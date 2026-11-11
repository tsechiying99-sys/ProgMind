# 多 Agent 语义路由与答案整合修改日志

## 一、修改日期与范围

- 修改日期：2026-10-19
- 本次修改严格围绕三项内容：
  1. 移除项目中已经接入的 Jev 影子能力；
  2. 将多 Agent 判断从关键词匹配改为结构化语义规划；
  3. 将多 Agent 原始结果拼接改为统一答案整合。
- 未修改用户当前在 `mcp/knowledge_base.py` 中尚未提交的改动。

## 二、移除 Jev

### 删除的文件

- `core/jev_client.py`
- `core/evidence_gate.py`
- `tests/test_evidence_gate.py`
- `CHANGELOG_Jev_2026-10-05.md`

### 清理的运行入口

- `api/main.py` 不再创建或关闭 Jev 客户端。
- 删除 Jev 意图路由影子调用。
- 删除 RAG 片段的 Jev 影子准入调用。
- `core/decision_models.py` 删除 `JevShadowDecision` 和
  `EvidenceShadowDecision`。
- `.env.example` 删除全部 `TYPESAFE_*`、`JEV_*` 和
  `RAG_EVIDENCE_SHADOW_*` 配置。

移除后，项目不再需要 TypeSafe/Jev 密钥，也不会向 Jev 服务发送请求。

## 三、多 Agent 语义规划

### 新增结构化模型

`core/decision_models.py` 新增：

- `CollaborationMode`：支持 `single`、`parallel`、`sequential`；
- `AgentTask`：保存目标角色、明确的子任务指令和依赖关系；
- `CollaborationPlan`：保存执行模式、任务列表、置信度和判断原因。

### 决策方式

`core/decision_engine.py` 新增 `SemanticCollaborationPlanner`，使用项目当前
配置的主模型判断是否需要多个 Agent。规划器会结合：

- 当前请求的完整语义；
- 最近四条对话历史；
- 主意图识别结果；
- 否定表达和用户限定范围；
- 子任务之间是否存在前后依赖。

模型只能从 `general`、`explain`、`qa`、`quiz` 中选择角色，并输出 JSON
结构化计划。服务端继续执行二次校验：

- 最多两个不同角色；
- `general` 不与专用角色组成多 Agent；
- 重复角色会被过滤；
- 非法角色会被过滤；
- 置信度低于 `COLLABORATION_MIN_CONFIDENCE` 时退回单 Agent；
- 低意图置信度请求优先澄清，不启动多 Agent；
- 规划器异常时保守退回主意图对应的单 Agent。

### 并行与串行

- `parallel`：两个互不依赖的子任务并行执行；
- `sequential`：后一个子任务会收到前一个成功结果作为前置上下文；
- 每个 Agent 只收到自己的子任务指令，不再让所有 Agent 重复处理整段请求。

`api/main.py` 调整了准备顺序：先读取 Memory，再把最近对话传给决策引擎，
使“继续讲”“再出一道类似的题”等省略表达可以结合上下文规划。

## 四、多 Agent 答案整合

`agents/agent_orchestrator.py` 新增统一答案整合步骤：

1. 收集成功的子任务回答；
2. 将用户原始请求、各子任务和对应回答交给当前主模型；
3. 删除重复内容；
4. 对冲突内容优先保留有依据且更谨慎的说法；
5. 输出一份连贯答案，不暴露内部 Agent 名称和路由过程。

如果答案整合调用失败，系统会执行本地段落规范化和去重，而不是直接把带角色
标签的原始回答拼接给用户。只有一个子任务成功时，直接返回该成功结果。

## 五、配置变化

新增环境变量：

```env
COLLABORATION_MIN_CONFIDENCE=0.70
```

该值用于控制语义规划结果启用多 Agent 的最低置信度。

## 六、测试变化

- 重写 `tests/test_decision_engine.py`，覆盖语义计划、历史上下文、低置信度
  澄清、RAG 门控和非法计划收缩。
- 新增 `tests/test_agent_collaboration.py`，覆盖：
  - 不依赖关键词的结构化多 Agent 计划；
  - 低协作置信度退回单 Agent；
  - 多 Agent 结果必须经过答案整合；
  - 串行任务能够接收前置任务结果。
- `tests/test_chat_rag.py` 删除 Jev 证据影子测试，并适配新的决策结构。

## 七、兼容与降级说明

- 原有主意图识别、GeneralAgent 降级和工具白名单保持不变。
- 语义规划器失败不会阻断正常回答，而是降级为单 Agent。
- 答案整合器失败不会让整轮请求失败，而是使用本地去重结果。
- 当前普通多 Agent 流式请求仍会等待协作和整合完成后统一输出，后续可单独增加
  `planning`、`agent_done`、`synthesizing` 等 SSE 进度事件。

## 八、验证结果

- Python 静态编译检查通过。
- `git diff --check` 通过。
- 完整单元测试结果：`30 passed`。
- pytest 仅提示当前受限环境无法写入 `.pytest_cache`，不影响测试执行和结果。
