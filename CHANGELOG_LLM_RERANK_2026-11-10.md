# RRF 融合后 LLM 重排修改日志

日期：2026-11-10
范围：聊天 RAG、增强检索、题目检索工具、检索响应元数据
状态：已实现

## 一、修改目标

在 BGE Dense 与 BM25 完成加权 RRF 融合后，引入真正的 LLM rerank。原有
`MCPToolManager._rerank()` 只会再次按照 `fusion_score` 排序，没有调用模型，
因此不能根据用户问题判断候选片段的直接回答价值。

修改后的主链路：

```text
用户问题
  → BGE Dense 召回
  → BM25 关键词召回
  → 加权 RRF 融合与初排
  → 选取最多 10 个候选
  → LLM 相关性重排
  → 最终 Top-K
```

## 二、核心实现

### `mcp/tool_manager.py`

- 将 `_rerank()` 改为真实异步 LLM 重排。
- 新增公开入口 `rerank_results()`，供聊天 RAG、搜索接口和工具统一复用。
- 发送给模型的候选数据仅包含索引、标题、截断正文和 RRF 分数。
- 要求模型只返回候选索引、0~1 相关性分数和简短理由，禁止改写候选正文。
- 校验所有索引必须合法、唯一且完整，防止模型遗漏或虚构候选。
- 最终结果保留原始 `fusion_score`，并新增：
  - `rrf_rank`
  - `rrf_score`
  - `rerank_rank`
  - `rerank_score`
  - `rerank_reason`
  - `rerank_applied`
  - `reranked_by`
- `score` 更新为最终 LLM 重排分数，前端和下游默认展示最终排序依据。
- LLM 返回空内容、非法 JSON、索引不完整、超时或网络异常时重试一次。
- 两次失败后自动退回原 RRF 顺序，不中断聊天和题目检索。

### `api/main.py`

- `_search_course_knowledge()` 在合并和按 RRF 分数初排后调用统一重排器。
- 默认先召回 10 个候选，而不是只召回最终 Top-K，为重排保留选择空间。
- fast 模式只省略查询改写，不再省略 RRF 后 LLM 重排。
- enhanced 模式完成多查询召回和去重后，再统一执行一次 LLM 重排，避免重复调用。
- `/search` 的 `reranked` 字段改为依据真实 `rerank_applied` 返回，不再固定为 `true`。
- RAG 上下文中的相关度优先展示 `rerank_score`。

### `tools/search_question_bank.py`

- 题目检索先执行知识库 Dense + BM25 + RRF。
- 筛出明确标记为题目、练习或习题的候选后，再调用统一 LLM 重排器。
- QuizAgent 与聊天 RAG 使用相同的重排实现和失败回退规则。

### `tools/registry.py`

- 注册题目检索工具时注入 `MCPToolManager`，避免工具自行创建第二套模型客户端。

## 三、安全与稳定性

- 候选正文被明确标记为待评估数据，模型需要忽略其中的命令和提示词。
- LLM 只能通过已有候选的整数索引进行排序，不能生成新的知识片段。
- 候选正文默认最多发送 1000 字，候选数量默认最多 10 条，控制 token 消耗。
- 单次调用默认 30 秒超时。
- 重排失败不会删除召回结果，而是回退到经过验证的 RRF 顺序。

## 四、新增配置

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `RAG_LLM_RERANK_ENABLED` | `true` | 是否启用 RRF 后 LLM 重排 |
| `RAG_RERANK_MAX_CANDIDATES` | `10` | 送入 LLM 的最大候选数 |
| `RAG_RERANK_CONTENT_CHARS` | `1000` | 每个候选正文最大字符数 |
| `RAG_RERANK_TIMEOUT_S` | `30` | 单次 LLM 重排超时秒数 |

## 五、测试覆盖

- 验证 LLM 可以改变原始 RRF 顺序。
- 验证重排只改变顺序和分数，不改写候选正文。
- 验证保留原 RRF 排名和分数。
- 验证非法模型输出连续失败后回退 RRF。
- 验证聊天 RAG 将 RRF 初排后的候选传入重排器。
- 验证题目检索工具在筛选题目候选后执行统一重排。
- 验证 `/search` 只在真实 LLM 重排成功时报告 `reranked=true`。

## 六、性能影响

每次成功使用 RAG 的请求默认增加一次 LLM 调用。相比仅使用 RRF，相关性判断更强，
但首字延迟和模型费用会增加。可以通过关闭 `RAG_LLM_RERANK_ENABLED` 立即回退到
纯 RRF，也可以减少候选数或正文字符数来控制成本。
