"""
核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 对召回结果用 LLM 打分，按相关性重新排序，
     解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content,extract_json_value

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常使用（熔断关闭）
    OPEN      = "open"       # 熔断开启，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    """工具调用后的结果"""
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        """平均耗时"""
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None #开启的时间

    def allow(self) -> bool:
        """
        在调用工具前判断是否允许真实请求通过。
        熔断器为closed状态说明可以使用工具，open状态不可以使用工具，但如果监管时间-开启时间>=恢复时间进行探测
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any
    read_only:   bool = True                 # 供审计和后续权限策略使用

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)

    def to_model_schema(self) -> Dict[str, Any]:
        """转换为 Anthropic tool-use 所需的公开定义。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at, reranked)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> Optional[Tool]:
        """只读获取已注册工具，避免调用方访问内部字典。"""
        return self._tools.get(name)

    def get_tools(self, names: Optional[List[str]] = None) -> Dict[str, Tool]:
        """获取指定工具；省略 names 时返回全部已注册工具的浅拷贝。"""
        if names is None:
            return dict(self._tools)
        return {name: self._tools[name] for name in names if name in self._tools}

    def get_tool_schemas(self, names: List[str]) -> List[Dict[str, Any]]:
        return [tool.to_model_schema() for tool in self.get_tools(names).values()]

    @property
    def registered_names(self) -> Tuple[str, ...]:
        return tuple(self._tools)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Any = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # >0 时对结果重排，取 Top-K
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 可选重排 → 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 调用方输入错误不是工具故障：不应触发 fallback、失败计数或熔断器。
        try:
            self._validate_params(tool, params)
        except (TypeError, ValueError) as ex:
            return ToolResult(success=False, data=None, tool_name=name, error=str(ex))

        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            data = await asyncio.wait_for(
                self._run_handler(tool.handler, params, context),
                timeout=tool.timeout_s,
            )
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            # 重排（针对返回列表的检索工具）
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

            # 写缓存：缓存最终返回结果，避免下次命中未重排的原始结果。
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl, cache_rerank_top_k, reranked)

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Any,
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    @staticmethod
    async def _run_handler(handler: Callable, params: Dict[str, Any], context: Any) -> Any:
        """统一执行同步/异步 handler；同步 I/O 不阻塞事件循环。"""
        if inspect.iscoroutinefunction(handler):
            return await handler(params, context)
        result = await asyncio.to_thread(handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        query = self._clean_text(query).strip()
        if not query:
            return []
        prompt = self._clean_text(f"""
        请将用户查询改写为最多 {n} 个适合知识库检索的中文子查询。

        要求：
        1. 保留原始查询中的人名、术语、数字和专有名词。
        2. 不要添加原查询不存在的事实。
        3. 每个查询必须是非空字符串。
        4. 只返回合法 JSON 数组，不要解释，不要 Markdown 代码块。

        用户查询：{json.dumps(query, ensure_ascii=False)}

        返回格式：
        ["子查询1", "子查询2", "子查询3"]
        """)

        last_error: Optional[Exception] = None

        # 第一次可能因为模型输出为空或格式错误失败，因此允许重试一次。
        for attempt in range(2):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                raw = extract_text_content(resp.content).strip()
                queries = extract_json_value(raw, expected_type=list)

                cleaned_queries: List[str] = []
                for item in queries:
                    if not isinstance(item, str):
                        continue
                    rewritten = self._clean_text(item).strip()
                    if not rewritten or len(rewritten) > 200:
                        continue
                    cleaned_queries.append(rewritten)

                # 始终保留原始查询，并保持顺序去重。
                result = list(dict.fromkeys([query, *cleaned_queries]))
                if result:
                    return result[: n + 1]

                raise ValueError("查询改写结果中没有有效字符串")
            except Exception as ex:
                last_error = ex
                logger.warning(
                    "查询改写第 %d 次失败: query=%r error=%s",
                    attempt + 1,
                    query,
                    ex,
                )
        logger.warning(
            "查询改写最终失败，使用原始查询: query=%r error=%s",
            query,
            last_error,
        )
        return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Any = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排 → Top-K

        这是解决"检索不全、召回不好"的完整方案。
        """
        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索
        recall_k = max(top_k, 5)
        # 强制保证原始问题位于第一项
        ordered_queries = [query]

        for rewritten_query in sub_queries:
            if rewritten_query not in ordered_queries:
                ordered_queries.append(rewritten_query)

        tasks = [
            self.call(
                tool_name,
                {
                    "query": current_query,
                    "top_k": recall_k,
                    # 原问题使用 BM25+BGE，改写问题只使用 BGE
                    "retrieval_mode": (
                        "hybrid"
                        if index == 0
                        else "dense"
                    ),
                },
                context,
                use_cache=True,
            )
            for index, current_query in enumerate(
                ordered_queries
            )
        ]
        #一个子查询异常不会取消其他子查询。
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 合并去重（）
        # 按文档身份去重，不把 score 放入去重键
        merged_by_key: Dict[tuple, Dict[str, Any]] = {}

        for result in results:
            if not (
                    isinstance(result, ToolResult)
                    and result.success
                    and isinstance(result.data, list)
            ):
                continue

            for item in result.data:
                if not isinstance(item, dict):
                    continue

                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                chunk = item.get("chunk", 0)

                if not content:
                    continue

                document_key = (
                    title,
                    content,
                    chunk,
                )

                existing = merged_by_key.get(document_key)

                # 同一文档被多个子查询召回时，只保留最高分版本
                if existing is None:
                    merged_by_key[document_key] = item
                    continue

                current_score = float(item.get("score", 0.0))
                existing_score = float(
                    existing.get("score", 0.0)
                )

                if current_score > existing_score:
                    merged_by_key[document_key] = item

        merged = list(merged_by_key.values())

        if not merged:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="所有子查询均无结果")

        # 4. 重排：用 LLM 对合并结果按相关性打分，取 Top-K
        reranked = await self._rerank(query, merged, top_k)
        return ToolResult(success=True, data=reranked, tool_name=tool_name, reranked=True)

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(
            self,
            query: str,
            items: List[Any],
            top_k: int,
    ) -> List[Any]:
        """
        对候选结果进行相关性评分。
        """
        valid_items = [
            item
            for item in items
            if isinstance(item, dict)
               and float(item.get("score", 0.0)) > 0.0
        ]

        return sorted(
            valid_items,
            key=lambda item: float(
                item.get(
                    "fusion_score",
                    item.get("score", 0.0),
                )
            ),
            reverse=True,
        )[:top_k]

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict, rerank_top_k: int = 0) -> str:
        """生成缓存key:不同插入顺序的同内容字典生成相同缓存 Key"""
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict, rerank_top_k: int = 0) -> Optional[Tuple[Any, bool]]:
        """获取缓存，如果没有到期返回内容和重排等级；如果到期删除缓存"""
        key = self._cache_key(name, params, rerank_top_k)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        """设置缓存（用一个字典来实现）"""
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params, rerank_top_k)] = (data, time.monotonic() + ttl, reranked)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        if not isinstance(params, dict):
            raise ValueError(f"工具 {tool.name} 参数必须是 JSON 对象")

        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        unknown = set(params) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"工具 {tool.name} 不允许参数: {', '.join(sorted(unknown))}")

        for key, value in params.items():
            field_schema = properties.get(key)
            if field_schema:
                self._validate_value(tool.name, key, value, field_schema)

    def _validate_value(
        self,
        tool_name: str,
        field_name: str,
        value: Any,
        schema: Dict[str, Any],
    ) -> None:
        expected_name = schema.get("type")
        expected_type = self._TYPE_MAP.get(expected_name)
        if expected_type is not None:
            if expected_name in {"number", "integer"} and isinstance(value, bool):
                raise ValueError(f"工具 {tool_name} 参数 {field_name} 类型错误: 期望 {expected_name}")
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"工具 {tool_name} 参数 {field_name} 类型错误: 期望 {expected_name}，实际 {type(value).__name__}"
                )

        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"工具 {tool_name} 参数 {field_name} 不在允许值中")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"工具 {tool_name} 参数 {field_name} 小于最小值 {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"工具 {tool_name} 参数 {field_name} 大于最大值 {schema['maximum']}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise ValueError(f"工具 {tool_name} 参数 {field_name} 过短")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                raise ValueError(f"工具 {tool_name} 参数 {field_name} 过长")
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                self._validate_value(tool_name, f"{field_name}[{index}]", item, schema["items"])

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }

    # ── 对缓存进行清理 ──────────────────────────────────────────────────────────────────
    def clear_cache(
            self,
            tool_name: str | None = None,
    ) -> None:
        if tool_name is None:
            self._cache.clear()
            return

        prefix = f"{tool_name}:"

        for key in list(self._cache):
            if key.startswith(prefix):
                del self._cache[key]
