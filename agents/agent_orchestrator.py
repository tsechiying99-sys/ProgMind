"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.decision_engine import DecisionEngine
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content
from mcp.tool_manager import MCPToolManager, Tool
from tools.contracts import ToolExecutionContext

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL = "general"  # 通用Agent
    EXPLAIN   = "explain"    # 讲解Agent
    QA = "qa"  # 答疑Agent
    QUIZ   = "quiz"    # 出题Agent


@dataclass(frozen=True)
class AgentProfile:
    """A role contract used for routing, prompting, and tool authorization."""

    role: str
    responsibilities: Tuple[str, ...]
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    escalation_conditions: Tuple[str, ...]
    tool_scope: Tuple[str, ...]
    temperature: float = 0.2
    max_tokens: int = 4096

    def to_prompt_block(self) -> str:
        def lines(values: Tuple[str, ...]) -> str:
            return "\n".join(f"- {value}" for value in values) or "- 无"

        return (
            f"[结构化角色契约]\n"
            f"角色：{self.role}\n"
            f"职责：\n{lines(self.responsibilities)}\n"
            f"处理流程：\n{lines(self.workflow)}\n"
            f"可用输入：\n{lines(self.input_contract)}\n"
            f"输出要求：\n{lines(self.output_contract)}\n"
            f"升级条件：\n{lines(self.escalation_conditions)}\n"
            f"工具权限：{', '.join(self.tool_scope) or '无'}\n"
            "只能声称执行了实际返回结果的工具；资料不足时必须明确说明。"
        )

@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  Optional[float] = None
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 0.0
    urgency_confidence: float = 0.0
    routing_source: str = "unknown"
    needs_clarification: bool = False
    needs_rag: bool = True
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    success: bool = True
    escalated:   bool  = False
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装角色契约、工具白名单、LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[MCPToolManager] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = model
        self._skill_manager = skill_manager
        self._tool_manager = tool_manager
        self.stats   = AgentStats()
        self._last_tools_used: List[str] = []
        self._last_tool_traces: List[Dict[str, Any]] = []

    def get_tools(self) -> Dict[str, Tool]:
        """从统一管理器读取当前角色白名单内已注册的工具。"""
        if self._tool_manager is None:
            return {}
        return self._tool_manager.get_tools(list(self.profile.tool_scope))

    async def handle(self, req: Request) -> AgentResponse:
        start_time = time.monotonic()
        succeeded = False
        content = ""
        self._last_tools_used = []
        self._last_tool_traces = []
        try:
            content = await self._call_llm(req)
            succeeded = True
        except Exception as ex:
            logger.exception("%s 处理失败", self.agent_type.value)
            content = "抱歉，处理您的请求时出现问题，请稍后重试。"
        finally:
            ms = (time.monotonic() - start_time) * 1000
            self.stats.total += 1
            self.stats.total_ms += ms
            if succeeded:
                self.stats.success += 1
            escalate = self._needs_escalation(content)
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=succeeded,
            latency_ms=ms,
            escalate=escalate,
            tools_used=list(self._last_tools_used),
            tool_traces=list(self._last_tool_traces),
        )

    async def _call_llm(self, req: Request) -> str:
        """Execute a bounded Anthropic tool-use loop with server-side allowlisting."""

        messages = self._build_messages(req)
        tools = self.get_tools()
        max_rounds = max(1, int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "3")))

        for round_index in range(max_rounds + 1):
            request_kwargs = self._llm_request(
                req,
                messages=messages,
                include_tools=True,
            )
            resp = await self._client.messages.create(**request_kwargs)
            blocks = list(resp.content or [])
            tool_uses = [
                block for block in blocks
                if self._block_type(block) == "tool_use"
            ]

            if not tool_uses:
                content = extract_text_content(blocks).strip()
                if not content:
                    block_types = [
                        self._block_type(block) or type(block).__name__
                        for block in blocks
                    ]
                    logger.error(
                        "LLM 返回空回答: stop_reason=%s, blocks=%s",
                        getattr(resp, "stop_reason", None),
                        block_types,
                    )
                    raise RuntimeError("LLM 返回了空回答")
                return content

            if round_index >= max_rounds:
                raise RuntimeError(
                    f"{self.agent_type.value} Agent 工具调用超过最大轮数 {max_rounds}"
                )

            messages.append({"role": "assistant", "content": blocks})
            tool_results = []
            for block in tool_uses:
                name = str(self._block_value(block, "name") or "")
                tool_use_id = str(self._block_value(block, "id") or "")
                args = self._block_value(block, "input") or {}
                tool = tools.get(name)
                started_at = time.monotonic()
                success = False
                error = ""

                if tool is None or self._tool_manager is None:
                    error = f"工具不在 {self.agent_type.value} Agent 白名单中"
                    result: Any = {"success": False, "error": error}
                else:
                    call_result = await self._tool_manager.call(
                        name,
                        args,
                        context=ToolExecutionContext.from_request(req, self.agent_type),
                    )
                    success = call_result.success
                    error = call_result.error or ""
                    result = call_result.data if success else {
                        "success": False,
                        "error": error,
                    }
                    if success and name not in self._last_tools_used:
                        self._last_tools_used.append(name)

                if not error and isinstance(result, dict):
                    error = str(result.get("error") or "")
                self._last_tool_traces.append({
                    "agent_type": self.agent_type.value,
                    "tool_name": name,
                    "tool_use_id": tool_use_id,
                    "input": dict(args) if isinstance(args, dict) else args,
                    "success": success,
                    "latency_ms": round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                    "error": error,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                    "is_error": not success,
                })

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError("Agent 工具调用循环异常结束")

    def _build_messages(self, req: Request) -> List[Dict[str, Any]]:
        """Build a fresh message list shared by normal and streaming calls."""

        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})
        return messages

    def _llm_request(
        self,
        req: Request,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        include_tools: bool = False,
    ) -> Dict[str, Any]:
        """Build model parameters and expose only the current role's allowlist."""

        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.profile.max_tokens,
            "temperature": self.profile.temperature,
            "system": self._build_system_prompt(req),
            "messages": messages if messages is not None else self._build_messages(req),
        }
        tools = self.get_tools()
        if include_tools and tools:
            request_kwargs["tools"] = [
                tool.to_model_schema()
                for tool in tools.values()
            ]
        return request_kwargs

    async def stream(self, req: Request) -> AsyncIterator[str]:
        """Stream text using the same role contract.

        Tool use is intentionally handled by the non-streaming execution path in
        this focused change.  The stream remains behavior-compatible and never
        receives tools that it cannot safely resolve mid-stream.
        """
        start_time = time.monotonic()
        succeeded = False
        chunks: List[str] = []
        try:
            async with self._client.messages.stream(**self._llm_request(req)) as response_stream:
                async for text in response_stream.text_stream:
                    if text:
                        chunks.append(text)
                        yield text

            if not "".join(chunks).strip():
                raise RuntimeError("LLM 返回了空回答")
            succeeded = True
        finally:
            ms = (time.monotonic() - start_time) * 1000
            self.stats.total += 1
            self.stats.total_ms += ms
            if succeeded:
                self.stats.success += 1

    def _build_system_prompt(self, req: Request) -> str:
        """组合结构化角色契约、澄清约束和动态 Skills。"""

        base_prompt = f"{self.system_prompt}\n\n{self.profile.to_prompt_block()}"
        if req.needs_clarification:
            base_prompt = (
                f"{base_prompt}\n\n[决策提示]\n"
                "当前意图置信度不足。不要猜测用户目标，只提出一个最关键的澄清问题。"
            )
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    def _needs_escalation(self, content: str) -> bool:
        """检测回答是否建议教师介入。"""
        keywords = [
            "建议咨询老师",
            "需要教师确认",
            "请老师进一步指导",
            "需要人工辅导",
        ]
        return any(keyword in content for keyword in keywords)


class GeneralAgent(BaseAgent):
    agent_type = AgentType.GENERAL
    system_prompt = (
        "你是 ProgMind 的通用学习 Agent，负责问候、意图不明确、复习请求和系统降级。"
        "你不能冒充专用 Agent，也不能声称调用了未授权工具。"
    )
    profile = AgentProfile(
        role="通用学习请求接待者",
        responsibilities=(
            "处理问候、复习概述和无法可靠分类的学习请求",
            "在信息不足时提出一个最关键的澄清问题",
            "在专用 Agent 失败时提供安全、有限且诚实的通用回应",
        ),
        workflow=(
            "判断当前信息是否足以直接回答",
            "信息不足时优先澄清目标、课程范围或期望输出",
            "信息充分时给出简洁回答，并在需要时建议进入专用学习流程",
        ),
        input_contract=(
            "学生当前消息",
            "最近对话和已提供的学习上下文",
        ),
        output_contract=(
            "不猜测未提供的课程事实",
            "需要澄清时只提出一个关键问题",
            "不得宣称使用任何专用工具",
        ),
        escalation_conditions=(
            "请求涉及必须由教师判断的课程规则",
            "系统无法提供可靠答案或专用能力持续不可用",
        ),
        tool_scope=(),
        temperature=0.2,
        max_tokens=4096,
    )


class ExplainAgent(BaseAgent):
    agent_type    = AgentType.EXPLAIN
    system_prompt = "你是 ProgMind 的学习讲解 Agent，必须以课程资料和明确上下文为依据。"
    profile = AgentProfile(
        role="课程知识讲解者",
        responsibilities=(
            "根据学生当前进度和薄弱点解释知识",
            "优先依据课程资料，区分资料结论与通用补充",
            "控制单次新增概念数量并检查学生理解",
        ),
        workflow=(
            "识别需要讲解的核心知识点",
            "检查必要的前置知识",
            "按概念、示例、理解检查的顺序组织回答",
        ),
        input_contract=(
            "学生当前问题",
            "最近对话和学习画像上下文",
            "已检索的课程资料",
        ),
        output_contract=(
            "先解释核心概念",
            "提供一个难度适中的简单示例",
            "最后提出一个简短的理解检查问题",
        ),
        escalation_conditions=(
            "课程资料不足以支持确定结论",
            "学生连续多轮仍无法理解且需要教师介入",
        ),
        tool_scope=(
            "check_prerequisites",
        ),
        temperature=0.2,
        max_tokens=4096,
    )

class QaAgent(BaseAgent):
    agent_type    = AgentType.QA
    system_prompt = "你是 ProgMind 的学习答疑 Agent，优先定位卡点并提供渐进提示。"
    profile = AgentProfile(
        role="学习问题诊断与答疑者",
        responsibilities=(
            "判断学生具体卡在哪一步",
            "提供提示、推理方向和检查方法",
            "避免在练习过程中直接泄露完整答案",
        ),
        workflow=(
            "识别问题和学生已尝试的步骤",
            "定位第一个关键卡点",
            "按由弱到强的顺序提供提示",
            "信息不足时只询问一个最关键的澄清问题",
        ),
        input_contract=(
            "学生的问题或报错",
            "学生已经尝试的步骤",
            "最近对话、学习画像和课程资料",
        ),
        output_contract=(
            "先指出卡点",
            "给出可执行的下一步提示",
            "明确区分课程资料与通用解释",
        ),
        escalation_conditions=(
            "问题依赖未提供的课程或环境信息",
            "可能涉及需要教师判断的作业规范",
        ),
        tool_scope=(
            "build_hint_plan",
        ),
        temperature=0.15,
        max_tokens=4096,
    )

class QuizAgent(BaseAgent):
    agent_type    = AgentType.QUIZ
    system_prompt = "你是 ProgMind 的练习出题 Agent，默认只出一道题且不泄露答案。"
    profile = AgentProfile(
        role="个性化练习设计者",
        responsibilities=(
            "根据已学内容和薄弱点设计练习",
            "控制题目难度和考查范围",
            "在学生作答前隐藏答案和完整解法",
        ),
        workflow=(
            "确定考查知识点",
            "结合上下文确定难度和题型",
            "生成一道边界清晰的题目",
            "等待学生提交答案后再评分或讲解",
        ),
        input_contract=(
            "目标知识点",
            "学习进度和薄弱点上下文",
            "题型或难度偏好",
        ),
        output_contract=(
            "注明考查知识点和难度",
            "默认只输出一道题",
            "学生作答前不得同时输出答案或完整解法",
        ),
        escalation_conditions=(
            "课程资料没有覆盖指定知识点",
            "用户要求违反课程或考试规则的内容",
        ),
        tool_scope=(
            "search_question_bank",
            "grade_answer",
            "record_learning_result",
        ),
        temperature=0.35,
        max_tokens=3072,
    )

# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.EXPLAIN:  AgentType.EXPLAIN,
        IntentCategory.QA:    AgentType.QA,
        IntentCategory.QUIZ:    AgentType.QUIZ,
        IntentCategory.REVIEW: AgentType.GENERAL,
        IntentCategory.GREETING: AgentType.GENERAL,
        IntentCategory.OTHER: AgentType.GENERAL,
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[MCPToolManager] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._decision_engine = decision_engine or DecisionEngine(self._intent_recognizer)
        self._skill_manager = skill_manager
        self._tool_manager = tool_manager

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [GeneralAgent(client, model, skill_manager, tool_manager)],
            AgentType.EXPLAIN: [ExplainAgent(client, model, skill_manager, tool_manager)],
            AgentType.QA: [QaAgent(client, model, skill_manager, tool_manager)],
            AgentType.QUIZ: [QuizAgent(client, model, skill_manager, tool_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_tool_manager(self, tool_manager: Optional[MCPToolManager]) -> None:
        """更新统一工具管理器，主要供测试替换和运行时重载。"""
        self._tool_manager = tool_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._tool_manager = tool_manager

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 生成或复用统一路由决策。
        await self._ensure_decision(req)

        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = (
            AgentType.GENERAL
            if req.needs_clarification
            else self._route(req.intent, req.urgency)
        )

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type)

        # 4. 升级检查
        escalated = response.escalate or req.urgency == UrgencyLevel.CRITICAL
        if escalated:
            logger.warning("学习请求 %s 建议人工关注: urgency=%s",req.request_id,req.urgency,)
            # 生产环境：此处创建工单、通知人工客服

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            # 用于判断该编排结果是否成功
            success=response.success,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        successful_responses = [
            response
            for response in responses
            if isinstance(response, AgentResponse) and response.success
        ]

        parts = [
            f"[{response.agent_type.value}]\n{response.content}"
            for response in successful_responses
        ]

        fallback_response: Optional[AgentResponse] = None
        if not successful_responses:
            fallback_response = await self._execute(req, AgentType.GENERAL)
            if fallback_response.success:
                successful_responses = [fallback_response]
                parts = [fallback_response.content]

        combined = "\n\n".join(parts) if parts else "学习服务暂时不可用，请稍后重试。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in successful_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in successful_responses
            for trace in response.tool_traces
        ]

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=(
                fallback_response.agent_type
                if fallback_response is not None and fallback_response.success
                else agent_types[0]
            ),
            intent=req.intent,
            #用于判断该编排结果是否成功
            success=bool(successful_responses),
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=tools_used,
            tool_traces=tool_traces,
        )

    async def stream(self, req: Request) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行一次请求。

        单 Agent 直接转发模型 token；复合意图继续使用原并行编排，完成后作为
        一个 delta 返回，避免多个 Agent 的 token 相互穿插破坏可读性。
        """
        t0 = time.monotonic()
        await self._ensure_decision(req)

        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            yield {
                "type": "meta",
                "intent": req.intent.value if req.intent else "other",
                "agent_type": collaboration[0].value,
            }
            result = await self.run_parallel(req, collaboration)
            yield {"type": "delta", "text": result.response}
            yield {"type": "done", "result": result}
            return

        requested_type = (
            AgentType.GENERAL
            if req.needs_clarification
            else self._route(req.intent, req.urgency)
        )
        agent = self._best_agent(requested_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            raise RuntimeError("学习服务暂时不可用，请稍后重试")

        yield {
            "type": "meta",
            "intent": req.intent.value if req.intent else "other",
            "agent_type": agent.agent_type.value,
        }

        chunks: List[str] = []
        try:
            async for text in agent.stream(req):
                chunks.append(text)
                yield {"type": "delta", "text": text}
        except Exception:
            # 已输出正文后不能安全切换；通用 Agent 自身失败也不再递归降级。
            if chunks or requested_type == AgentType.GENERAL:
                raise
            logger.warning("%s 流式处理失败，降级到 GeneralAgent", requested_type.value)
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback is None or fallback is agent:
                raise
            agent = fallback
            async for text in agent.stream(req):
                chunks.append(text)
                yield {"type": "delta", "text": text}

        content = "".join(chunks).strip()
        if not content:
            raise RuntimeError("LLM 返回了空回答")

        escalated = (
            agent._needs_escalation(content)
            or req.urgency == UrgencyLevel.CRITICAL
        )
        result = OrchestratorResult(
            request_id=req.request_id,
            response=content,
            agent_type=agent.agent_type,
            intent=req.intent,
            success=True,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        yield {"type": "done", "result": result}

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    async def _ensure_decision(self, req: Request) -> None:
        """当调用方未提供决策时，统一补齐路由及门控字段。"""
        if req.intent is not None:
            return

        decision = await self._decision_engine.decide(
            req.message,
            history=req.history,
            request_id=req.request_id,
        )
        req.intent = decision.intent
        req.urgency = decision.urgency
        req.intent_confidence = decision.intent_confidence
        req.urgency_confidence = decision.urgency_confidence
        req.routing_source = decision.source
        req.needs_clarification = decision.needs_clarification
        threshold = float(os.getenv("RAG_DECISION_THRESHOLD", "0.55"))
        req.needs_rag = decision.should_use_rag(threshold)

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        根据学习意图选择 Agent。
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.GENERAL

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。
        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        """
        if req.needs_clarification:
            return []

        msg = req.message.lower()
        targets: List[AgentType] = []

        explain_keywords = [
            "解释", "讲一下", "什么是", "原理", "概念",
        ]
        qa_keywords = [
            "为什么", "哪里错", "不理解", "怎么改", "报错",
        ]
        quiz_keywords = [
            "出题", "练习", "考考我", "测验",
        ]

        if (req.intent == IntentCategory.EXPLAIN or
            any(keyword in msg for keyword in explain_keywords)
        ):
            targets.append(AgentType.EXPLAIN)

        if (req.intent == IntentCategory.QA
            or any(keyword in msg for keyword in qa_keywords)
        ):
            targets.append(AgentType.QA)

        if (req.intent == IntentCategory.QUIZ
            or any(keyword in msg for keyword in quiz_keywords)
        ):
            targets.append(AgentType.QUIZ)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。

        优先选择健康 Agent。
        如果该类型只有一个实例，或者所有实例都被 Monitor 降权，
        仍返回评分最高的实例；是否真正不可用应由实际执行结果决定。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        healthy = [
            agent for agent in agents
            if agent.stats.monitor_penalty < 0.7
        ]
        candidate = healthy if healthy else agents

        return max(candidate, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="学习服务暂时不可用，请稍后重试",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时统一降级到通用 Agent。
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning("%s 失败，降级到 GeneralAgent", agent_type.value)
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
