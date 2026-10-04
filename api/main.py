"""
ProgMind 智能学习系统 — FastAPI 入口


所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import json
import logging
import os
import pathlib
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

#保证可以跨包import
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel,Field

load_dotenv()

#初始化项目日志系统，并创建当前模块的 logger，让 ProgMind 在运行时能够按照统一格式输出运行信息。
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
   ╔══════════════════════╗
   ║   ProgMind  v2.0     ║
       智能学习助手       
   ╚══════════════════════╝
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_knowledge_base = None
_monitor      = None
_evaluator    = None
_skill_manager = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _knowledge_base, _monitor, _evaluator, _skill_manager

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from tools import ToolDependencies, register_learning_tools

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    _knowledge_base = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {_knowledge_base.doc_count} 个文档片段")
    register_learning_tools(
        _tool_manager,
        ToolDependencies(
            knowledge_base=_knowledge_base,
            memory_manager=_memory,
        ),
    )

    # Agent 只通过 MCPToolManager 获取和执行白名单工具。
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        tool_manager=_tool_manager,
    )

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        chat_runner=_run_eval_chat,
    )

    logger.info("EchoMind 已就绪")
    yield

    await _monitor.stop()
    logger.info("EchoMind 已关闭")

#评测调用
async def _run_eval_chat(
        *,
        message: str,
        user_id: str,
        conv_id: str,
        wait_for_profile: bool = False,
) -> Dict[str, Any]:
    req = ChatRequest(
        message=message,
        user_id=user_id,
        conv_id=conv_id,
    )

    return await _execute_chat(
        req,
        conv_id=conv_id,
        profile_mode="wait" if wait_for_profile else "skip",
    )

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoMind 智能客服",
    version="2.0.0",
    #Fast API的生命周期管理函数
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    #Agent的回答
    response:    str
    #识别出的意图
    intent:      str
    agent_type:  str
    #是否需要人工升级
    escalated:   bool
    #Agent执行耗时
    latency_ms:  float
    #是否需要使用知识库
    knowledge_used: bool = False


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()

async def _execute_chat(
    req: ChatRequest,
    *,
    conv_id: Optional[str] = None,
    profile_mode: str = "background",
) -> Dict[str, Any]:
    """
    执行完整聊天链路。

    profile_mode:
      background: 正常接口使用，后台更新画像
      wait:        评测使用，等待画像更新完成
      skip:        不更新画像
    """
    from memory.conversation_memory import MsgRole

    if _orchestrator is None or _memory is None:
        raise RuntimeError("服务未就绪")

    actual_conv_id = conv_id or req.conv_id or str(uuid.uuid4())

    # 真实 Memory + RAG 准备过程
    orch_req, knowledge_used = await _prepare_chat_request(
        req,
        actual_conv_id,
    )

    # 真实意图识别、路由和 Agent 执行
    result = await _orchestrator.run(orch_req)

    # 与生产接口相同的记忆写入
    await _memory.add_message(
        req.user_id,
        actual_conv_id,
        MsgRole.USER,
        req.message,
    )
    await _memory.add_message(
        req.user_id,
        actual_conv_id,
        MsgRole.ASSISTANT,
        result.response,
    )

    profile_updated: Optional[bool] = None

    if profile_mode == "wait":
        profile_updated = await _memory.update_profile(
            req.user_id,
            actual_conv_id,
        )
    elif profile_mode == "background":
        asyncio.create_task(
            _memory.update_profile(req.user_id, actual_conv_id)
        )

    profile = (
        await _memory.get_profile(req.user_id)
        if profile_mode == "wait"
        else {}
    )

    return {
        "conv_id": actual_conv_id,
        "response": result.response,
        "intent": result.intent.value if result.intent else "other",
        "agent_type": result.agent_type.value,
        "success": result.success,
        "escalated": result.escalated,
        "latency_ms": round(result.latency_ms, 1),
        "knowledge_used": knowledge_used,

        # 仅供内部评测，不必返回给正常前端
        "context": orch_req.context,
        "profile_updated": profile_updated,
        "profile": profile,
    }
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    try:
        result = await _execute_chat(req, profile_mode="background")
    except RuntimeError as ex:
        raise HTTPException(503, str(ex)) from ex

    return ChatResponse(
        conv_id=result["conv_id"],
        response=result["response"],
        intent=result["intent"],
        agent_type=result["agent_type"],
        escalated=result["escalated"],
        latency_ms=result["latency_ms"],
        knowledge_used=result["knowledge_used"],
    )


@app.post("/chat/stream", tags=["对话"])
async def chat_stream(req: ChatRequest):
    """SSE 流式对话接口；业务流程与 /chat 保持一致。"""
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    async def event_generator():
        # 尽快建立 SSE 连接；该注释事件不会被前端当成正文。
        yield ": connected\n\n"
        started_at = time.monotonic()
        first_delta_sent = False
        try:
            orch_req, knowledge_used = await _prepare_chat_request(req, conv_id)

            async for event in _orchestrator.stream(orch_req):
                event_type = event.get("type")
                if event_type == "meta":
                    yield _sse_event("meta", {
                        "conv_id": conv_id,
                        "intent": event.get("intent", "other"),
                        "agent_type": event.get("agent_type", "explain"),
                    })
                    continue

                if event_type == "delta":
                    text = str(event.get("text", ""))
                    if text:
                        if not first_delta_sent:
                            first_delta_sent = True
                            logger.info(
                                "流式首段已发送: conv_id=%s elapsed_ms=%.1f",
                                conv_id,
                                (time.monotonic() - started_at) * 1000,
                            )
                        yield _sse_event("delta", {"text": text})
                    continue

                if event_type != "done":
                    continue

                result = event["result"]
                try:
                    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await _memory.add_message(
                        req.user_id, conv_id, MsgRole.ASSISTANT, result.response
                    )
                    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))
                except Exception:
                    # 回答已经输出，记忆写入失败不应让客户端把本轮回答判定为失败。
                    logger.exception("流式回答写入记忆失败: conv_id=%s", conv_id)

                yield _sse_event("done", {
                    "conv_id": conv_id,
                    "response": result.response,
                    "intent": result.intent.value if result.intent else "other",
                    "agent_type": result.agent_type.value,
                    "escalated": result.escalated,
                    "latency_ms": round(result.latency_ms, 1),
                    "knowledge_used": knowledge_used,
                })
                logger.info(
                    "流式回答完成: conv_id=%s elapsed_ms=%.1f",
                    conv_id,
                    (time.monotonic() - started_at) * 1000,
                )
                return
        except asyncio.CancelledError:
            logger.info("客户端取消流式请求: conv_id=%s", conv_id)
            raise
        except Exception as ex:
            logger.exception("流式对话失败: conv_id=%s", conv_id)
            yield _sse_event("error", {"message": str(ex) or "流式生成失败"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _prepare_chat_request(req: ChatRequest, conv_id: str):
    """并行读取 Memory/RAG，并构建普通和流式接口共用的编排请求。"""
    from agents.agent_orchestrator import Request as OrcReq

    memory_task = asyncio.create_task(
        _memory.get_context(req.user_id, conv_id, query=req.message)
    )
    knowledge_task = asyncio.create_task(_build_knowledge_context(req.message))
    mem_ctx, knowledge_result = await asyncio.gather(memory_task, knowledge_task)
    knowledge_text, knowledge_used = knowledge_result

    history = [
        {"role": message.role.value, "content": message.content}
        for message in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    context_parts = [mem_ctx.to_prompt_text()]
    if knowledge_text:
        context_parts.append(knowledge_text)

    return OrcReq(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context="\n\n".join(part for part in context_parts if part),
        history=history,
    ), knowledge_used


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """编码单个 SSE 事件，关闭 ASCII 转义以便直接传输中文。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def _search_course_knowledge(
    query: str,
    top_k: int,
    *,
    enhanced: bool = False,
) -> List[Dict[str, Any]]:
    """Normal chat RAG service; it is deliberately not exposed as an Agent tool."""
    if _knowledge_base is None:
        return []

    queries = [query]
    if enhanced and _tool_manager is not None:
        rewritten = await _tool_manager.rewrite_query(query, n=3)
        queries = list(dict.fromkeys([query, *rewritten]))

    recall_k = max(top_k, 5)
    tasks = [
        _knowledge_base.search_handler(
            {
                "query": current_query,
                "top_k": recall_k,
                "retrieval_mode": "hybrid" if index == 0 else "dense",
            },
            None,
        )
        for index, current_query in enumerate(queries)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: Dict[tuple, Dict[str, Any]] = {}
    for result in results:
        if isinstance(result, Exception) or not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict) or not str(item.get("content", "")).strip():
                continue
            key = (
                str(item.get("title", "")),
                str(item.get("content", "")),
                item.get("chunk", 0),
            )
            previous = merged.get(key)
            if previous is None or float(item.get("score", 0.0)) > float(previous.get("score", 0.0)):
                merged[key] = item

    return sorted(
        merged.values(),
        key=lambda item: float(item.get("fusion_score", item.get("score", 0.0))),
        reverse=True,
    )[:top_k]


async def _build_knowledge_context(message: str, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    知识库属于固定预处理流程，不作为模型可自主调用的 Agent 工具。
    """
    if _knowledge_base is None:
        return "", False
    if not _should_use_knowledge(message):
        return "", False
    try:
        # 对话链路默认使用单次向量检索，避免查询改写和重排额外产生两次 LLM
        # 请求。需要更高召回率时可设置 CHAT_RAG_MODE=enhanced 恢复完整链路。
        if os.getenv("CHAT_RAG_MODE", "fast").strip().lower() == "enhanced":
            items = await _search_course_knowledge(message, top_k, enhanced=True)
        else:
            items = await _search_course_knowledge(message, top_k)

        # 快速检索没有命中时，才执行查询改写，提高召回率。
        fallback_enabled = (os.getenv("CHAT_RAG_FALLBACK_ENHANCED","true",)
                            .strip().lower() in {"1", "true", "yes", "on"})

        fast_result_empty = (
                not items
        )

        if fallback_enabled and fast_result_empty:
            logger.info(
                "RAG 快速检索无结果，启用增强检索: query=%r",
                message,
            )
            items = await _search_course_knowledge(message, top_k, enhanced=True)

        parts = ["[知识库检索结果]"]
        seen = set()
        used = False
        for item in items:
            if not isinstance(item, dict):
                continue
            # 降级结果不能算成功使用知识库
            if item.get("fallback"):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")

            dedupe_key = (title, content)
            if not content or dedupe_key in seen:
                continue

            seen.add(dedupe_key)

            if not content:
                continue
            used = True
            parts.append(f"{len(seen)}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

            if len(seen) >= top_k:
                break

        if not used:
            return "", False
        parts.append("请优先依据以上课程资料回答。"
                    "如果资料不足，请明确说明哪些内容来自通用知识，"
                    "不要编造课程中的定义、结论或示例。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str) -> bool:
    """纯寒暄不检索，其他学习问题检索课程资料。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    return True

@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    top_k = max(1, min(top_k, 8))
    results = await _search_course_knowledge(query, top_k, enhanced=True)
    return {"query": query, "results": results, "reranked": True}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None

class EvalTurnInput(BaseModel):
    message: str = Field(min_length=1)

    # 确定性断言
    expected_intent: Optional[str] = None
    expected_agent_type: Optional[str] = None
    expect_knowledge: Optional[bool] = None

    # 回答中必须出现的词
    required_response_terms: List[str] = Field(default_factory=list)

    # Memory/RAG 准备出来的上下文中必须出现的词
    expected_context_terms: List[str] = Field(default_factory=list)

    # 给 Judge 提供参考答案
    reference_answer: Optional[str] = None

    # 本轮结束后是否同步等待画像更新
    update_profile_after: bool = False

    expected_profile_terms: List[str] = Field(
        default_factory=list
    )

class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    name: Optional[str] = None
    question: Optional[str] = None

    # 保留字符串格式兼容性，同时支持完整结构
    turns: Optional[List[Union[str, EvalTurnInput]]] = None

    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    # 普通评测默认不能覆盖基线
    save_as_baseline: bool = False


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    if _knowledge_base is None or _tool_manager is None:
        raise HTTPException(503, "知识库未初始化")
    count = _knowledge_base.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    _tool_manager.clear_cache("search_question_bank")
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": _knowledge_base.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    if _knowledge_base is None or _tool_manager is None:
        raise HTTPException(503, "知识库未初始化")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = _knowledge_base.add_documents(docs)
    _tool_manager.clear_cache("search_question_bank")
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": _knowledge_base.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if _knowledge_base is None:
        raise HTTPException(503, "知识库未初始化")
    return {"total_chunks": _knowledge_base.doc_count}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    save_as_baseline = (
        body.save_as_baseline
        if body is not None
        else False
    )

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        save_as_baseline = save_as_baseline,
    )
    return {
        "timestamp": report.timestamp,
        "baseline_saved": save_as_baseline,
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
