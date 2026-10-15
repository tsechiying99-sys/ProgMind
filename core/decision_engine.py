"""正式意图识别与 Jev 影子决策的统一入口。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from core.decision_models import JevShadowDecision, RoutingDecision
from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.jev_client import JevClient

logger = logging.getLogger(__name__)


_INTENT_CRITERIA = {
    "explain": "希望讲解概念、原理或知识点",
    "qa": "针对具体疑问、错误、代码或不理解之处寻求帮助",
    "quiz": "希望生成练习题、接受测验或提交练习答案",
    "review": "希望复习、总结、回顾薄弱点或错题",
    "greeting": "普通问候、感谢或开始学习",
    "other": "不属于以上意图，或现有信息不足以判断",
}


class DecisionEngine:
    """现有识别器负责生产路由，Jev 只并行产生影子结果。"""

    def __init__(
        self,
        primary: IntentRecognizer,
        jev_client: Optional[JevClient] = None,
        *,
        shadow_enabled: bool = False,
        confidence_threshold: float = 0.55,
    ):
        self._primary = primary
        self._jev_client = jev_client
        self._shadow_enabled = shadow_enabled
        self._confidence_threshold = confidence_threshold

    async def decide(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        request_id: str = "",
    ) -> RoutingDecision:
        started = time.monotonic()
        primary_task = asyncio.create_task(
            self._primary.recognize(message, history=history)
        )
        shadow_task = None
        if (
            self._shadow_enabled
            and self._jev_client is not None
            and self._jev_client.enabled
        ):
            shadow_task = asyncio.create_task(
                self._run_jev_shadow(message, history)
            )

        primary = await primary_task
        shadow = await shadow_task if shadow_task is not None else None

        needs_clarification = (
            primary.intent == IntentCategory.OTHER
            or primary.confidence < self._confidence_threshold
        )
        needs_rag_probability = self._rag_probability(primary.intent)

        if shadow is not None:
            logger.info(
                "Jev 路由影子结果: request_id=%s primary=%s jev=%s confidence=%.3f",
                request_id,
                primary.intent.value,
                shadow.intent.value if shadow.intent else "unavailable",
                shadow.intent_confidence,
            )

        return RoutingDecision(
            intent=primary.intent,
            intent_confidence=max(0.0, min(float(primary.confidence), 1.0)),
            urgency=primary.urgency,
            urgency_confidence=0.0,
            needs_rag_probability=needs_rag_probability,
            needs_clarification=needs_clarification,
            source="existing_intent_recognizer",
            latency_ms=(time.monotonic() - started) * 1000,
            jev_shadow=shadow,
        )

    @staticmethod
    def _rag_probability(intent: IntentCategory) -> float:
        if intent == IntentCategory.GREETING:
            return 0.0
        if intent == IntentCategory.OTHER:
            return 0.25
        return 1.0

    async def _run_jev_shadow(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> JevShadowDecision:
        assert self._jev_client is not None
        started = time.monotonic()
        try:
            payload = await self._jev_client.evaluate(
                state={
                    "message": message,
                    "recent_history": (history or [])[-3:],
                },
                questions={
                    "intent": {
                        "type": "choice",
                        "instructions": "学生当前最主要的学习意图是什么？",
                        "criteria": _INTENT_CRITERIA,
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "学生请求表现出的时间紧迫程度",
                        "criteria": [
                            "没有明确时间压力",
                            "希望近期处理，但不紧急",
                            "希望今天或尽快处理",
                            "明确要求立即处理或需要教师及时介入",
                        ],
                    },
                    "needs_rag": {
                        "type": "noul",
                        "instructions": "回答是否需要查询课程知识库中的事实、定义、示例或课程规范？",
                    },
                    "needs_clarification": {
                        "type": "noul",
                        "instructions": "当前信息是否不足以可靠判断学生的学习目标？",
                    },
                },
            )
            answers = payload["answers"]
            intent_answer = answers["intent"]
            intent = IntentCategory(intent_answer["choice"])
            return JevShadowDecision(
                available=True,
                intent=intent,
                intent_confidence=float(intent_answer.get("confidence", 0.0)),
                urgency_score=float(answers["urgency"].get("score", 0.0)),
                urgency_confidence=float(answers["urgency"].get("confidence", 0.0)),
                needs_rag_probability=float(answers["needs_rag"].get("noul", 0.0)),
                needs_clarification_probability=float(
                    answers["needs_clarification"].get("noul", 0.0)
                ),
                probabilities={
                    key: float(value)
                    for key, value in intent_answer.get("probabilities", {}).items()
                },
                model=str(payload.get("model", self._jev_client.model)),
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as ex:
            logger.warning("Jev 路由影子调用失败: %s", ex)
            return JevShadowDecision(
                available=False,
                model=self._jev_client.model,
                latency_ms=(time.monotonic() - started) * 1000,
                error=str(ex),
            )
