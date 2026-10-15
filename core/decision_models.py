"""模型无关的结构化决策结果。"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from core.intent_recognizer import IntentCategory, UrgencyLevel


@dataclass(frozen=True)
class JevShadowDecision:
    """Jev 影子决策，只用于评测，不参与正式业务分支。"""

    available: bool
    intent: Optional[IntentCategory] = None
    intent_confidence: float = 0.0
    urgency_score: float = 0.0
    urgency_confidence: float = 0.0
    needs_rag_probability: float = 0.0
    needs_clarification_probability: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class RoutingDecision:
    """正式路由使用的统一决策结构。"""

    intent: IntentCategory
    intent_confidence: float
    urgency: UrgencyLevel
    urgency_confidence: float
    needs_rag_probability: float
    needs_clarification: bool
    source: str
    latency_ms: float
    jev_shadow: Optional[JevShadowDecision] = None

    def should_use_rag(self, threshold: float) -> bool:
        """低置信度澄清请求不提前加载课程资料。"""
        return (
            not self.needs_clarification
            and self.needs_rag_probability >= threshold
        )


@dataclass(frozen=True)
class EvidenceShadowDecision:
    """单个 RAG 片段的 Jev 影子准入结果。"""

    index: int
    relevance: float
    answer_evidence: float
    contradiction: float
    prompt_injection: float
    route: str
    latency_ms: float
    error: Optional[str] = None
