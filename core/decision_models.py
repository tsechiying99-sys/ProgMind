"""模型无关的结构化决策结果。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from core.intent_recognizer import IntentCategory, UrgencyLevel


class CollaborationMode(str, Enum):
    """多 Agent 的执行方式。"""

    SINGLE = "single"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class AgentTask:
    """语义规划器拆分出的单个 Agent 子任务。"""

    agent_type: str
    instruction: str
    depends_on: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CollaborationPlan:
    """结构化多 Agent 协作计划。"""

    mode: CollaborationMode = CollaborationMode.SINGLE
    tasks: Tuple[AgentTask, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    reason: str = ""


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
    collaboration: CollaborationPlan = field(default_factory=CollaborationPlan)

    def should_use_rag(self, threshold: float) -> bool:
        """低置信度澄清请求不提前加载课程资料。"""
        return (
            not self.needs_clarification
            and self.needs_rag_probability >= threshold
        )
