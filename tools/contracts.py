"""Shared contracts for tool execution and dependency injection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ToolExecutionContext:
    request_id: str
    user_id: str
    conv_id: str
    message: str
    agent_type: str
    intent: Optional[str] = None
    urgency: Optional[str] = None
    conversation_context: str = ""
    history: Tuple[Dict[str, str], ...] = ()

    @classmethod
    def from_request(cls, request: Any, agent_type: Any) -> "ToolExecutionContext":
        intent = getattr(request, "intent", None)
        urgency = getattr(request, "urgency", None)
        return cls(
            request_id=str(getattr(request, "request_id", "")),
            user_id=str(getattr(request, "user_id", "")),
            conv_id=str(getattr(request, "conv_id", "")),
            message=str(getattr(request, "message", "")),
            agent_type=str(getattr(agent_type, "value", agent_type)),
            intent=getattr(intent, "value", None),
            urgency=getattr(urgency, "name", None),
            conversation_context=str(getattr(request, "context", "")),
            history=tuple(getattr(request, "history", None) or ()),
        )


@dataclass(frozen=True)
class ToolDependencies:
    knowledge_base: Any
    memory_manager: Any


def object_schema(properties: Dict[str, Any], required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False, #可以阻止 LLM偷偷添加未声明字段。
    }


async def resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def require_context(context: Any) -> ToolExecutionContext:
    if not isinstance(context, ToolExecutionContext):
        raise ValueError("该工具只能在经过认证的 Agent 请求上下文中调用")
    return context
