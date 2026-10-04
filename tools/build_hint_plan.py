"""
生成由弱到强的提示层级，避免 QA Agent 直接泄露答案。
工具不生成具体提示内容，而是生成“提示策略”。
这可以理解为给 LLM 加了一层结构化教学策略。
"""
from typing import Any, Dict

from mcp.tool_manager import Tool
from tools.contracts import ToolExecutionContext, object_schema, require_context


def _handle(params: Dict[str, Any], context: ToolExecutionContext) -> Dict[str, Any]:
    context = require_context(context)
    return {
        "question": params.get("question") or context.message,
        "attempted_step": params.get("attempted_step", ""),
        "hint_levels": [
            "先指出需要检查的概念或条件",
            "再提示下一步推理方向",
            "仍无法继续时给出局部示例，但不直接交付完整答案",
        ],
        "reveal_full_answer": False,
    }


def create_tool() -> Tool:
    return Tool(
        name="build_hint_plan",
        description="生成逐级提示方案，默认不泄露练习的完整答案。",
        handler=_handle,
        schema=object_schema({
            "question": {"type": "string", "minLength": 1, "maxLength": 500},
            "attempted_step": {"type": "string", "maxLength": 300},
        }, ("question",)),
    )
