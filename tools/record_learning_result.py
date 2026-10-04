"""
把已完成练习结果写入 Memory。
"""

from typing import Any, Dict

from mcp.tool_manager import Tool
from tools.contracts import object_schema, require_context, resolve


def create_tool(memory_manager: Any) -> Tool:
    async def handle(params: Dict[str, Any], context: Any):
        context = require_context(context)
        record = dict(params)
        record["source_agent"] = context.agent_type
        return await resolve(memory_manager.record_learning_result(
            context.user_id,
            context.conv_id,
            record,
        ))

    return Tool(
        name="record_learning_result",
        description="把已完成练习的评分结果写入当前学生的学习记录。",
        handler=handle,
        schema=object_schema({
            "topic": {"type": "string", "minLength": 1, "maxLength": 160},
            "question": {"type": "string", "minLength": 1, "maxLength": 1000},
            "student_answer": {"type": "string", "minLength": 1, "maxLength": 4000},
            "score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
            "correct": {"type": "boolean"},
            "feedback": {"type": "string", "maxLength": 1000},
        }, ("topic", "question", "student_answer", "score", "correct")),
        read_only=False,
    )
