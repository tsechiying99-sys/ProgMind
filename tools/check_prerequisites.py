"""
为知识讲解生成前置知识检查框架。
"""
from typing import Any, Dict

from mcp.tool_manager import Tool
from tools.contracts import object_schema


def _handle(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    return {
        "topic": params["topic"].strip(),
        "known_topics": params.get("known_topics", []),
        "checks": [
            "确认学生理解题目中的核心术语",
            "确认学生具备示例所需的基础知识",
            "从最少的新概念开始讲解",
        ],
        "verified": False,
        "note": "该工具只生成检查清单，不会断言学生已掌握前置知识。",
    }


def create_tool() -> Tool:
    return Tool(
        name="check_prerequisites",
        description="为知识讲解生成保守的前置知识检查清单。",
        handler=_handle,
        schema=object_schema({
            "topic": {"type": "string", "minLength": 1, "maxLength": 120},
            "known_topics": {
                "type": "array",
                "items": {"type": "string", "maxLength": 80},
            },
        }, ("topic",)),
    )
