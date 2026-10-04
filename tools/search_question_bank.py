"""
从课程知识库中筛选明确属于题目或练习的内容。
"""

from typing import Any, Dict

from mcp.tool_manager import Tool
from tools.contracts import object_schema


def create_tool(knowledge_base: Any) -> Tool:
    async def handle(params: Dict[str, Any], context: Any):
        candidates = await knowledge_base.search_handler({
            "query": params["query"],
            "top_k": min(params.get("top_k", 5) * 3, 24),
            "retrieval_mode": "hybrid",
        }, context)
        markers = ("练习", "题目", "习题", "quiz", "question")
        questions = [
            item for item in candidates
            if any(marker in f"{item.get('title', '')} {item.get('content', '')}".lower() for marker in markers)
        ][:params.get("top_k", 5)]
        return {
            "configured": bool(questions),
            "questions": questions,
            "note": "仅返回课程知识库中明确标记为题目或练习的内容。",
        }

    return Tool(
        name="search_question_bank",
        description="在课程知识库中查找明确标记为题目或练习的内容；没有独立题库时如实返回空结果。",
        handler=handle,
        schema=object_schema({
            "query": {"type": "string", "minLength": 1, "maxLength": 300},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
        }, ("query",)),
        cache_ttl=120.0,
    )
