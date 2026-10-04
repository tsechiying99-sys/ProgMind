"""
对学生答案进行确定性的初步评分。
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from mcp.tool_manager import Tool
from tools.contracts import object_schema


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def _handle(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    answer = _normalize(params["student_answer"])
    reference = _normalize(params["reference_answer"])
    key_points: List[str] = params.get("key_points", [])
    threshold = params.get("pass_threshold", 0.8)

    if key_points:
        matched = [point for point in key_points if _normalize(point) in answer]
        missing = [point for point in key_points if point not in matched]
        score = 100.0 * len(matched) / len(key_points)
        method = "key_point_coverage"
    else:
        matched, missing = [], []
        score = 100.0 * SequenceMatcher(None, answer, reference).ratio()
        method = "normalized_similarity"
    score = round(score, 1)
    return {
        "score": score,
        "correct": score >= threshold * 100,
        "method": method,
        "matched_key_points": matched,
        "missing_key_points": missing,
        "note": "这是确定性规则评分，不代替教师对开放题的人工评阅。",
    }


def create_tool() -> Tool:
    return Tool(
        name="grade_answer",
        description="用参考答案或关键点对学生答案进行确定性初步评分。",
        handler=_handle,
        schema=object_schema({
            "student_answer": {"type": "string", "minLength": 1, "maxLength": 4000},
            "reference_answer": {"type": "string", "minLength": 1, "maxLength": 4000},
            "key_points": {"type": "array", "items": {"type": "string", "maxLength": 300}},
            "pass_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        }, ("student_answer", "reference_answer")),
    )
