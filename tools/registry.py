"""Explicit application-level registration for all learning tools."""

from mcp.tool_manager import MCPToolManager
from tools import build_hint_plan, check_prerequisites
from tools import grade_answer, record_learning_result, search_question_bank
from tools.contracts import ToolDependencies


def register_learning_tools(manager: MCPToolManager, deps: ToolDependencies) -> None:
    """Register once at startup; modules have no import-time side effects."""
    factories = (
        # 教学与提示规则工具
        check_prerequisites.create_tool(),
        build_hint_plan.create_tool(),
        # 题目检索工具
        search_question_bank.create_tool(deps.knowledge_base),
        # 评分与学习结果写入工具
        grade_answer.create_tool(),
        record_learning_result.create_tool(deps.memory_manager),
    )
    for tool in factories:
        manager.register(tool)
