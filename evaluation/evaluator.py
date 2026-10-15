"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""

import json
import logging
import os
import pathlib
import statistics
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional,Awaitable,Callable
from anthropic import AsyncAnthropic
from core.intent_recognizer import IntentRecognizer

from core.llm_utils import extract_json_value, extract_text_content

logger = logging.getLogger(__name__)

ChatRunner = Callable[..., Awaitable[Dict[str, Any]]]


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        """返回一个平均值"""
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    """评测结果"""
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是教学质量评估专家。

    学生问题:
    {question}

    教学回答:
    {response}

    {context_section}

    {reference_section}

    评分原则:
    - relevance: 回答是否直接解决学生当前问题
    - accuracy: 回答是否符合参考答案和课程资料
    - completeness: 是否覆盖当前阶段所需的关键信息
    - helpfulness: 是否适合学生当前学习阶段并能促进理解

    如果背景信息包含课程资料，accuracy 必须依据课程资料评分。
    如果提供参考答案，优先依据参考答案判断准确性。
    不要因为回答篇幅长而自动给高分。

    只返回 JSON:
    {{
      "relevance": 0.0,
      "accuracy": 0.0,
      "completeness": 0.0,
      "helpfulness": 0.0
    }}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
        reference_answer: Optional[str] = None,
    ) -> QualityScores:
        #对于空回复的处理
        if not response or not response.strip():
            return QualityScores(
                0.0,
                0.0,
                0.0,
                0.0,
                judge_failed=True,
                error="被评测回答为空",
            )

        #限制提示词的长度
        safe_question = self._clean_text(question)[:2000]
        safe_response = self._clean_text(response)[:6000]
        safe_context = self._clean_text(context or "")[-6000:]
        safe_reference = self._clean_text(reference_answer or "")[:2000]

        ctx_section = f"背景信息: {safe_context}" if safe_context else ""
        reference_section = (
            f"参考答案:\n{safe_reference}"
            if safe_reference
            else ""
        )

        prompt = self.JUDGE_PROMPT.format(
            question=safe_question,
            response=safe_response,
            context_section=ctx_section,
            reference_section=reference_section,
        )
        prompt = self._clean_text(prompt)
        last_error: Optional[Exception] = None

        for attempt in range(2):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "4096")),
                    temperature=0.0,
                    messages=[{"role": "user","content": prompt,}],
                )

                raw = extract_text_content(resp.content).strip()

                if not raw:
                    block_types = [
                        getattr(block, "type", type(block).__name__)
                        for block in (resp.content or [])
                    ]

                    raise ValueError(
                        "Judge 返回空内容: "
                        f"stop_reason={getattr(resp, 'stop_reason', None)}, "
                        f"blocks={block_types}"
                    )

                data = extract_json_value(
                    raw,
                    expected_type=dict,
                )

                values: Dict[str, float] = {}

                for name in (
                        "relevance",
                        "accuracy",
                        "completeness",
                        "helpfulness",
                ):
                    if name not in data:
                        raise ValueError(
                            f"Judge 缺少字段: {name}"
                        )

                    value = float(data[name])

                    if not math.isfinite(value):
                        raise ValueError(
                            f"Judge 字段不是有限数值: {name}"
                        )

                    if not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"Judge 字段超出范围: "
                            f"{name}={value}"
                        )

                    values[name] = value

                return QualityScores(**values)

            except Exception as ex:
                last_error = ex

                logger.warning(
                    "LLM Judge 第 %d 次失败: %s",
                    attempt + 1,
                    ex,
                )

        return QualityScores(
            relevance=0.0,
            accuracy=0.0,
            completeness=0.0,
            helpfulness=0.0,
            judge_failed=True,
            error=str(last_error or "Judge 未返回有效结果"),
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], [] #predictions保存模型预测的结果；ground_truth保存人工标准答案
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            history: Optional[List[Dict[str, str]]] = None
            if isinstance(case.context, dict):
                raw_history = case.context.get("history")
                if isinstance(raw_history, list):
                    valid_history: List[Dict[str, str]] = []
                    for message in raw_history:
                        if not isinstance(message, dict):
                            continue
                        role = str(message.get("role", "")).strip()
                        content = str(message.get("content", "")).strip()
                        if role not in {"user", "assistant"}:
                            continue
                        if not content:
                            continue
                        valid_history.append({
                            "role": role,
                            "content": content,
                        })
                    if valid_history:
                        history = valid_history

            result = await self._recognizer.recognize(
                case.message,
                history=history,
            )
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "history_used": bool(history),
                "history_count": len(history or []),
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth)) #预测正确累计加一；错误加0
        accuracy = correct / len(predictions) if predictions else 0.0 #准确率=正确个数/总个数

        # 每类 F1
        labels = sorted(set(ground_truth + predictions)) #收集真实答案和预测结果中出现过的所有意图识别类别。
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            #真实是该类别，预测也是该类别。
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            #预测成该类别，但真实不是。
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            #真实是该类别，但没有被预测出来。
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            #系统所有预测为 label 的消息中，有多少真的是 label？
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            #所有真实的 label 消息中，系统成功识别出了多少？
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}
        #使用调和平均:因为调和平均会更重地惩罚较低的一项。
        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
        chat_runner: Optional[ChatRunner] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._orchestrator     = orchestrator
        self._chat_runner      = chat_runner
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = [] #保存当前进程生命周期内的所有评估报告。
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None #持久化基线文件路径。
        self._baseline: Optional[EvalReport] = self._load_baseline() #启动时尝试从磁盘读取上一次保存的报告。

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
        save_as_baseline: bool = False,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        run_id = uuid.uuid4().hex[:8]
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {} #用于保存 IntentEvaluator.evaluate() 的返回结果。
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i, run_id)
                results.extend(case_results)
                for result in case_results:
                    judge_failed = bool(result.metadata.get("judge_failed",False))
                    if judge_failed:
                        continue
                    for metric in all_scores:
                        if metric in result.scores:
                            all_scores[metric].append(result.scores[metric])


        # 3. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics, results)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
        )
        self._history.append(report)
        if save_as_baseline:
            self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int,
                                    run_id:str) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""

        turns = self._normalize_turns(case)
        if not turns:
            return []

        if self._chat_runner is None:
            raise RuntimeError("ChatRunner 未初始化")

        conv_id = str(case.get("conv_id") or f"eval-conv-{run_id}-{case_idx}")
        user_id = str(case.get("user_id") or f"eval-user-{run_id}-{case_idx}")

        results: List[EvalResult] = []

        for turn_idx, turn in enumerate(turns):
            message = turn["message"]
            update_profile_after = bool(turn.get("update_profile_after", False))
            outcome = await self._chat_runner(
                message = message,
                user_id = user_id,
                conv_id = conv_id,
                wait_for_profile = update_profile_after,
            )
            response = str(outcome.get("response", "")).strip()
            context = str(outcome.get("context", ""))
            actual_intent = str(outcome.get("intent", "other"))
            actual_agent = str(outcome.get("agent_type", ""))
            knowledge_used = bool(
                outcome.get("knowledge_used", False)
            )

            checks: Dict[str, bool] = {
                "execution_success": bool(outcome.get("success")),
                "response_nonempty": bool(response),
            }

            expected_intent = turn.get("expected_intent")
            if expected_intent is not None:
                checks["intent_match"] = (
                        actual_intent == expected_intent
                )

            expected_agent = turn.get("expected_agent_type")
            if expected_agent is not None:
                checks["agent_match"] = (
                        actual_agent == expected_agent
                )

            expect_knowledge = turn.get("expect_knowledge")
            if expect_knowledge is not None:
                checks["knowledge_match"] = (
                        knowledge_used is bool(expect_knowledge)
                )

            required_terms = turn.get(
                "required_response_terms",
                [],
            )
            if required_terms:
                checks["response_terms"] = all(
                    str(term).lower() in response.lower()
                    for term in required_terms
                )

            expected_context_terms = turn.get(
                "expected_context_terms",
                [],
            )
            if expected_context_terms:
                checks["context_terms"] = all(
                    str(term).lower() in context.lower()
                    for term in expected_context_terms
                )

            if update_profile_after:
                checks["profile_updated"] = (
                        outcome.get("profile_updated") is True
                )

            expected_profile_terms = turn.get(
                "expected_profile_terms",
                [],
            )

            if expected_profile_terms:
                profile_text = json.dumps(
                    outcome.get("profile", {}),
                    ensure_ascii=False,
                ).lower()

                checks["profile_terms"] = all(
                    str(term).lower() in profile_text
                    for term in expected_profile_terms
                )

            scores = await self._judge.judge(
                question=message,
                response=response,
                context=context or None,
                reference_answer=turn.get("reference_answer"),
            )

            deterministic_passed = all(checks.values())

            passed = (
                    deterministic_passed
                    and not scores.judge_failed
                    and scores.overall >= self.PASS_THRESHOLD
            )

            failed_checks = [
                name
                for name, succeeded in checks.items()
                if not succeeded
            ]

            test_id = (
                f"dialog_{case_idx}_turn_{turn_idx}"
            )

            results.append(
                EvalResult(
                    test_id=test_id,
                    passed=passed,
                    scores={
                        "relevance": scores.relevance,
                        "accuracy": scores.accuracy,
                        "completeness": scores.completeness,
                        "helpfulness": scores.helpfulness,
                        "overall": scores.overall,
                    },
                    detail=(
                        f"intent={actual_intent}, "
                        f"agent={actual_agent}, "
                        f"knowledge={knowledge_used}, "
                        f"quality={scores.overall:.3f}, "
                        f"failed_checks={failed_checks}"
                    ),
                    metadata={
                        "case_name": case.get(
                            "name",
                            f"dialog_{case_idx}",
                        ),
                        "question": message,
                        "response": response,
                        "expected_intent": expected_intent,
                        "actual_intent": actual_intent,
                        "expected_agent_type": expected_agent,
                        "actual_agent_type": actual_agent,
                        "knowledge_used": knowledge_used,
                        "checks": checks,
                        "profile_updated": outcome.get(
                            "profile_updated"
                        ),
                        "profile": outcome.get("profile", {}),
                        "judge_failed": scores.judge_failed,
                        "judge_error": scores.error,
                        "turn": turn_idx,
                        "conv_id": conv_id,
                    },
                )
            )

        return results

    @staticmethod
    def _normalize_turns(case: Dict[str, Any],) -> List[Dict[str, Any]]:
        """提取测试事例case中的信息"""
        turns = case.get("turns")

        if isinstance(turns, list):
            normalized = []

            for turn in turns:
                if isinstance(turn, str):
                    message = turn.strip()
                    if message:
                        normalized.append({"message": message})

                elif isinstance(turn, dict):
                    message = str(
                        turn.get("message", "")
                    ).strip()

                    if message:
                        normalized.append({
                            **turn,
                            "message": message,
                        })

            return normalized

        question = str(case.get("question", "")).strip()

        if not question:
            return []

        return [{
            "message": question,
            "expected_intent": case.get("expected_intent"),
            "expected_agent_type": case.get(
                "expected_agent_type"
            ),
            "expect_knowledge": case.get(
                "expect_knowledge"
            ),
            "reference_answer": case.get(
                "reference_answer"
            ),
        }]

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        """统一把单轮和多轮格式转换成List集合"""
        turns = case.get("turns")
        if isinstance(turns, list):
            messages = []
            for turn in turns:
                if isinstance(turn, dict):
                    message = str(turn.get("message", "")).strip()
                else:
                    message = str(turn).strip()
                if message:
                    messages.append(message)
            return messages

        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        """返回最新的8条历史消息"""
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """回归检测，与固定基线进行比较"""
        if self._baseline is None:
            return []
        previous = self._baseline.avg_scores
        regressions = []

        for metric, value in current.items():
            baseline_value = previous.get(metric)
            if baseline_value is None:
                continue

            # 使用绝对百分点差，含义比相对百分比更直观
            delta = value - baseline_value
            if delta < -0.05:
                regressions.append(
                    f"{metric}: "
                    f"{baseline_value:.3f} → {value:.3f} "
                    f"(下降 {abs(delta):.3f})"
                )

        return regressions

    def _recommendations(
            self,
            scores: Dict[str, float],
            intent_metrics: Dict[str, Any],
            results: List[EvalResult],
    ) -> List[str]:
        recommendations: List[str] = []

        failed_results = [
            result for result in results
            if not result.passed
        ]

        failed_checks: Dict[str, int] = {}

        for result in failed_results:
            checks = result.metadata.get("checks", {})

            if not isinstance(checks, dict):
                continue

            for check_name, succeeded in checks.items():
                if succeeded is False:
                    failed_checks[check_name] = (
                            failed_checks.get(check_name, 0) + 1
                    )

        if failed_checks.get("execution_success"):
            recommendations.append(
                "Agent 执行失败：检查模型请求、超时和降级日志"
            )

        if failed_checks.get("response_nonempty"):
            recommendations.append(
                "存在空回答：检查模型内容提取和流式结束处理"
            )

        if failed_checks.get("intent_match"):
            recommendations.append(
                "意图识别与标注不一致：先确认测试标注是否符合问题语义，"
                "再调整意图规则或 Few-shot"
            )

        if failed_checks.get("agent_match"):
            recommendations.append(
                "Agent 路由与预期不一致：检查意图到 Agent 的路由映射"
            )

        if failed_checks.get("knowledge_match"):
            recommendations.append(
                "RAG 使用状态与预期不一致：检查检索阈值、知识库内容和检索开关"
            )

        if failed_checks.get("context_terms"):
            recommendations.append(
                "RAG/Memory 上下文缺少预期内容：检查召回结果是否真正进入 Agent 上下文"
            )

        if failed_checks.get("response_terms"):
            recommendations.append(
                "回答缺少必要内容：检查检索结果是否被 Agent 正确使用"
            )

        if failed_checks.get("profile_updated"):
            recommendations.append(
                "用户画像更新失败：检查画像 LLM 输出和 ChromaDB 写入"
            )

        if failed_checks.get("profile_terms"):
            recommendations.append(
                "画像已写入但缺少预期薄弱点：检查画像提取提示词和合并逻辑"
            )

        judge_failures = sum(
            bool(result.metadata.get("judge_failed"))
            for result in results
        )

        if judge_failures:
            recommendations.append(
                f"LLM Judge 失败 {judge_failures} 次："
                "本次质量平均分不应视为完整有效结果"
            )

        if scores.get("intent_accuracy", 1.0) < 0.90:
            recommendations.append(
                "独立意图识别准确率低于 90%：补充易混淆类别测试数据"
            )

        if scores.get("relevance", 1.0) < 0.75:
            recommendations.append(
                "回答相关性偏低：检查 Agent 是否聚焦用户问题"
            )

        if scores.get("accuracy", 1.0) < 0.75:
            recommendations.append(
                "回答准确性偏低：检查课程资料、参考答案和事实依据"
            )

        if scores.get("completeness", 1.0) < 0.75:
            recommendations.append(
                "回答完整性偏低：补充当前学习阶段所需的关键说明"
            )

        if scores.get("helpfulness", 1.0) < 0.75:
            recommendations.append(
                "回答有用性偏低：调整讲解难度并增加示例或引导"
            )

        if not recommendations:
            recommendations.append(
                "所有确定性检查和质量指标均达标，继续保持"
            )

        return recommendations

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        """读取评测基线"""
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        """保存评测基线"""
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        """将data转换为EvalReport格式"""
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

from evaluation.default_cases import JAVA_20_TURN_DIALOG, JAVA_INTENT_CASE_DATA

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("请给我讲解一下 Python 装饰器","explain"),
    IntentTestCase("为什么列表推导式通常比普通循环更简洁？","qa"),
    IntentTestCase("给我出三道关于 Python 函数的选择题","quiz"),
    IntentTestCase("帮我复习一下今天学过的递归知识","review"),
    IntentTestCase("你好","greeting"),
    *[
        IntentTestCase(message, expected_intent)
        for message, expected_intent in JAVA_INTENT_CASE_DATA
    ],
]

DEFAULT_DIALOG_CASES = [
    {
        "name": "讲解后出题",
        "turns": [
            {
                "message": "给我讲一下 Python 函数参数",
                "expected_intent": "explain",
                "expected_agent_type": "explain",
                "required_response_terms": ["参数"],
            },
            {
                "message": "根据刚才的内容给我出一道题",
                "expected_intent": "quiz",
                "expected_agent_type": "quiz",
                "expected_context_terms": ["函数参数"],
            },
        ],
    },
    {
        "name": "薄弱点画像与后续使用",
        "turns": [
            {
                "message": "请解释 Python 闭包",
                "expected_intent": "explain",
                "expected_agent_type": "explain",
                "expect_knowledge": True,
                "required_response_terms": ["闭包"],
                "reference_answer": (
                    "闭包是函数以及它定义时引用的外部作用域变量的组合。"
                ),
            },
            {
                "message": (
                    "我还是不理解 Python 闭包，"
                    "尤其不明白外层函数结束后变量为什么不会消失，"
                    "这是我的薄弱点。"
                ),
                "expected_intent": "qa",
                "expected_agent_type": "qa",
                "required_response_terms": ["闭包"],
                "expect_knowledge": True,
                "update_profile_after": True,
                "expected_profile_terms": ["外层函数","变量","不会消失",],
                "reference_answer": (
                    "内部函数仍然持有对外层作用域变量的引用，"
                    "因此外层函数结束后变量不会立即消失。"
                ),
            },
            {
                "message": "请根据我的薄弱点，换一种更简单的方式讲解",
                "expected_intent": "review",
                "expected_agent_type": "general",
                "expect_knowledge": True,
                "expected_context_terms": [
                    "[用户画像]",
                    "weak_points",
                    "闭包",
                ],
            },
        ],
    },
    {
        "name": "RAG 默认参数检索",
        "turns": [
            {
                "message": (
                    "为什么 Python 默认参数是在函数定义时求值，"
                    "而不是每次调用时求值？"
                ),
                "expected_intent": "qa",
                "expected_agent_type": "qa",
                "expect_knowledge": True,
                "expected_context_terms": [
                    "PROGMIND-COURSE-FUNC-001",
                    "定义时求值",
                ],
                "required_response_terms": [
                    "定义时",
                ],
                "reference_answer": (
                    "Python 默认参数表达式在函数定义时求值。"
                ),
            },
        ],
    },
    JAVA_20_TURN_DIALOG,
]
