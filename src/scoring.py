"""成员信号分数计算。

定义（README §5）：

    s_i = 1          RAG 回答正确
    s_i = -lambda    RAG 回答 Unknown
    s_i = 0          RAG 给出错误的 Yes/No

一篇文档的成员得分为其所有问题的信号分数均值。
"""

from __future__ import annotations


def signal_score(
    predicted_answer: str,
    ground_truth: str,
    unknown_penalty: float,
) -> float:
    """计算单个问题的信号分数。"""
    if predicted_answer == "unknown":
        return -unknown_penalty
    if predicted_answer == ground_truth:
        return 1.0
    return 0.0


def aggregate_score(scores: list[float]) -> float:
    """对一篇文档的问题信号分数取均值。"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def normalize_label(answer: str) -> str:
    """把模型输出归一化到 yes/no/unknown 三者之一。"""
    label = answer.strip().lower()
    if label in {"yes", "no", "unknown"}:
        return label
    return "unknown"
