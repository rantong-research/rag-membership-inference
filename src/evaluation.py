"""离线评估：对已保存的成员/非成员语义测试结果做统计与分析。

只依赖标准库 + numpy/scipy/scikit-learn，不依赖 GPU 或任何模型，可直接运行：

    python evaluate.py
    python -m src.evaluation --member member_semantic_test.json \
        --nonmember nonmember_semantic_test.json

输出包括：文档级 ROC-AUC、不同阈值下的准确率/精确率/召回率/假阳性率、
成员与非成员得分均值与 Bootstrap 置信区间、Mann-Whitney U 检验、
Yes/No/Unknown 分布、目标文档 Recall@k 以及检索成功/失败时的回答正确率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# 结果读取与标准化
# ---------------------------------------------------------------------------

def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层结构应为数组")
    return data


def _doc_mia_score(doc: dict[str, Any]) -> float:
    if "mia_score" in doc:
        return float(doc["mia_score"])
    scores = [q.get("signal_score", 0.0) for q in doc.get("questions", [])]
    return sum(scores) / len(scores) if scores else 0.0


def _q_answer_matches(question: dict[str, Any]) -> bool | None:
    # 成员结果使用 answer_matches，非成员旧结果使用 correct，这里统一处理。
    if "answer_matches" in question:
        return bool(question["answer_matches"])
    if "correct" in question:
        return bool(question["correct"])
    return None


def build_tables(
    member_docs: list[dict[str, Any]],
    nonmember_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造文档级与问题级的扁平表格。"""
    doc_rows: list[dict[str, Any]] = []
    for docs, label in ((member_docs, 1), (nonmember_docs, 0)):
        for doc in docs:
            doc_rows.append({
                "membership": "member" if label else "nonmember",
                "label": label,
                "target_id": doc.get("target_id"),
                "target_source_line": doc.get("target_source_line"),
                "mia_score": _doc_mia_score(doc),
                "question_count": len(doc.get("questions", [])),
            })

    question_rows: list[dict[str, Any]] = []
    for docs, label in ((member_docs, 1), (nonmember_docs, 0)):
        for doc in docs:
            for question in doc.get("questions", []):
                question_rows.append({
                    "membership": "member" if label else "nonmember",
                    "label": label,
                    "target_id": doc.get("target_id"),
                    "ground_truth": str(question.get("ground_truth", "")).lower(),
                    "predicted_answer": str(question.get("predicted_answer", "")).lower(),
                    "answer_matches": _q_answer_matches(question),
                    "signal_score": float(question.get("signal_score", 0.0)),
                    "target_retrieved": bool(question.get("target_retrieved", False)),
                    "target_rank": question.get("target_rank"),
                    "target_distance": question.get("target_distance"),
                })
    return doc_rows, question_rows


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def roc_auc(labels: list[int], scores: list[float]) -> float:
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_arr, scores_arr))


def threshold_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
    greater: bool,
) -> dict[str, Any]:
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    pred = scores_arr > threshold if greater else scores_arr >= threshold

    tp = int(np.sum(pred & (labels_arr == 1)))
    fp = int(np.sum(pred & (labels_arr == 0)))
    fn = int(np.sum(~pred & (labels_arr == 1)))
    tn = int(np.sum(~pred & (labels_arr == 0)))

    total = len(labels_arr)
    return {
        "threshold": threshold,
        "rule": ">" if greater else ">=",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def best_threshold(
    labels: list[int],
    scores: list[float],
) -> dict[str, Any]:
    """在候选阈值上按准确率扫描，返回最优判定规则。"""
    best: dict[str, Any] | None = None
    for threshold in sorted(set(scores)):
        for greater in (True, False):
            metric = threshold_metrics(labels, scores, threshold, greater)
            if best is None or metric["accuracy"] > best["accuracy"]:
                best = metric
    return best if best is not None else {}


def bootstrap_ci(
    metric: Callable[..., float],
    samples: list[list[float]],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """对 metric(*samples) 做独立重采样 Bootstrap，返回 95% 置信区间。"""
    arrays = [np.asarray(s) for s in samples]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)

    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in arrays]
        try:
            estimates[i] = metric(*resampled)
        except (ValueError, ZeroDivisionError):
            estimates[i] = float("nan")

    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return {"low": float("nan"), "high": float("nan"),
                "n_boot": n_boot, "alpha": alpha, "valid_boots": 0}

    low = float(np.percentile(valid, 100 * alpha / 2))
    high = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return {
        "low": low, "high": high,
        "mean": float(np.mean(valid)),
        "n_boot": n_boot, "alpha": alpha,
        "valid_boots": int(len(valid)),
    }


def _dist(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    dist: dict[str, int] = {}
    for row in rows:
        value = row[key]
        dist[value] = dist.get(value, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def evaluate(
    member_path: Path,
    nonmember_path: Path,
    unknown_penalty: float = 0.5,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    member_docs = load_documents(member_path)
    nonmember_docs = load_documents(nonmember_path)
    doc_rows, question_rows = build_tables(member_docs, nonmember_docs)

    member_scores = [r["mia_score"] for r in doc_rows if r["label"] == 1]
    nonmember_scores = [r["mia_score"] for r in doc_rows if r["label"] == 0]
    labels = [r["label"] for r in doc_rows]
    scores = [r["mia_score"] for r in doc_rows]

    # 文档级 ROC-AUC
    auc = roc_auc(labels, scores)

    # Mann-Whitney U（双尾）
    mw = stats.mannwhitneyu(
        member_scores, nonmember_scores,
        alternative="two-sided",
        method="auto",
    )

    # 得分均值与 Bootstrap 置信区间
    mean_member = float(np.mean(member_scores)) if member_scores else float("nan")
    mean_nonmember = float(np.mean(nonmember_scores)) if nonmember_scores else float("nan")

    bootstrap = {
        "member_mean": bootstrap_ci(
            lambda s: float(np.mean(s)), [member_scores], seed=bootstrap_seed
        ),
        "nonmember_mean": bootstrap_ci(
            lambda s: float(np.mean(s)), [nonmember_scores], seed=bootstrap_seed
        ),
        "mean_difference": bootstrap_ci(
            lambda a, b: float(np.mean(a) - np.mean(b)),
            [member_scores, nonmember_scores],
            seed=bootstrap_seed,
        ),
    }

    # 阈值指标
    thresholds = [
        threshold_metrics(labels, scores, 0.0, greater=True),
        threshold_metrics(labels, scores, 0.0, greater=False),
        best_threshold(labels, scores),
    ]

    # 问题级统计
    member_q = [q for q in question_rows if q["label"] == 1]
    nonmember_q = [q for q in question_rows if q["label"] == 0]

    def q_acc(rows: list[dict[str, Any]]) -> float:
        matches = [q["answer_matches"] for q in rows if q["answer_matches"] is not None]
        return sum(matches) / len(matches) if matches else float("nan")

    # 检索统计（仅成员文档有目标文档）
    member_retrieved = [q for q in member_q if q["target_retrieved"]]
    member_not_retrieved = [q for q in member_q if not q["target_retrieved"]]
    retrieval_rate = (
        len(member_retrieved) / len(member_q) if member_q else float("nan")
    )

    # 目标文档 Rank 分布（命中时）
    ranks = [q["target_rank"] for q in member_retrieved
             if q["target_rank"] is not None]

    report: dict[str, Any] = {
        "unknown_penalty": unknown_penalty,
        "counts": {
            "member_documents": len(member_docs),
            "nonmember_documents": len(nonmember_docs),
            "member_questions": len(member_q),
            "nonmember_questions": len(nonmember_q),
        },
        "scores": {
            "member_mean": mean_member,
            "nonmember_mean": mean_nonmember,
            "mean_difference": mean_member - mean_nonmember,
            "member_median": float(np.median(member_scores)) if member_scores else float("nan"),
            "nonmember_median": float(np.median(nonmember_scores)) if nonmember_scores else float("nan"),
            "member_min": float(np.min(member_scores)) if member_scores else float("nan"),
            "member_max": float(np.max(member_scores)) if member_scores else float("nan"),
            "nonmember_min": float(np.min(nonmember_scores)) if nonmember_scores else float("nan"),
            "nonmember_max": float(np.max(nonmember_scores)) if nonmember_scores else float("nan"),
        },
        "roc_auc": auc,
        "mann_whitney": {
            "statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
        },
        "bootstrap": bootstrap,
        "thresholds": thresholds,
        "question_accuracy": {
            "member": q_acc(member_q),
            "nonmember": q_acc(nonmember_q),
        },
        "answer_distribution": {
            "member": _dist(member_q, "predicted_answer"),
            "nonmember": _dist(nonmember_q, "predicted_answer"),
        },
        "ground_truth_distribution": {
            "member": _dist(member_q, "ground_truth"),
            "nonmember": _dist(nonmember_q, "ground_truth"),
        },
        "retrieval": {
            "target_retrieval_rate": retrieval_rate,
            "retrieved_question_accuracy": q_acc(member_retrieved),
            "not_retrieved_question_accuracy": q_acc(member_not_retrieved),
            "target_rank_distribution": _dist_ranks(ranks),
        },
    }
    return report


def _dist_ranks(ranks: list[Any]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for rank in ranks:
        key = f"rank_{int(rank)}" if isinstance(rank, (int, float)) else str(rank)
        dist[key] = dist.get(key, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def format_report(report: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return "nan" if (value is None or (isinstance(value, float) and np.isnan(value))) else f"{value:.4f}"

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("RAG 成员推断离线评估报告")
    lines.append("=" * 64)

    c = report["counts"]
    lines.append(f"成员文档: {c['member_documents']} 篇 / {c['member_questions']} 个问题")
    lines.append(f"非成员文档: {c['nonmember_documents']} 篇 / {c['nonmember_questions']} 个问题")

    s = report["scores"]
    lines.append("")
    lines.append("--- 文档级 MIA 得分 ---")
    lines.append(f"成员均值: {pct(s['member_mean'])}   非成员均值: {pct(s['nonmember_mean'])}")
    lines.append(f"均值差: {pct(s['mean_difference'])}")
    lines.append(f"成员中位/范围: {pct(s['member_median'])} / [{pct(s['member_min'])}, {pct(s['member_max'])}]")
    lines.append(f"非成员中位/范围: {pct(s['nonmember_median'])} / [{pct(s['nonmember_min'])}, {pct(s['nonmember_max'])}]")

    lines.append("")
    lines.append("--- 区分能力 ---")
    lines.append(f"文档级 ROC-AUC: {pct(report['roc_auc'])}")
    mw = report["mann_whitney"]
    lines.append(f"Mann-Whitney U = {mw['statistic']:.2f}, p = {mw['p_value']:.4f}")

    b = report["bootstrap"]
    lines.append("Bootstrap 95% CI（得分均值差）: "
                 f"[{pct(b['mean_difference']['low'])}, {pct(b['mean_difference']['high'])}]")

    lines.append("")
    lines.append("--- 阈值判定 ---")
    for metric in report["thresholds"]:
        if not metric:
            continue
        lines.append(
            f"score {metric['rule']} {metric['threshold']:+.3f}: "
            f"acc={pct(metric['accuracy'])}, prec={pct(metric['precision'])}, "
            f"recall={pct(metric['recall'])}, fpr={pct(metric['fpr'])}"
        )

    qa = report["question_accuracy"]
    lines.append("")
    lines.append("--- 问题级回答正确率 ---")
    lines.append(f"成员: {pct(qa['member'])}   非成员: {pct(qa['nonmember'])}")

    ad = report["answer_distribution"]
    lines.append("")
    lines.append("--- RAG 回答分布 (yes/no/unknown) ---")
    lines.append(f"成员: {ad['member']}")
    lines.append(f"非成员: {ad['nonmember']}")

    r = report["retrieval"]
    lines.append("")
    lines.append("--- 目标文档检索（仅成员） ---")
    lines.append(f"目标文档检索率: {pct(r['target_retrieval_rate'])}")
    lines.append(f"检索命中时回答正确率: {pct(r['retrieved_question_accuracy'])}")
    lines.append(f"检索未命中时回答正确率: {pct(r['not_retrieved_question_accuracy'])}")
    lines.append(f"命中时排名分布: {r['target_rank_distribution']}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 成员推断离线评估")
    parser.add_argument(
        "--member", type=Path, default=Path("member_semantic_test.json"),
        help="成员结果 JSON 路径",
    )
    parser.add_argument(
        "--nonmember", type=Path, default=Path("nonmember_semantic_test.json"),
        help="非成员结果 JSON 路径",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("evaluation_report.json"),
        help="评估报告 JSON 输出路径",
    )
    parser.add_argument(
        "--unknown-penalty", type=float, default=0.5,
        help="Unknown 惩罚系数（用于报告元信息）",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="Bootstrap 随机种子",
    )
    args = parser.parse_args(argv)

    report = evaluate(
        member_path=args.member,
        nonmember_path=args.nonmember,
        unknown_penalty=args.unknown_penalty,
        bootstrap_seed=args.bootstrap_seed,
    )

    print(format_report(report))

    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估报告已写入: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
