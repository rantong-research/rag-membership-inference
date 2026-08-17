"""差分隐私 RAG 的端到端实验编排（对比方案）。

与 pipeline.py 的唯一差异：把 answer_with_rag 换成 DP 的
voter + baseline + 直方图 + 加噪（SVT）机制，其余（数据划分、探测生成、
评分、评估）完全相同。
"""

from __future__ import annotations

import json
import random

from tqdm.auto import tqdm

from src import data as data_module
from src import dp_rag
from src import embeddings as embeddings_module
from src import llm as llm_module
from src import query_generation
from src import scoring
from src.config import Config
from src.evaluation import evaluate
from src.pipeline import ensure_split, ensure_vectorstore
from src.reranker import build_reranker


def _run_group_dp(
    chat_model, client, vector_store, sampled_records, config: Config,
    membership: str, rng, reranker,
):
    detailed_docs = []
    flat_rows = []
    total_private_answers = 0
    total_private_tokens = 0

    desc = "测试成员文档(DP)" if membership == "member" else "测试非成员文档(DP)"
    for record in tqdm(sampled_records, desc=desc):
        target_text, _, _ = data_module.extract_document_text(record, config)
        target_source_line = record.get("_source_line")
        target_id = data_module.extract_original_id(record, target_source_line, config)

        probes = query_generation.generate_semantic_probes(
            chat_model,
            document_text=target_text,
            question_count=config.questions_per_document,
        )

        doc_result = {
            "membership": membership,
            "target_id": target_id,
            "target_source_line": target_source_line,
            "summary": probes["summary"],
            "questions": [],
        }

        for question_index, probe in enumerate(probes["questions"], start=1):
            question = probe["question"]
            ground_truth = probe["answer"]
            semantic_query = (
                f"{probes['summary']}\n\n{question}\n\n"
                "Please answer with Yes, No, or Unknown."
            )

            # 检索 n*k 篇（可选重排序）
            docs, scores = dp_rag.retrieve_for_dp(
                vector_store, config, semantic_query, reranker
            )

            # 目标文档诊断（在 n*k 篇里找目标）
            target_retrieved = False
            target_rank = None
            target_distance = None
            for rank, doc in enumerate(docs, start=1):
                if doc.metadata.get("source_line") == target_source_line:
                    target_retrieved = True
                    target_rank = rank
                    if scores is not None:
                        target_distance = scores[rank - 1]
                    break

            # 每次查询独立的差分隐私预算（单次查询 ε=40）
            budget = dp_rag.BudgetTracker(
                config.dp_total_budget, config.dp_per_token_budget
            )
            result = dp_rag.answer_dp_rag(
                client, config, semantic_query, docs, budget, rng
            )
            if result["answer_private"]:
                total_private_answers += 1
            total_private_tokens += result["n_private_explanation_tokens"]
            predicted_answer = result["answer"]
            answer_matches = predicted_answer == ground_truth
            score = scoring.signal_score(
                predicted_answer, ground_truth, config.unknown_penalty
            )

            top1_distance = scores[0] if scores else None
            top1_source_line = docs[0].metadata.get("source_line") if docs else None

            question_result = {
                "question_index": question_index,
                "question": question,
                "semantic_query": semantic_query,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "answer_matches": answer_matches,
                "signal_score": score,
                "target_retrieved": target_retrieved,
                "target_rank": target_rank,
                "target_distance": target_distance,
                "top1_distance": top1_distance,
                "top1_source_line": top1_source_line,
                "rag_reason": result["reason"],
                "answer_private": result["answer_private"],
                "n_private_explanation_tokens": result["n_private_explanation_tokens"],
                "budget_remaining": result["budget"]["remaining"],
                "baseline_answer": result["baseline"]["answer"],
                "voter_answers": [v["answer"] for v in result["voters"]],
            }
            doc_result["questions"].append(question_result)

            flat_rows.append({
                "membership": membership,
                "target_id": target_id,
                "target_source_line": target_source_line,
                "question_index": question_index,
                "question": question,
                "semantic_query": semantic_query,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "answer_matches": answer_matches,
                "signal_score": score,
                "target_retrieved": target_retrieved,
                "target_rank": target_rank,
                "target_distance": target_distance,
                "top1_distance": top1_distance,
                "top1_source_line": top1_source_line,
                "rag_reason": result["reason"],
                "answer_private": result["answer_private"],
                "budget_remaining": result["budget"]["remaining"],
            })

        scores = [q["signal_score"] for q in doc_result["questions"]]
        matches = [q["answer_matches"] for q in doc_result["questions"]]
        retrievals = [q["target_retrieved"] for q in doc_result["questions"]]

        doc_result["mia_score"] = scoring.aggregate_score(scores)
        doc_result["answer_match_rate"] = (
            sum(matches) / len(matches) if matches else 0.0
        )
        doc_result["target_retrieval_rate"] = (
            sum(retrievals) / len(retrievals) if retrievals else 0.0
        )
        detailed_docs.append(doc_result)

    summary = {
        "total_private_answers": total_private_answers,
        "total_private_explanation_tokens": total_private_tokens,
        "total_private_uses": total_private_answers + total_private_tokens,
    }
    return flat_rows, detailed_docs, summary


def _save(flat_rows, detailed_docs, csv_path, json_path):
    import pandas as pd

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(detailed_docs, fh, ensure_ascii=False, indent=2)


def run_dp_experiment(config: Config) -> dict:
    """执行 DP-RAG 对比实验并返回评估报告。"""
    # 1. 数据与向量库（复用非 DP 流程）
    ensure_split(config)
    embeddings = embeddings_module.create_embedding_model(config)
    vector_store = ensure_vectorstore(config, embeddings)

    # 2. 对话模型（探测生成）与本地 DP 客户端、重排序器
    chat_model = llm_module.create_chat_model(config)
    client = dp_rag.create_client(config)
    reranker = build_reranker(config)

    # 3. DP 随机源（每次查询独立创建预算）
    rng = random.Random(config.dp_random_seed)

    # 4. 抽样（与非 DP 完全相同的种子）
    members = data_module.load_jsonl(config.member_output_path)
    nonmembers = data_module.load_jsonl(config.nonmember_output_path)
    rng_member = random.Random(config.member_seed)
    rng_nonmember = random.Random(config.nonmember_seed)
    sampled_members = rng_member.sample(
        members, min(config.member_test_count, len(members))
    )
    sampled_nonmembers = rng_nonmember.sample(
        nonmembers, min(config.nonmember_test_count, len(nonmembers))
    )

    # 5. 测试成员与非成员
    member_flat, member_docs, member_budget = _run_group_dp(
        chat_model, client, vector_store, sampled_members, config, "member",
        rng, reranker,
    )
    nonmember_flat, nonmember_docs, nonmember_budget = _run_group_dp(
        chat_model, client, vector_store, sampled_nonmembers, config, "nonmember",
        rng, reranker,
    )

    _save(member_flat, member_docs, config.member_result_dp_csv, config.member_result_dp_json)
    _save(nonmember_flat, nonmember_docs, config.nonmember_result_dp_csv, config.nonmember_result_dp_json)

    # 6. 评估
    report = evaluate(config.member_result_dp_json, config.nonmember_result_dp_json)
    report["dp_budget"] = {
        "per_query_total": config.dp_total_budget,
        "per_query_token": config.dp_per_token_budget,
        "max_private_uses_per_query": int(
            config.dp_total_budget / config.dp_per_token_budget
        ),
        "member": member_budget,
        "nonmember": nonmember_budget,
    }

    with config.evaluation_report_dp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report
