"""端到端实验编排：加载知识库 → 抽样 → 生成探测 → RAG 回答 → 评分 → 评估。

成员与非成员共用同一套问题生成、检索、回答与评分逻辑，保证两者输出
字段完全一致（统一使用 answer_matches 与 target_* 诊断字段）。
"""

from __future__ import annotations

import json
import random

from tqdm.auto import tqdm

from src import data as data_module
from src import embeddings as embeddings_module
from src import llm as llm_module
from src import query_generation
from src import rag
from src import reranker
from src import scoring
from src import vectorstore
from src.config import Config
from src.evaluation import evaluate


def _run_group(chat_model, vector_store, sampled_records, config: Config, membership: str, reranker_obj=None):
    """对一组文档执行语义探测 + RAG 回答，返回 (扁平行, 详细文档)。"""
    detailed_docs = []
    flat_rows = []

    desc = "测试成员文档" if membership == "member" else "测试非成员文档"
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

            retrieved_documents, retrieval_scores = reranker.retrieve(
                vector_store, config, semantic_query, config.retrieve_k, reranker_obj
            )

            retrieved_info = []
            target_retrieved = False
            target_rank = None
            target_distance = None

            for rank, retrieved_document in enumerate(retrieved_documents, start=1):
                distance = (
                    retrieval_scores[rank - 1] if retrieval_scores else None
                )
                retrieved_source_line = (
                    retrieved_document.metadata.get("source_line")
                )
                is_target = (
                    target_source_line is not None
                    and retrieved_source_line == target_source_line
                )
                if is_target:
                    target_retrieved = True
                    target_rank = rank
                    target_distance = distance

                retrieved_info.append({
                    "rank": rank,
                    "distance": distance,
                    "is_target": is_target,
                    "metadata": retrieved_document.metadata,
                    "content_preview": retrieved_document.page_content[:300],
                })

            rag_result = rag.answer_with_rag(
                chat_model,
                semantic_query=semantic_query,
                retrieved_documents=retrieved_documents,
            )
            predicted_answer = rag_result["answer"]
            answer_matches = predicted_answer == ground_truth
            score = scoring.signal_score(
                predicted_answer, ground_truth, config.unknown_penalty
            )

            top1_distance = retrieval_scores[0] if retrieval_scores else None
            top1_source_line = (
                retrieved_documents[0].metadata.get("source_line")
                if retrieved_documents else None
            )

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
                "rag_reason": rag_result["reason"],
                "raw_response": rag_result["raw_response"],
                "retrieved_documents": retrieved_info,
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
                "rag_reason": rag_result["reason"],
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

    return flat_rows, detailed_docs


def _save(flat_rows, detailed_docs, csv_path, json_path):
    import pandas as pd

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(detailed_docs, fh, ensure_ascii=False, indent=2)


def ensure_split(config: Config) -> None:
    """若成员/非成员划分文件缺失或规模与配置不符，则从 real_10k.jsonl 重新划分。"""
    needed = (
        config.member_output_path,
        config.nonmember_output_path,
        config.split_manifest_path,
    )
    if all(path.exists() for path in needed):
        try:
            with config.split_manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if manifest.get("member_count") == config.knowledge_base_size:
                return
            print("[INFO] 划分规模与配置不一致，重新划分。")
        except Exception:
            print("[INFO] 划分清单读取失败，重新划分。")
    else:
        print("[INFO] 划分文件缺失，从 real_10k.jsonl 重新划分。")
    data_module.split_and_save(config)


def ensure_vectorstore(config: Config, embeddings):
    """加载已有向量库；若目录缺失或集合为空，则构建/补齐 8000 条向量。"""
    if (
        config.persist_directory.exists()
        and any(config.persist_directory.iterdir())
    ):
        store = vectorstore.load_vectorstore(config, embeddings)
        count = vectorstore.collection_count(store)
        if count > 0:
            print(f"[INFO] 加载已有向量库，文档数: {count}")
            return store
        # 空集合：不删除目录，直接向现有集合写入（避免 Windows 文件占用 WinError 32）
        print("[INFO] 检测到空向量库，直接向现有集合写入 8000 条。")
        documents, document_ids = data_module.build_member_documents(config)
        vectorstore.add_documents_batched(store, documents, document_ids)
        print(f"[INFO] 向量库构建完成，文档数: {vectorstore.collection_count(store)}")
        return store

    # 目录缺失：从零构建
    documents, document_ids = data_module.build_member_documents(config)
    store = vectorstore.build_vectorstore(
        config, embeddings, documents, document_ids
    )
    print(f"[INFO] 向量库构建完成，文档数: {vectorstore.collection_count(store)}")
    return store


def run_experiment(config: Config) -> dict:
    """执行完整实验并返回评估报告。"""
    # 1. 确保数据划分文件存在
    ensure_split(config)

    # 2. 构建/加载向量库
    embeddings = embeddings_module.create_embedding_model(config)
    vector_store = ensure_vectorstore(config, embeddings)

    # 3. 构建对话模型与重排序器
    chat_model = llm_module.create_chat_model(config)
    reranker_obj = reranker.build_reranker(config)

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

    member_flat, member_docs = _run_group(
        chat_model, vector_store, sampled_members, config, "member", reranker_obj
    )
    nonmember_flat, nonmember_docs = _run_group(
        chat_model, vector_store, sampled_nonmembers, config, "nonmember", reranker_obj
    )

    _save(
        member_flat, member_docs,
        config.member_result_csv, config.member_result_json,
    )
    _save(
        nonmember_flat, nonmember_docs,
        config.nonmember_result_csv, config.nonmember_result_json,
    )

    report = evaluate(config.member_result_json, config.nonmember_result_json)

    with config.evaluation_report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report
