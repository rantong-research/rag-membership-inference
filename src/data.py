"""数据读取、成员/非成员划分与 LangChain Document 构建。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from src.config import Config


def iter_jsonl(file_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """逐行遍历 jsonl，产出 (行号, 记录)；空行自动跳过。"""
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到数据文件: {file_path.resolve()}")

    with file_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第 {line_number} 行不是合法 JSON：{error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            yield line_number, record


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """读取整个 jsonl，返回记录列表。"""
    return [record for _, record in iter_jsonl(file_path)]


def find_first_string_field(
    record: dict[str, Any],
    candidates: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """返回第一个非空字符串字段及其取值。"""
    for field in candidates:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return None, None


def extract_document_text(
    record: dict[str, Any],
    config: Config,
    source_line: int | None = None,
) -> tuple[str, str | None, str | None]:
    """从记录中抽取正文并组装 page_content。"""
    text_field, text = find_first_string_field(
        record, config.text_field_candidates
    )
    if text is None:
        where = f"第 {source_line} 行" if source_line else "记录"
        raise ValueError(
            f"{where} 找不到文档文本字段，当前字段为: "
            f"{list(record.keys())}。请修改 text_field_candidates。"
        )

    title_field, title = find_first_string_field(
        record, config.title_field_candidates
    )
    page_content = f"{title}\n\n{text}" if title else text
    return page_content, text_field, title_field


def extract_original_id(
    record: dict[str, Any],
    source_line: int | None,
    config: Config,
) -> str:
    """提取文档原始 ID；缺失时回退到源文件行号。"""
    for field in config.id_field_candidates:
        value = record.get(field)
        if value is not None:
            return str(value)

    embedded = record.get("_source_line")
    if embedded is not None:
        return f"source_line_{embedded}"

    if source_line is not None:
        return f"source_line_{source_line}"
    return "unknown"


def split_members_and_nonmembers(
    records: list[dict[str, Any]],
    member_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按固定种子均匀随机划分成员/非成员。

    records 每一项需包含 source_line 与 record 两个键。
    """
    total = len(records)
    if total < member_count:
        raise ValueError(
            f"数据集只有 {total} 条，无法抽取 {member_count} 条。"
        )

    rng = random.Random(seed)
    member_positions = set(rng.sample(range(total), member_count))

    members: list[dict[str, Any]] = []
    nonmembers: list[dict[str, Any]] = []
    for position, item in enumerate(records):
        bucket = members if position in member_positions else nonmembers
        bucket.append(item)

    rng.shuffle(members)
    rng.shuffle(nonmembers)
    return members, nonmembers


def save_jsonl_split(
    items: list[dict[str, Any]],
    output_path: Path,
    membership: str,
) -> None:
    """把划分结果写回 jsonl，并注入 _source_line 与 _membership。"""
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            out = dict(item["record"])
            out["_source_line"] = item["source_line"]
            out["_membership"] = membership
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def split_and_save(config: Config) -> None:
    """从 real_10k.jsonl 划分成员/非成员，并保存 jsonl 与 manifest。"""
    records = [
        {"source_line": line_number, "record": record}
        for line_number, record in iter_jsonl(config.data_path)
    ]

    members, nonmembers = split_members_and_nonmembers(
        records, config.knowledge_base_size, config.data_seed
    )

    save_jsonl_split(members, config.member_output_path, "member")
    save_jsonl_split(nonmembers, config.nonmember_output_path, "nonmember")

    manifest = {
        "source_file": str(config.data_path.resolve()),
        "random_seed": config.data_seed,
        "total_documents": len(records),
        "member_count": len(members),
        "nonmember_count": len(nonmembers),
        "member_source_lines": [item["source_line"] for item in members],
        "nonmember_source_lines": [item["source_line"] for item in nonmembers],
    }
    with config.split_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def build_member_documents(
    config: Config,
) -> tuple[list[Any], list[str]]:
    """从已保存的成员 jsonl 构建 LangChain Document 与 Chroma ID。"""
    from langchain_core.documents import Document

    members = load_jsonl(config.member_output_path)
    documents: list[Any] = []
    document_ids: list[str] = []

    for record in tqdm(members, desc="准备成员文档"):
        source_line = record.get("_source_line")

        page_content, text_field, title_field = extract_document_text(
            record, config, source_line
        )
        original_id = extract_original_id(record, source_line, config)

        # 使用源文件行号生成确定且唯一的 Chroma ID
        chroma_id = f"real_doc_{source_line}"
        metadata = {
            "source": str(config.data_path),
            "source_line": source_line,
            "original_id": original_id,
            "membership": "member",
            "text_field": text_field,
        }
        if title_field is not None:
            metadata["title_field"] = title_field

        documents.append(
            Document(page_content=page_content, metadata=metadata)
        )
        document_ids.append(chroma_id)

    return documents, document_ids
