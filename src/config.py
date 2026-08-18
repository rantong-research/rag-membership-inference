"""集中配置：所有路径、模型与实验参数的单点来源。

说明
----
- 当前已存在的向量库 chroma_real_8k（集合 real_8k_members）是用
  BAAI/bge-small-zh-v1.5（512 维）构建的，与 README 推荐不一致。
- 推荐改用 BAAI/bge-base-en-v1.5（768 维）重建知识库，并换用新的
  持久化目录；不同维度的向量绝不能混在同一个 Chroma 集合里。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """实验配置。修改此处即可让整条流水线保持一致。"""

    # ---- 数据 ----
    data_path: Path = Path("real_10k.jsonl")
    member_output_path: Path = Path("real_9500_members.jsonl")
    nonmember_output_path: Path = Path("real_500_nonmembers.jsonl")
    split_manifest_path: Path = Path("real_10k_split.json")

    knowledge_base_size: int = 9500
    data_seed: int = 42

    # ---- Embedding / 向量库 ----
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    device: str = "cuda"
    normalize_embeddings: bool = True
    encode_batch_size: int = 64
    persist_directory: Path = Path("./chroma_bge_base_en_v15_9500")
    collection_name: str = "real_9500_bge_base_en"

    # ---- 大语言模型（OpenAI 兼容接口，通过 .env 配置）----
    # chat_model / api_key / base_url 留空时，从环境变量（api_key_env /
    # base_url_env / chat_model_env）读取；不建议在公开仓库中写死本地模型参数。
    chat_model: str = ""
    temperature: float = 0.0
    enable_thinking: bool = False
    api_key: str = ""
    base_url: str = ""
    api_key_env: str = "api_key"
    base_url_env: str = "base_url"
    chat_model_env: str = "chat_model"

    # ---- 测试 ----
    member_test_count: int = 50
    nonmember_test_count: int = 50
    questions_per_document: int = 3
    retrieve_k: int = 10
    member_seed: int = 2027
    nonmember_seed: int = 2026
    unknown_penalty: float = 0.5

    # ---- 结果 ----
    member_result_csv: Path = Path("member_semantic_test.csv")
    member_result_json: Path = Path("member_semantic_test.json")
    nonmember_result_csv: Path = Path("nonmember_semantic_test.csv")
    nonmember_result_json: Path = Path("nonmember_semantic_test.json")
    evaluation_report: Path = Path("evaluation_report.json")

    # ---- 差分隐私 RAG 结果（对比方案）----
    member_result_dp_csv: Path = Path("member_semantic_test_dp.csv")
    member_result_dp_json: Path = Path("member_semantic_test_dp.json")
    nonmember_result_dp_csv: Path = Path("nonmember_semantic_test_dp.csv")
    nonmember_result_dp_json: Path = Path("nonmember_semantic_test_dp.json")
    evaluation_report_dp: Path = Path("evaluation_report_dp.json")

    # ---- 差分隐私 RAG（对比方案）----
    n_voters: int = 30
    dp_retrieve_k: int = 2
    max_chunk_chars: int = 800
    dp_total_budget: float = 40.0
    dp_per_token_budget: float = 2.0
    dp_threshold_ratio: float = 0.5
    dp_answer_threshold_ratio: float = 0.0
    # 严格逐 token 解释需要 assistant 消息的 "partial": True 续写能力（DashScope 专有）。
    # 本地 vLLM（OpenAI 兼容）不识别该字段，且 max_tokens=1 贪心会退化成
    # "TheTheThe..."，故本地置 False，走「一次性生成 + 逐词 DP 选择」回退路径。
    dp_strict_per_token: bool = False
    max_explanation_tokens: int = 24
    dp_random_seed: int = 2028

    # ---- 重排序（可选，两种方案通用）----
    use_reranker: bool = True
    coarse_recall_factor: int = 2
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- 字段候选 ----
    text_field_candidates: tuple[str, ...] = (
        "text", "content", "page_content", "document", "body",
    )
    id_field_candidates: tuple[str, ...] = (
        "id", "doc_id", "_id", "document_id",
    )
    title_field_candidates: tuple[str, ...] = (
        "title", "name", "document_title",
    )

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的配置清单，用于复现实验（README §12）。"""
        result: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result


# README 推荐的正式配置：英文漏洞语料使用 bge-base-en-v1.5（768 维）。
DEFAULT = Config()

# 与现有产物（chroma_real_8k / real_8k_members / 既有结果 JSON）匹配的旧配置。
LEGACY_BGE_SMALL_ZH = Config(
    embedding_model="BAAI/bge-small-zh-v1.5",
    embedding_dim=512,
    persist_directory=Path("./chroma_real_8k"),
    collection_name="real_8k_members",
)
