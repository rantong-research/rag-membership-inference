"""Cross-Encoder 重排序（可选，两种 RAG 方案通用）。

先粗召回 coarse_k = final_k * coarse_recall_factor 篇，再用 Cross-Encoder
对「查询, 片段」打分，取 top final_k 篇。
"""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[Any], top_k: int) -> list[Any]:
        if not documents:
            return documents
        model = self._load()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(scores, documents), key=lambda x: float(x[0]), reverse=True
        )
        return [doc for _, doc in ranked[:top_k]]


def build_reranker(config):
    """按配置返回重排序器；未开启时返回 None。"""
    if not config.use_reranker:
        return None
    return CrossEncoderReranker(config.cross_encoder_model)


def retrieve(vector_store, config, query, k, reranker=None):
    """通用检索：可选「粗召回 2k → Cross-Encoder 重排到 k」。

    返回 (docs, scores)：
    - 无重排时 scores 为向量距离列表；
    - 有重排时 scores 为 None（重排分无绝对距离意义）。
    """
    if reranker is None:
        results = vector_store.similarity_search_with_score(query, k=k)
        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        return docs, scores

    coarse_k = k * config.coarse_recall_factor
    docs = vector_store.similarity_search(query, k=coarse_k)
    docs = reranker.rerank(query, docs, k)
    return docs, None
