"""Chroma 向量库的构建与加载。"""

from __future__ import annotations

from src.config import Config


def build_vectorstore(config: Config, embeddings, documents, document_ids):
    """从零构建 Chroma 向量库并持久化。"""
    if (
        config.persist_directory.exists()
        and any(config.persist_directory.iterdir())
    ):
        raise RuntimeError(
            f"向量库目录已经存在且不为空："
            f"{config.persist_directory.resolve()}\n"
            "为避免重复写入，请更换目录名称，"
            "或者确认后手动删除旧目录。"
        )

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_directory),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    add_documents_batched(store, documents, document_ids)
    return store


def add_documents_batched(store, documents, document_ids, batch_size: int = 4000):
    """分批写入文档，避免超过 Chroma 的 max_batch_size（默认 5461）。"""
    total = len(documents)
    for start in range(0, total, batch_size):
        store.add_documents(
            documents=documents[start:start + batch_size],
            ids=document_ids[start:start + batch_size],
        )
    return store


def load_vectorstore(config: Config, embeddings):
    """加载已持久化的 Chroma 向量库。

    注意：加载时必须使用与构建时完全相同的 Embedding 模型与维度，
    否则会因向量维度不一致而报错。
    """
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_directory),
        embedding_function=embeddings,
    )


def collection_count(store) -> int:
    """返回向量库中已存储的文档数量；异常时返回 0。"""
    try:
        return int(store._collection.count())
    except Exception:
        return 0
