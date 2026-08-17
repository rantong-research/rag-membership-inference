"""GPU Embedding 模型创建。"""

from __future__ import annotations

from src.config import Config


def create_embedding_model(config: Config):
    """创建 SentenceTransformer Embedding 模型。

    需要 CUDA 可用；否则立即报错，避免静默回退到 CPU。
    """
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，请检查 PyTorch CUDA 版本和显卡驱动。"
        )

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.device},
        encode_kwargs={
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.encode_batch_size,
        },
        # 不要放进 encode_kwargs，否则会出现 show_progress_bar 重复参数错误
        show_progress=True,
    )
