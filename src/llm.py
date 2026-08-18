"""大语言模型封装：模型构建、消息文本提取与 JSON 解析。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from src.config import Config


def create_chat_model(config: Config):
    """通过 OpenAI 兼容接口构建对话模型。"""
    load_dotenv()

    api_key = config.api_key or os.getenv(config.api_key_env, "")
    base_url = config.base_url or os.getenv(config.base_url_env, "")

    if not api_key or not base_url:
        raise RuntimeError(
            f"未配置 api_key / base_url（config 或环境变量）。"
        )

    from langchain.chat_models import init_chat_model

    extra_body: dict[str, Any] = {}
    if not config.enable_thinking:
        # 关闭思考模式，减少输出波动与额外开销
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    return init_chat_model(
        model=config.chat_model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature,
        extra_body=extra_body,
    )


def get_message_text(message: Any) -> str:
    """从 AIMessage（或裸对象）中提取纯文本。"""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
        return "\n".join(texts).strip()

    return str(content).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出中稳健地解析 JSON 对象（容忍 Markdown 代码块与前后缀）。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型没有返回 JSON 对象：{text}")

    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 模型偶发输出非法转义（如 "\x"、"\ " 等）：把后面不跟合法转义字符的
        # 单个反斜杠补成双反斜杠后重试，避免整个实验因一处坏 JSON 而中断。
        fixed = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', text)
        return json.loads(fixed)
