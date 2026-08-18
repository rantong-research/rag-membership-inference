"""严格逐 token 自回归生成（正确版，使用 partial=true 续写）。

核心：每步只输出 1 个 token，把 prefix 作为 assistant 消息（带 partial=true）
append 回消息，让模型从 prefix 后面续写下一个 token。

关键点：
1. assistant 消息加 "partial": true，表示这是"未完成的前缀"，请继续写；
2. 用 max_tokens=1（部分接口用 max_completion_tokens=1）限制每步只出 1 个 token；
3. 关闭 thinking：extra_body={"chat_template_kwargs": {"enable_thinking": False}}；
4. 结束判断用 token == "" 或 finish_reason == "stop"，不要用 token.strip()。

用法：python test_strict_token.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "qwen3.5-plus"          # 可改 qwen3.8-max / qwen-plus / qwen3-4b
MAX_STEPS = 24

client = OpenAI(
    base_url=os.getenv("base_url"),
    api_key=os.getenv("api_key"),
)

BASE_MESSAGES = [
    {
        "role": "system",
        "content": "Explain a yes/no/unknown answer in one short sentence of 3-8 words.",
    },
    {
        "role": "user",
        "content": (
            'Question: Does the described XSS vulnerability affect Foo version 1.0?\n'
            '\n'
            'The answer is "yes".\n'
            '\n'
            'Contexts:\n'
            '[1] A reflected XSS in Foo version 1.0 allows script injection via the \'q\' parameter.\n'
            '\n'
            'Reason:'
        ),
    },
]


def generate_one_token(messages):
    """单次 API 调用，返回 (token, finish_reason)。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    token = choice.message.content or ""
    return token, choice.finish_reason


def generate_strict_token_by_token(base_messages, max_steps=24):
    """严格逐 token：每步单独一次 API 调用，prefix 带 partial=true 续写。"""
    prefix = ""
    for step in range(1, max_steps + 1):
        messages = [dict(m) for m in base_messages]
        if prefix:
            messages.append({
                "role": "assistant",
                "content": prefix,
                "partial": True,       # 关键：标记为未完成前缀
            })

        try:
            token, finish = generate_one_token(messages)
        except Exception as e:
            print(f"step {step:2d}: 异常 {type(e).__name__}: {e}")
            break

        print(
            f"step {step:2d}: token={token!r} "
            f"finish={finish} prefix={prefix + token!r}"
        )

        # 结束判断：token 为空 或 finish_reason=stop（不要用 token.strip()）
        if token == "" or finish == "stop":
            break
        prefix += token

    return prefix


if __name__ == "__main__":
    result = generate_strict_token_by_token(BASE_MESSAGES, MAX_STEPS)
    print("\n最终文本:", repr(result))
