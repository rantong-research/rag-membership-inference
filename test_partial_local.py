"""测试本地 OpenAI 兼容服务（vLLM）是否支持 assistant 消息的 "partial" 字段。

背景："partial": True 是 DashScope 专有字段，用于标记 assistant 消息是「未完成前缀」，
要求模型从该前缀继续生成下一个 token。严格逐 token 的 DP 解释依赖此能力。

测试内容：
  A) 逐 token + partial=True    —— 每步 max_tokens=1，prefix 作为 assistant 消息并带 partial
  B) 逐 token + 普通 assistant  —— 同样每步 max_tokens=1，但不带 partial 字段
  C) 一次性生成 (max_tokens=32) —— 对照：确认模型本身能生成连贯文本

判定：
  - 若 A 与 B 输出几乎相同 → partial 字段被本地服务忽略（不支持）。
  - 若 A/B 逐 token 退化成 "TheTheThe..." 而 C 连贯 → 退化来自 max_tokens=1 贪心本身，
    与 partial 无关。结论：本地应关闭严格逐 token（dp_strict_per_token=False），
    改走「一次性生成 + 逐词 DP 选择」。

用法：python test_partial_local.py [base_url] [model]
默认：http://localhost:6006/v1  qwen3-4b
"""
import sys

from openai import OpenAI

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:6006/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-4b"
MAX_STEPS = 12

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

BASE_MESSAGES = [
    {"role": "system", "content": "Explain a yes/no/unknown answer in one short sentence of 3-8 words."},
    {"role": "user", "content": (
        'Question: Does the described XSS vulnerability affect Foo version 1.0?\n'
        '\nThe answer is "yes".\n\nContexts:\n'
        "[1] A reflected XSS in Foo version 1.0 allows script injection via the 'q' parameter.\n"
        '\nReason:'
    )},
]


def one_token(messages):
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0, max_tokens=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    c = resp.choices[0]
    return (c.message.content or ""), c.finish_reason


def run_stepwise(use_partial):
    label = "A partial=True" if use_partial else "B plain assistant"
    print(f"\n=== {label} ===")
    prefix = ""
    for step in range(1, MAX_STEPS + 1):
        msgs = [dict(m) for m in BASE_MESSAGES]
        if prefix:
            amsg = {"role": "assistant", "content": prefix}
            if use_partial:
                amsg["partial"] = True
            msgs.append(amsg)
        try:
            tok, finish = one_token(msgs)
        except Exception as e:
            print(f"  step {step:2d}: EXC {type(e).__name__}: {str(e)[:120]}")
            break
        print(f"  step {step:2d}: tok={tok!r} finish={finish}")
        if tok == "" or finish == "stop":
            break
        prefix += tok
    return prefix


def run_oneshot():
    print("\n=== C one-shot (max_tokens=32, thinking off) ===")
    resp = client.chat.completions.create(
        model=MODEL, messages=BASE_MESSAGES, temperature=0, max_tokens=32,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    c = resp.choices[0]
    print(f"  finish={c.finish_reason}\n  content={c.message.content!r}")
    return c.message.content or ""


if __name__ == "__main__":
    print(f"model={MODEL} base_url={BASE_URL}")
    a = run_stepwise(use_partial=True)
    b = run_stepwise(use_partial=False)
    c = run_oneshot()
    print("\n--- 结论 ---")
    print("A partial=True  :", repr(a))
    print("B plain         :", repr(b))
    print("C one-shot      :", repr(c))
    print("partial 被识别 ?", "否（A 与 B 一致，字段被忽略）" if a == b else "是")
