"""端到端实验入口（需要 GPU + .env 配置的 api_key/base_url）。

用法:
    python run_experiment.py            # 使用 README 推荐的 bge-base-en-v1.5 配置
    python run_experiment.py --legacy   # 使用与现有 chroma_real_8k 匹配的旧配置
"""

from __future__ import annotations

import argparse
import json

from src.config import DEFAULT, LEGACY_BGE_SMALL_ZH, Config
from src.evaluation import format_report
from src.pipeline import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 RAG 成员推断实验")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用与现有 chroma_real_8k（bge-small-zh-v1.5, 512 维）匹配的配置",
    )
    args = parser.parse_args()

    config: Config = LEGACY_BGE_SMALL_ZH if args.legacy else DEFAULT

    print("[INFO] 实验配置:")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    report = run_experiment(config)
    print(format_report(report))
    print(f"\n[INFO] 评估报告已写入: {config.evaluation_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
