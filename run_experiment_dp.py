"""差分隐私 RAG 对比实验入口（需要 GPU + .env）。

用法:
    python run_experiment_dp.py            # bge-base-en-v1.5（768 维）
    python run_experiment_dp.py --legacy   # 旧 chroma_real_8k 配置
"""

from __future__ import annotations

import argparse
import json

from src.config import DEFAULT, LEGACY_BGE_SMALL_ZH, Config
from src.evaluation import format_report
from src.pipeline_dp import run_dp_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="运行差分隐私 RAG 对比实验")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用与现有 chroma_real_8k 匹配的旧配置",
    )
    args = parser.parse_args()

    config: Config = LEGACY_BGE_SMALL_ZH if args.legacy else DEFAULT

    print("[INFO] 实验配置:")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    report = run_dp_experiment(config)
    print(format_report(report))
    print("\n[INFO] DP 预算使用:", report.get("dp_budget"))
    print(f"[INFO] 评估报告已写入: {config.evaluation_report_dp.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
