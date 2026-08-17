"""离线评估入口：对既有成员/非成员结果做统计，不依赖 GPU 或模型。

用法:
    python evaluate.py
    python evaluate.py --member member_semantic_test.json \
        --nonmember nonmember_semantic_test.json --out evaluation_report.json
"""

from src.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
