"""RAG 黑盒成员推断实验：模块化实现。

模块概览
--------
- config:         集中配置（路径、模型、实验参数）
- data:           数据读取、成员/非成员划分、Document 构建
- llm:            大语言模型构建与输出解析
- embeddings:     GPU Embedding 模型创建
- vectorstore:    Chroma 向量库构建与加载
- query_generation: 语义 Yes/No 探测问题生成
- rag:            RAG 检索增强回答与答案归一化
- scoring:        成员信号分数计算
- evaluation:     离线统计评估（ROC-AUC、阈值指标、Bootstrap）
- pipeline:       端到端实验编排
"""

__version__ = "0.2.0"
