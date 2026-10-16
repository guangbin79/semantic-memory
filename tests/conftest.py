"""测试封闭性：强制重排关闭。

config 默认在门禁 PASSED 后指向常规模型路径（config.py 注释）；本机若存在该
文件，单测会意外加载真模型（慢 + GPU 依赖）。全套件强制 MEMORY_RERANK_MODEL=
空 = 关闭；需要真件的测试自行覆盖（真件路径按 opt-in 惯例走环境显式指定）。
"""
import os

os.environ["MEMORY_RERANK_MODEL"] = ""  # noqa: E402 —— 必须在 import memory_mcp 前生效
