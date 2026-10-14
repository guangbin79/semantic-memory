"""嵌入器：Qwen3-Embedding-0.6B GGUF via llama-cpp-python。可选依赖，缺失时优雅降级。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("memory_mcp.embedder")


class Embedder:
    """单例式惰性加载。available=False 时所有调用方应跳过向量路径。"""

    def __init__(self, model_path: Path | None, dim: int = 1024):
        self.dim = dim
        self._llm = None
        self.available = False
        if model_path is None:
            log.info("MEMORY_MODEL_PATH 未设置，向量通道关闭（纯 BM25 模式）")
            return
        if not Path(model_path).exists():
            log.warning("模型文件不存在: %s，降级为纯 BM25", model_path)
            return
        try:
            from llama_cpp import Llama  # 可选依赖，延迟导入

            self._llm = Llama(
                model_path=str(model_path),
                embedding=True,
                n_ctx=512,
                n_gpu_layers=-1,   # 有 GPU 则全卸载，无则自动 CPU
                verbose=False,
            )
            self.available = True
            log.info("嵌入模型已加载: %s", model_path)
        except Exception as e:  # llama_cpp 未安装 / 加载失败均降级
            log.warning("嵌入模型加载失败（%s），降级为纯 BM25", e)

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """返回 (n, dim) float32，已 L2 归一化；不可用时返回 None。"""
        if not self.available or not texts:
            return None
        vecs = []
        for t in texts:
            out = self._llm.embed(t[:4000])  # 截断保护
            vecs.append(np.asarray(out, dtype=np.float32))
        mat = np.stack(vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
