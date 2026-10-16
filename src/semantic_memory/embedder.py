"""嵌入器：与 ctx-semantic 完全同源的向量与方式（Qwen3-Embedding-0.6B Q8_0 GGUF）。

对齐 ctx-semantic.embedder 的约定：
- 模型：Qwen3-Embedding-0.6B Q8_0 GGUF，dim 1024，pooling LAST（显式指定）；
- 非对称嵌入：查询侧加 Qwen3 Instruct 前缀，文档侧原文（调用方传原文）；
- n_ctx=8192，批大小 32 子批，输出 float32 按行 L2 归一化（1e-12 保护）；
- 设备回退确定化：GPU 尝试 → 探针嵌入 → nvidia-smi PID 验证（识别静默
  CPU 回退）→ 任何失败重建于 CPU。GPU 期嵌入失败（如 OOM）重建 CPU 重试一次。

与 ctx-semantic 的唯一差异（本项目规格）：**GPU 为默认，CPU 为可选**——
ctx-semantic 以安装 profile 区分（cpu 默认/cuda 可选），本项目以运行时
环境变量 MEMORY_DEVICE 区分（cuda 默认/cpu 可选），同一份代码两条路径。
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

from ._compat import INSTALL_HINT, HAS_NUMPY, np, npt

log = logging.getLogger("semantic_memory.embedder")

MODEL_NAME: Final = "qwen/qwen3-embedding-0.6b-gguf-q8"  # 与 ctx-semantic 相同的向量键
DIM: Final = 1024
GGUF_NAME: Final = "Qwen3-Embedding-0.6B-Q8_0.gguf"
QUERY_INSTRUCT: Final = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
EMBED_BATCH_SIZE: Final = 32
N_CTX: Final = 8192


def format_query(text: str) -> str:
    """Qwen3 查询侧 Instruct 前缀（文档侧保持原文）。"""
    return f"Instruct: {QUERY_INSTRUCT}\nQuery: {text}"


def _l2_normalized(mat: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    m = mat.astype(np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _pid_vram_on_gpu(pid: int) -> str | None:
    """nvidia-smi 报告的该 pid 显存占用；不在 GPU 上返回 None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("nvidia-smi 查询失败: %s", exc)
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(pid):
            return parts[1] if len(parts) > 1 else "unknown"
    return None


def _preload_cuda_libs() -> None:
    """尽力预装 nvidia-wheel 的 CUDA 运行库（RTLD_GLOBAL）。

    仅对使用 nvidia wheel 分发 CUDA 运行时的安装（如 cu124 wheel）必要；
    系统 CUDA 或 CPU wheel 下 glob 不到任何库，静默跳过——不以此为 GPU 门禁。
    """
    for pattern in (
        "lib/**/nvidia/cuda_runtime/lib/libcudart.so.12",
        "lib/**/nvidia/cublas/lib/libcublas.so.12",
        "lib/**/nvidia/cublas/lib/libcublasLt.so.12",
    ):
        hits = sorted(glob.glob(str(Path(sys.prefix) / pattern), recursive=True))
        if hits:
            try:
                ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                log.warning("CUDA 运行库预载失败 %s: %s", hits[0], exc)


class Embedder:
    """惰性加载。available=False 时调用方跳过向量通道（纯 BM25 运行）。

    device 语义：
    - "cuda"（默认）：尝试 GPU，nvidia-smi 验证失败或任何异常 → CPU 重建；
    - "cpu"：直接 n_gpu_layers=0。
    """

    def __init__(
        self,
        model_path: Path | None,
        dim: int = DIM,
        device: str = "cuda",
    ):
        self.dim = dim
        self.prefer = device          # 用户意图：cuda|cpu
        self.device: str | None = None  # 实际运行设备：cuda|cpu|None
        self._gguf_path = Path(model_path) if model_path else None
        self._model = None
        self.available = False
        if not HAS_NUMPY:
            log.warning("numpy 未安装，向量通道关闭（纯 BM25 模式）：%s", INSTALL_HINT)
        elif self._gguf_path is None:
            log.info("MEMORY_MODEL_PATH 未设置，向量通道关闭（纯 BM25 模式）")
        elif not self._gguf_path.exists():
            log.warning("模型文件不存在: %s，向量通道关闭", self._gguf_path)
        else:
            self.available = True  # 惰性：首次 embed 才真正加载

    # ---------- 加载与回退 ----------

    def _new_model(self, use_gpu: bool):
        import llama_cpp  # 延迟导入：非嵌入路径不付 CUDA dlopen 成本

        return llama_cpp.Llama(
            model_path=str(self._gguf_path),
            embedding=True,
            n_gpu_layers=-1 if use_gpu else 0,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_ctx=N_CTX,
            verbose=False,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.prefer == "cpu":
            self._model = self._new_model(False)
            probe = np.stack(self._model.embed(["device probe"]))
            self.device = "cpu"
        else:
            _preload_cuda_libs()
            try:
                model = self._new_model(True)
                probe = np.stack(model.embed(["device probe"]))
                vram = _pid_vram_on_gpu(os.getpid())
                if vram is None:
                    raise RuntimeError(
                        "CUDA 会话已建立但 nvidia-smi 无此 pid 显存——"
                        "判定为静默 CPU 回退，拒绝声称 GPU"
                    )
                self._model, self.device = model, "cuda"
                log.info("嵌入器运行于 GPU: pid=%s vram=%s", os.getpid(), vram)
            except Exception as exc:
                log.warning("GPU 初始化失败（%s: %s）——回退 CPU 嵌入器",
                            type(exc).__name__, exc)
                self._model = self._new_model(False)
                probe = np.stack(self._model.embed(["device probe"]))
                self.device = "cpu"
        assert probe.shape == (1, self.dim), (
            f"{MODEL_NAME} 嵌入维度 {probe.shape}，期望 (1, {self.dim})"
        )

    def _rebuild_on_cpu(self, exc: Exception) -> None:
        log.warning("GPU 嵌入失败（%s: %s）——CPU 重建", type(exc).__name__, exc)
        self._model = self._new_model(False)
        self.device = "cpu"

    def _embed_all(self, texts: list[str]) -> npt.NDArray[np.floating]:
        mats = [
            np.stack(self._model.embed(texts[i : i + EMBED_BATCH_SIZE]))
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]
        return np.concatenate(mats)

    # ---------- 对外接口 ----------

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32] | None:
        """文档侧：原文嵌入 -> (n, 1024) float32 行归一化；不可用返回 None。"""
        if not self.available or not texts:
            return None
        self._ensure_loaded()
        try:
            mat = self._embed_all(texts)
        except Exception as exc:
            if self.device != "cuda":
                raise
            self._rebuild_on_cpu(exc)
            mat = self._embed_all(texts)
        return _l2_normalized(mat)

    def embed_query(self, text: str) -> npt.NDArray[np.float32] | None:
        """查询侧：内部自动加 Instruct 前缀，调用方传原始查询文本。"""
        mat = self.embed_batch([format_query(text)])
        return mat[0] if mat is not None else None
