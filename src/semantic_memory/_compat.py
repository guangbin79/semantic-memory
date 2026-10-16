"""可选 numpy 守门：零依赖基座的核心触点。

在场：np/npt 为真模块、HAS_NUMPY=True —— 与直接 import 逐字节等价。
缺席：np/npt 为 None —— 向量腿整体关闭（embedder.available=False），
BM25/FTS 通道不受影响；/health 与 /api/stats 附 install_hint 字段。
"""
from __future__ import annotations

try:
    import numpy as np
    import numpy.typing as npt

    HAS_NUMPY = True
except ImportError:  # 零依赖基座：裸 python3 也可跑（纯 BM25）
    np = None  # type: ignore[assignment]
    npt = None  # type: ignore[assignment]
    HAS_NUMPY = False

INSTALL_HINT: str = (
    "numpy is not installed - vector channel disabled (BM25-only). "
    "Install with: pip install 'semantic-memory[vector]'"
    " (or [embed] for the full embedding stack)"
)
