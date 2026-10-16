"""python -m semantic_memory 入口。

双形态：包内运行（python3 -m semantic_memory）走相对导入；
源码直跑（python3 src/semantic_memory/__main__.py，未安装场景）时
__package__ 为空——把 src/ 挂上 sys.path 后按绝对包名导入。
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # 源码直跑
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semantic_memory.server import main
else:
    from .server import main

if __name__ == "__main__":
    main()
