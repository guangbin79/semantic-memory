# 验证环境导入面（verify-env）

> src/ 改动后的验证语义：pytest 与裸 CLI 走不同导入面——CLI 试跑必须显式 PYTHONPATH。

## 导入面二分
- pytest：pyproject `[tool.pytest.ini_options] pythonpath=["src"]` → 测试永远打 src/ 新代码。
- `venv/bin/python3.12 -m semantic_memory ...`：走 venv site-packages 的**安装拷贝**（非
  editable）→ 不含未安装的 src 改动；不认识的新子命令会落回 serve 分支，副作用＝顺带跑
  startup maintenance＋project key backfill 后绑端口（端口被真服务占用时报
  `OSError: Address already in use`，误导诊断方向）。
- CLI 验证正道：`PYTHONPATH=src venv/bin/python3.12 -m semantic_memory <cmd>`。
- 根治选项（未做，记录待决策）：`pip install -e .` 转 editable——影响共享 venv 的其他
  会话，不做单侧静默切换。

## 实证
- 2026-10-24 gt-61cf：sweep 子命令落地后裸 `-m` 试跑落入 serve 分支；`PYTHONPATH=src` 即正。
