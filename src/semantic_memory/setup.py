"""semantic-memory setup 子命令（任务 8）：探测→校验下载→写 config→冒烟→报告。

一键满配安装（D6）：--profile full|embed|bm25（默认 full）。禁起常驻服务、
禁动 ~/.config/opencode。HTTP 客户端复用 cloud.py 的 httpx2 惯例（httpx
同构 API；两者皆缺且需下载 → 自解释报错，bm25 档零下载不受影响）。

下载三硬性：.part 断点续传（Range 续接，服务器不支持则整体重下）、固定
sha256 常量表校验（不符删 .part 重下一次，再败 exit 1）、落位原子改名。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import write_config
from .paths import resolve_data_dir

# ---- 固定 sha256 常量表 ----
# 蒸馏/重排：本机 ~/.local/share/memory-mcp/models/ 真件 sha256sum（与 HF
#   仓库元数据 lfs.sha256 逐字节一致，证据见 .omo/evidence/task-8-*.md）。
# 嵌入：本机无真件，取 HF API 元数据 lfs.sha256（= git-lfs pointer oid）。
# URL 三源：download_models.sh（蒸馏/重排）+ download_model.sh（嵌入）。
EMBED_URL = ("https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/"
             "Qwen3-Embedding-0.6B-Q8_0.gguf")
DISTILL_URL = ("https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/"
               "Qwen3-1.7B-Q8_0.gguf")
RERANK_URL = ("https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/"
              "resolve/main/qwen3-reranker-0.6b-q8_0.gguf")


@dataclass(frozen=True)
class ModelSpec:
    key: str        # embed | distill | rerank
    url: str
    filename: str
    sha256: str
    size: int       # 字节（磁盘余量门槛与进度显示用）


MODELS: dict[str, ModelSpec] = {
    "embed": ModelSpec(
        "embed", EMBED_URL, "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
        639_150_592),
    "distill": ModelSpec(
        "distill", DISTILL_URL, "Qwen3-1.7B-Q8_0.gguf",
        "0becaa825564295d82e9af4d008bca5f8b7f5f73bf1c6a0b58f7c53ef26b47fd",
        1_834_426_944),
    "rerank": ModelSpec(
        "rerank", RERANK_URL, "qwen3-reranker-0.6b-q8_0.gguf",
        "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48",
        639_153_184),
}
_PROFILE_MODELS = {"full": ("embed", "distill", "rerank"), "embed": ("embed",), "bm25": ()}
MIN_FREE_BYTES = 3 * 1024**3   # 目标盘余量硬门槛（计划任务 8）
_CHUNK = 1 << 20


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_client():
    """cloud.py 同款惯例：httpx2（httpx 同构）→ http → None（自解释报错）。"""
    try:
        import httpx2
    except ImportError:
        try:
            import httpx as httpx2  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        return httpx2.Client(follow_redirects=True, timeout=120.0)
    except Exception as exc:  # 坏代理 env 等环境问题：带原因降级，不裸 traceback
        print(f"setup: HTTP 客户端构造失败（{type(exc).__name__}: {exc}）", file=sys.stderr)
        return None


def _download_once(spec: ModelSpec, part: Path, client) -> None:
    """流式下载到 .part；已有 .part 时 Range 续接（206），否则整体写。"""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", spec.url, headers=headers) as resp:
        if resp.status_code == 206 and offset:
            mode = "ab"
        elif resp.status_code == 200:
            mode, offset = "wb", 0   # 服务器不理会 Range → 整体重下
        else:
            raise RuntimeError(f"HTTP {resp.status_code}")
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open(mode) as f:  # "wb" 自带截断（不支持续传的整体重下）
            for chunk in resp.iter_bytes(_CHUNK):
                f.write(chunk)


def _download_verified(spec: ModelSpec, target: Path, client) -> str | None:
    """下载+校验+原子落位；失败删 .part 重下一次，再败返回错误消息。"""
    part = target.with_name(target.name + ".part")
    last = ""
    for _attempt in (1, 2):
        try:
            _download_once(spec, part, client)
            if _sha256_file(part) == spec.sha256:
                part.replace(target)
                return None
            last = f"sha256 校验不符（期望 {spec.sha256[:12]}…）"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        part.unlink(missing_ok=True)
    return f"{spec.filename} 下载失败（已重试一次）: {last}"


def _has_nvidia_smi() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    return subprocess.run([exe], capture_output=True).returncode == 0


# ---- 冒烟（bm25 档全跳过；函数粒度可 mock，真件路径见任务证据）----

def _smoke_embed(path: Path, device: str) -> str | None:
    from .embedder import Embedder
    e = Embedder(path, device=device)
    if not e.available:
        return f"嵌入模型不可用: {path}（需要 numpy+llama-cpp-python: pip install semantic-memory[embed]）"
    vec = e.embed_query("memory")
    if vec is None or tuple(vec.shape) != (e.dim,):
        return f"嵌入冒烟失败: {path}"
    return None


def _smoke_distill(path: Path, device: str) -> str | None:
    from .distiller import Distiller, _parse
    d = Distiller(path, device=device)
    if not d.available:
        return f"蒸馏模型不可用: {path}"
    if _parse("{这是坏 JSON，必须 fail-open") is not None:
        return "蒸馏 fail-open 契约破坏：坏 JSON 应降级 None"
    return None


def _smoke_rerank(path: Path, device: str) -> str | None:
    from .modelslot import ModelSlot
    from .reranker import Reranker
    r = Reranker(path, device=device, budget_ms=120_000, slot=ModelSlot())
    if not r.available:
        return f"重排模型不可用: {path}"
    ranked = r.rerank("how to deploy the service", [
        ("rel", "deploy the service with semantic-memory setup, then start the server"),
        ("off1", "the weather is nice today, good for a walk"),
        ("off2", "a recipe for banana bread with walnuts"),
    ])
    if ranked is None:
        return f"重排冒烟失败（fail-open 降级）: {path}"
    if ranked[0][0] != "rel":
        return f"重排判别序异常：相关候选未居首（首位 {ranked[0][0]!r}）"
    return None


def _run_smoke(profile: str, models_dir: Path, device: str) -> list[str]:
    if profile == "bm25":
        return []
    errors = [e for e in (
        _smoke_embed(models_dir / MODELS["embed"].filename, device),
        _smoke_distill(models_dir / MODELS["distill"].filename, device) if profile == "full" else None,
        _smoke_rerank(models_dir / MODELS["rerank"].filename, device) if profile == "full" else None,
    ) if e]
    return errors


def run_setup(argv: list[str] | None = None, *, client=None, out=sys.stdout) -> int:
    """setup 主流程；client 注入点供 mock transport 测试。返回进程退出码。"""
    ap = argparse.ArgumentParser(
        prog="semantic-memory setup",
        description="探测→下载（sha256 校验+断点续传）→写 config.toml→冒烟→报告",
    )
    ap.add_argument("--profile", choices=("full", "embed", "bm25"), default="full",
                    help="full=嵌入+蒸馏+重排（默认）；embed=仅嵌入；bm25=零下载纯检索")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="auto=按 nvidia-smi 探测（默认）")
    ap.add_argument("--data-dir", help="数据目录（默认按 MEMORY_DATA_DIR/XDG/~/semantic-memory-mcp 解析）")
    ap.add_argument("--json", action="store_true", help="机器可读报告（agent 通道）")
    ap.add_argument("--yes", action="store_true", help="免确认（非交互 stdin 自动继续）")
    ns = ap.parse_args(argv)

    data_dir = Path(ns.data_dir).expanduser() if ns.data_dir else resolve_data_dir()
    models_dir = data_dir / "models"
    device = ns.device
    if device == "auto":
        device = "cuda" if _has_nvidia_smi() else "cpu"

    # ---- ① 探测：已有文件 sha256 复核→跳过；不符视同缺失重下 ----
    specs = [MODELS[k] for k in _PROFILE_MODELS[ns.profile]]
    missing, skipped, corrupt = [], [], []
    for spec in specs:
        target = models_dir / spec.filename
        if target.is_file() and _sha256_file(target) == spec.sha256:
            skipped.append(spec.key)
        else:
            if target.exists():
                corrupt.append(spec.key)
            missing.append(spec)
    data_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        free = shutil.disk_usage(data_dir).free
        if free < MIN_FREE_BYTES:
            print(f"setup: 目标盘余量不足（{free / 1024**3:.1f}G < 3G）: {data_dir}",
                  file=sys.stderr)
            return 1
        if not ns.yes and sys.stdin.isatty():
            total = sum(s.size for s in missing) / 1024**3
            answer = input(f"将下载 {len(missing)} 个模型（约 {total:.1f}G）到 {models_dir}，继续? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("setup: 已取消", file=sys.stderr)
                return 1
        if client is None:
            client = _default_client()
            if client is None:
                print("setup: 无可用 HTTP 客户端（httpx2/http 均未安装），无法下载。"
                      "bm25 档零下载不受影响，或安装任一客户端后重试", file=sys.stderr)
                return 1

    # ---- ② 下载缺失模型（.part 续传 + sha256 校验 + 一次重试）----
    installed = []
    for spec in missing:
        if ns.yes or ns.json or not sys.stdin.isatty():
            print(f"下载 {spec.filename}（{spec.size / 1024**3:.2f}G）…", file=sys.stderr)
        err = _download_verified(spec, models_dir / spec.filename, client)
        if err:
            print(f"setup: {err}", file=sys.stderr)
            return 1
        installed.append(spec.key)

    # ---- ③ 写 config.toml（B2 write_config；rerank 走默认路径不写）----
    values: dict = {"device": device}
    if ns.profile in ("full", "embed"):
        values["model_path"] = str(models_dir / MODELS["embed"].filename)
    if ns.profile == "full":
        values["distill_model"] = str(models_dir / MODELS["distill"].filename)
    if ns.profile == "bm25":
        values["allow_bm25_only"] = True
    config_path = write_config(data_dir, values)

    # ---- ④ 冒烟（bm25 全跳过；真件）----
    for msg in _run_smoke(ns.profile, models_dir, device):
        print(f"setup: 冒烟失败——{msg}", file=sys.stderr)
        return 1

    # ---- ⑤ 报告 ----
    if ns.json:
        print(json.dumps({
            "installed": installed, "skipped": skipped,
            "config_path": str(config_path), "data_dir": str(data_dir), "device": device,
        }, ensure_ascii=False), file=out)
        return 0
    for key in corrupt:
        print(f"注意：已存在的 {key} 模型 sha256 不符，已重新下载", file=sys.stderr)
    print(f"语义记忆安装完成（profile={ns.profile} device={device}）")
    for k in installed:
        print(f"  已下载 {k:8s} -> {models_dir / MODELS[k].filename}")
    for k in skipped:
        print(f"  已存在 {k:8s} -> {models_dir / MODELS[k].filename}（sha256 复核通过）")
    print(f"  配置 {config_path}")
    print("下一步：semantic-memory 起服；opencode 用户把 integrations/opencode/memory.ts 装为插件")
    return 0
