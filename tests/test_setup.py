"""setup 子命令测试（任务 8）：mock httpx2 传输，下载/续传/校验/跳过/三档 config/JSON schema。"""
import hashlib
import io
import json
import shutil as _shutil
import tomllib
from types import SimpleNamespace

import pytest

httpx2 = pytest.importorskip("httpx2")  # setup 下载通道测试需 HTTP 客户端（mock transport）

from semantic_memory import setup as setup_mod
from semantic_memory.setup import ModelSpec, _download_verified, run_setup


def fake_spec(key: str, size: int = 8192) -> tuple[bytes, ModelSpec]:
    blob = (bytes(range(256)) * ((size // 256) + 1))[:size]
    return blob, ModelSpec(key, f"http://test/{key}.gguf", f"{key}.gguf",
                           hashlib.sha256(blob).hexdigest(), len(blob))


class FileServer:
    """按预设字节服务 GET；带 Range 头 → 206 切片（续传语义）。可注入坏内容。"""

    def __init__(self, blob: bytes, bad_for_calls: int = 0):
        self.blob, self.bad_for_calls, self.calls = blob, bad_for_calls, 0
        self.range_headers: list[str | None] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        rng = request.headers.get("range")
        self.range_headers.append(rng)
        blob = b"corrupted!" if self.calls <= self.bad_for_calls else self.blob
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        content = blob[start:]
        if rng:
            return httpx2.Response(206, content=content, headers={
                "Content-Range": f"bytes {start}-{len(blob) - 1}/{len(blob)}"})
        return httpx2.Response(200, content=content)


def mock_client(handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


@pytest.fixture()
def no_smoke(monkeypatch):
    """冒烟全绿桩（真件冒烟归真机证据/F3，测试面只验证接线与失败路径）。"""
    for name in ("_smoke_embed", "_smoke_distill", "_smoke_rerank"):
        monkeypatch.setattr(setup_mod, name, lambda *a: None)


@pytest.fixture()
def tiny_models(monkeypatch):
    """三模型换 8KB 假件（sha256 对得上 mock 内容）。"""
    specs = {}
    for key in ("embed", "distill", "rerank"):
        _, spec = fake_spec(key)
        monkeypatch.setitem(setup_mod.MODELS, key, spec)
        specs[key] = spec
    return specs


# ---------------- _download_verified：下载/续传/校验/重试 ----------------

def test_download_verified_ok(tmp_path):
    blob, spec = fake_spec("embed")
    srv = FileServer(blob)
    target = tmp_path / spec.filename
    assert _download_verified(spec, target, mock_client(srv)) is None
    assert target.read_bytes() == blob
    assert not target.with_name(spec.filename + ".part").exists()  # .part 已清/已改名


def test_download_resume_from_part(tmp_path):
    blob, spec = fake_spec("embed")
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(blob[:4096])                       # 半截 .part → Range 续接
    srv = FileServer(blob)
    assert _download_verified(spec, tmp_path / spec.filename, mock_client(srv)) is None
    assert (tmp_path / spec.filename).read_bytes() == blob
    assert srv.range_headers == ["bytes=4096-"]         # 续传而非整体重下
    assert srv.calls == 1


def test_download_sha_mismatch_delete_and_retry_then_fail(tmp_path, capsys):
    _, spec = fake_spec("embed")
    srv = FileServer(b"always wrong bytes", bad_for_calls=99)
    err = _download_verified(spec, tmp_path / spec.filename, mock_client(srv))
    assert err and "下载失败" in err and "sha256" in err
    assert srv.calls == 2                               # 删 .part 后重下一次
    assert not list(tmp_path.iterdir())                 # .part 已删、target 未落位


def test_download_retry_recovers(tmp_path):
    blob, spec = fake_spec("embed")
    srv = FileServer(blob, bad_for_calls=1)             # 首次坏内容，重试正确
    assert _download_verified(spec, tmp_path / spec.filename, mock_client(srv)) is None
    assert (tmp_path / spec.filename).read_bytes() == blob
    assert srv.calls == 2


def test_download_http_error_reported(tmp_path):
    _, spec = fake_spec("embed")

    def broken(request):
        return httpx2.Response(500)

    err = _download_verified(spec, tmp_path / spec.filename, mock_client(broken))
    assert err and "下载失败" in err and "500" in err


# ---------------- run_setup：三档 profile / config / 报告 ----------------

def setup_json(tmp_path, extra=(), *, specs=None, client=None):
    argv = ["--data-dir", str(tmp_path), "--json", "--yes", *extra]
    out = io.StringIO()
    code = run_setup(argv, client=client, out=out)
    return code, (json.loads(out.getvalue()) if out.getvalue().strip() else None)


def client_for(specs: dict, **kw) -> httpx2.Client:
    blobs = {s.url: b for b, s in ((fake_spec(k, s.size)[0], s) for k, s in specs.items())}
    holder = SimpleNamespace(blobs=blobs, calls=[], bad_for=kw.get("bad_for", 0))

    def handler(request: httpx2.Request) -> httpx2.Response:
        holder.calls.append(str(request.url))
        blob = blobs[str(request.url)]
        if len(holder.calls) <= holder.bad_for:
            blob = b"bad"
        rng = request.headers.get("range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        return httpx2.Response(206 if rng else 200, content=blob[start:])

    return mock_client(handler), holder


def test_run_setup_full_config_and_json(tmp_path, tiny_models, no_smoke):
    client, holder = client_for(tiny_models)
    code, report = setup_json(tmp_path, specs=tiny_models, client=client)
    assert code == 0
    assert set(report) == {"installed", "skipped", "config_path", "data_dir", "device"}
    assert report["installed"] == ["embed", "distill", "rerank"]
    assert report["skipped"] == []
    cfg = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert cfg["model_path"] == str(tmp_path / "models" / tiny_models["embed"].filename)
    assert cfg["distill_model"] == str(tmp_path / "models" / tiny_models["distill"].filename)
    assert cfg["device"] == report["device"]
    assert "rerank_model" not in cfg                       # rerank 走默认路径不写


def test_run_setup_embed_profile(tmp_path, tiny_models, no_smoke):
    client, _ = client_for(tiny_models)
    code, report = setup_json(tmp_path, ["--profile", "embed"], client=client)
    assert code == 0 and report["installed"] == ["embed"]
    cfg = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert "model_path" in cfg and "distill_model" not in cfg


def test_run_setup_bm25_no_download_no_smoke(tmp_path, monkeypatch):
    def boom(*a):  # bm25 档不得触碰网络与冒烟
        raise AssertionError("bm25 profile must not download/smoke")
    monkeypatch.setattr(setup_mod, "_default_client", boom)
    for name in ("_smoke_embed", "_smoke_distill", "_smoke_rerank"):
        monkeypatch.setattr(setup_mod, name, boom)
    out = io.StringIO()
    code = run_setup(["--profile", "bm25", "--data-dir", str(tmp_path), "--json", "--yes"], out=out)
    assert code == 0
    assert json.loads(out.getvalue())["installed"] == []
    cfg = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert cfg["allow_bm25_only"] is True
    assert "model_path" not in cfg


def test_run_setup_skips_existing_verified(tmp_path, tiny_models, no_smoke):
    blob, spec = fake_spec("embed", tiny_models["embed"].size)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / spec.filename).write_bytes(blob)   # 已在且 sha256 符 → 跳过
    client, holder = client_for(tiny_models)
    code, report = setup_json(tmp_path, ["--profile", "embed"], client=client)
    assert code == 0 and report["skipped"] == ["embed"] and report["installed"] == []
    assert all("embed" not in u for u in holder.calls)        # 未对该模型发请求


def test_run_setup_corrupt_existing_redownloads(tmp_path, tiny_models, no_smoke):
    spec = tiny_models["embed"]
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / spec.filename).write_bytes(b"stale corrupt content")
    client, _ = client_for(tiny_models)
    code, report = setup_json(tmp_path, ["--profile", "embed"], client=client)
    assert code == 0 and report["installed"] == ["embed"]
    assert (tmp_path / "models" / spec.filename).read_bytes()  # 重下后内容即假件字节


def test_run_setup_disk_low_refuses(tmp_path, monkeypatch, tiny_models, no_smoke):
    monkeypatch.setattr(_shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=1024**3, total=10, used=1))
    client, _ = client_for(tiny_models)
    code, report = setup_json(tmp_path, client=client)
    assert code == 1 and report is None                      # 探测失败：无 JSON 报告
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").iterdir())


def test_run_setup_no_http_client(tmp_path, monkeypatch, tiny_models, no_smoke):
    monkeypatch.setattr(setup_mod, "_default_client", lambda: None)
    code, _ = setup_json(tmp_path, client=None)              # 强制走默认客户端路径
    assert code == 1


def test_run_setup_device_auto_probe(tmp_path, monkeypatch, tiny_models, no_smoke):
    client, _ = client_for(tiny_models)
    monkeypatch.setattr(setup_mod, "_has_nvidia_smi", lambda: True)
    code, report = setup_json(tmp_path, ["--profile", "bm25"])
    assert code == 0 and report["device"] == "cuda"
    monkeypatch.setattr(setup_mod, "_has_nvidia_smi", lambda: False)
    code, report = setup_json(tmp_path, ["--profile", "bm25"])
    assert code == 0 and report["device"] == "cpu"


def test_run_setup_smoke_failure_exits(tmp_path, tiny_models, monkeypatch, capsys):
    monkeypatch.setattr(setup_mod, "_smoke_embed",
                        lambda path, device: f"嵌入冒烟失败: {path}")
    client, _ = client_for(tiny_models)
    code, _ = setup_json(tmp_path, client=client)
    assert code == 1
    assert "冒烟失败" in capsys.readouterr().err
