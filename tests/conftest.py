"""测试封闭性与共享 HTTP 测试面。

1. 强制重排关闭：config 默认在门禁 PASSED 后指向常规模型路径（config.py 注释）；
   本机若存在该文件，单测会意外加载真模型（慢 + GPU 依赖）。全套件强制
   MEMORY_RERANK_MODEL=空 = 关闭；需要真件的测试自行覆盖（真件路径按
   opt-in 惯例走环境显式指定）。
2. 数据目录隔离（防开发机真 config/数据泄入，也防迁移逻辑碰真旧目录）：
   MEMORY_DATA_DIR 与 XDG_DATA_HOME 钉到临时目录；测试 tier 语义时自行
   monkeypatch 覆盖。
3. 进程内 REST 测试服：ThreadingHTTPServer(port 0) + http.client 薄封装。
   注意：server 模块单例（cfg/store）由首个导入它的测试模块按其 env 建库，
   全测试共享同一单例（与历史行为一致）；故 fixture 为 session
   级且懒导入 server。
"""
import http.client
import json
import os
import tempfile
import threading
from urllib.parse import urlencode

import pytest

os.environ.setdefault("MEMORY_RERANK_MODEL", "")  # 未设→关闭保封闭性（防真模型意外加载）；显式指定→保留 opt-in
_ISOLATION = tempfile.mkdtemp(prefix="mcp-test-isolation-")  # 会话级隔离根（tmp 自身会回收）
os.environ["MEMORY_DATA_DIR"] = os.path.join(_ISOLATION, "data")   # 新数据目录（含 config.toml 落点）
os.environ["XDG_DATA_HOME"] = os.path.join(_ISOLATION, "xdg")      # 旧默认目录同链隔离（迁移测试安全）

class RestResult:
    """http.client 响应薄封装：对齐旧 TestClient 习惯（status_code/json/text/headers）。"""

    def __init__(self, status: int, body: bytes, headers: dict):
        self.status_code = status
        self.headers = headers
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8")

    def json(self):
        return json.loads(self._body)


class RestClient:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port

    def request(self, method: str, path: str, *, json_body=None, data: bytes | None = None,
                headers: dict | None = None, params: dict | None = None) -> RestResult:
        if params:
            path = f"{path}?{urlencode(params)}"
        hdrs = dict(headers or {})
        payload = b""
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif data is not None:
            payload = data
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.request(method, path, body=payload or None, headers=hdrs)
            resp = conn.getresponse()
            return RestResult(resp.status, resp.read(), dict(resp.headers))
        finally:
            conn.close()

    def get(self, path: str, **kw) -> RestResult:
        return self.request("GET", path, **kw)

    def post(self, path: str, json=None, **kw) -> RestResult:
        return self.request("POST", path, json_body=json, **kw)

    def delete(self, path: str, **kw) -> RestResult:
        return self.request("DELETE", path, **kw)


def raw_request(port: int, raw: bytes) -> bytes:
    """裸 socket 请求（构造畸形头用）：读至对端关闭后整体返回。"""
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(raw)
        chunks = []
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)


@pytest.fixture(scope="session")
def client():
    from http.server import ThreadingHTTPServer

    from semantic_memory import server as _srv

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _srv._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="test-rest").start()
    yield RestClient("127.0.0.1", httpd.server_address[1])
    httpd.shutdown()
    httpd.server_close()
