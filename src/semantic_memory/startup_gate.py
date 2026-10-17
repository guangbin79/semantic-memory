"""分层严格启动契约（任务 9 / R2）。

"已配置" = env 或 config.toml 任一（Config.from_env 已合并两档）。判定序：
① 非回环绑定且未配 token → 拒启 exit 2（M-1：README 契约 enforced——裸暴露读写/遗忘面；
   不受 allow_bm25_only 豁免：那只管模型档，不豁免网络面）；
② allow_bm25_only=true → 恒放行；
③ 嵌入已配置但文件缺失 → 拒启 exit 2（stderr 指明路径 + setup 修复提示）；
④ 蒸馏已配置但缺失 → 同拒；
⑤ 全未配置 → 招募模式放行 + INFO install_hint（BM25-only 合法）。

拒启必须先于端口绑定（serve 前调用）；禁热路径校验——只查文件在不在，
不加载模型（加载惰性且昂贵）。rerank 不入门：默认路径缺失=优雅关闭是
既有语义。check_startup 为纯函数，不依赖起服，可独立单测。
"""
from __future__ import annotations

import ipaddress
import logging
import sys

from .config import Config

log = logging.getLogger(__name__)

RECRUIT_HINT = (
    "招募模式：未配置任何模型，纯 BM25 运行（合法）。"
    "一键满配：semantic-memory setup"
)


def _is_loopback_bind(host: str) -> bool:
    """绑定地址是否回环：localhost / 127.0.0.0/8 / ::1；其余（含 0.0.0.0、
    hostname、LAN IP）按非回环处理——与 README "required when non-loopback" 同语义。
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # 非 IP 字面量（hostname）按非回环从严


def check_startup(cfg: Config, *, err=sys.stderr) -> int:
    """启动门：返回 0=放行 / 2=拒启（exit 2 语义，消息已写入 err）。"""
    if cfg.token is None and not _is_loopback_bind(cfg.host):
        print(
            f"semantic-memory: 非回环绑定 {cfg.host} 且未设置 MEMORY_TOKEN——"
            f"REST 面将无鉴权暴露（读/写/单条遗忘）\n"
            f"修复：设置 MEMORY_TOKEN（或 config.toml token=），或改绑回环地址",
            file=err,
        )
        return 2
    if cfg.allow_bm25_only:
        return 0
    if cfg.model_path is not None and not cfg.model_path.is_file():
        print(
            f"semantic-memory: 配置的嵌入模型不存在: {cfg.model_path}\n"
            f"修复：semantic-memory setup --profile embed（或修正 MEMORY_MODEL_PATH/config.toml）",
            file=err,
        )
        return 2
    if cfg.distill_model is not None and not cfg.distill_model.is_file():
        print(
            f"semantic-memory: 配置的蒸馏模型不存在: {cfg.distill_model}\n"
            f"修复：semantic-memory setup --profile full（或修正 MEMORY_DISTILL_MODEL/config.toml）",
            file=err,
        )
        return 2
    if cfg.model_path is None and cfg.distill_model is None:
        log.info(RECRUIT_HINT)
    return 0
