#!/usr/bin/env bash
# 集成测试自持服务生命周期：8766＋tmp DB 拉起→test-hooks→teardown；生产 8765 不受扰。
# 此前 npm test 假设「运行脚本已启动服务」但脚本链里无此步→轮询 30s 必超时（2026-10-26 修复）。
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)
PORT=${MEMORY_TEST_PORT:-8766}
exec 9>"/tmp/sm-itest-$PORT.lock"          # flock 全生命周期互斥：封 curl 探活→bind 的 TOCTOU 窗（B1）
flock -n 9 || { echo "另一份测试实例持有锁（端口 $PORT）" >&2; exit 1; }
if curl -sm1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "端口 $PORT 已被占用（遗留测试服务？），拒绝叠跑" >&2; exit 1
fi
DB=$(mktemp /tmp/sm-itest-XXXXXX.db)
LOG=$(mktemp /tmp/sm-itest-XXXXXX.log)
# MEMORY_DISTILL_MODEL 空串=显式禁用蒸馏（config.py opt 空串语义）→ flush 原文直存：
# test-hooks 断言的是 raw store 形态（全量 transcript 原样落库）；不置空则拾取生产
# config.toml 的蒸馏模型，typed 蒸馏改写内容＋project 归属漂移，断言必挂
export MEMORY_DISTILL_MODEL=
export MEMORY_PLUGIN_TOOL_CAPTURE=bash   # execute.after 白名单默认关闭，test §5 依赖 bash 捕获
MEMORY_PORT=$PORT MEMORY_DB_PATH="$DB" "$ROOT/venv/bin/semantic-memory" serve >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; rm -f "$DB" "$DB"-wal "$DB"-shm "$LOG"' EXIT
up=0
for _ in $(seq 1 60); do
  curl -sm1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 0.5
done
[ "$up" = 1 ] || { echo "测试服务未就绪，日志尾部：" >&2; tail -5 "$LOG" >&2; exit 1; }
MEMORY_PLUGIN_URL="http://127.0.0.1:$PORT" MEMORY_DB_PATH="$DB" node test-hooks.mjs
