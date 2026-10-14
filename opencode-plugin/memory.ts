/**
 * semantic-memory — opencode v2 插件（控制面，薄适配层）
 *
 * 安装：cp memory.ts ~/.config/opencode/plugins/
 * 依赖：零运行时依赖（仅用 fetch/setTimeout）。
 *
 * 配套服务：memory-mcp（Python FastMCP），默认 http://127.0.0.1:8765
 * 配置（环境变量）：
 *   MEMORY_PLUGIN_URL           服务地址，默认 http://127.0.0.1:8765
 *   MEMORY_PLUGIN_TOKEN         与服务端 MEMORY_TOKEN 对应（可空）
 *   MEMORY_PLUGIN_LIMIT         每次召回条数，默认 5
 *   MEMORY_PLUGIN_BUDGET_CHARS  注入块字符预算，默认 2000（≈500 tokens）
 *   MEMORY_PLUGIN_TOOL_CAPTURE  tool execute.after 捕获白名单（逗号分隔，默认空=关闭）
 *   MEMORY_PLUGIN_AUTOSTART     =1 时服务未启动则尝试 spawn `memory-mcp`
 *
 * 设计约束（勿破坏）：
 * 1. 插件零智能——检索/嵌入/存储全在服务端，这里只做转发与降级；
 * 2. 服务不可达时静默跳过，绝不阻塞会话（fail-open）；
 * 3. 注入必须同时注册 context/compaction/generate 三个钩子，
 *    否则压缩请求丢失注入并打穿 prefix cache（superpowers#2391 教训）；
 * 4. 召回集合未变化时不重复注入（uid 集合哈希判断），注入点放在
 *    消息序列尾部（合成消息），不污染 system 前缀，保护 prefix cache。
 */
import { Plugin } from "@opencode/plugin"; // 仅类型导入，v2 运行时零开销

const URL_BASE = (process.env.MEMORY_PLUGIN_URL ?? "http://127.0.0.1:8765").replace(/\/$/, "");
const TOKEN = process.env.MEMORY_PLUGIN_TOKEN ?? "";
const LIMIT = Number(process.env.MEMORY_PLUGIN_LIMIT ?? 5);
const BUDGET = Number(process.env.MEMORY_PLUGIN_BUDGET_CHARS ?? 2000);
const TOOL_CAPTURE = new Set(
  (process.env.MEMORY_PLUGIN_TOOL_CAPTURE ?? "").split(",").map((s) => s.trim()).filter(Boolean),
);
const AUTOSTART = process.env.MEMORY_PLUGIN_AUTOSTART === "1";

const TIMEOUT_MS = 1500;

// ---------- 服务通信（全部 fail-open） ----------

async function post(path: string, body: unknown): Promise<any | null> {
  try {
    const res = await fetch(`${URL_BASE}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null; // 服务未启动 / 超时 / 网络异常：静默降级
  }
}

let serviceChecked = false;
async function ensureService(): Promise<boolean> {
  if (serviceChecked) return true;
  try {
    const res = await fetch(`${URL_BASE}/health`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
    serviceChecked = res.ok;
  } catch {
    serviceChecked = false;
  }
  if (!serviceChecked && AUTOSTART) {
    try {
      const { spawn } = await import("node:child_process");
      spawn("memory-mcp", [], { detached: true, stdio: "ignore" }).unref();
      // 给模型/DB 一点启动时间，下一轮钩子自然会连上
    } catch { /* 忽略 */ }
  }
  return serviceChecked;
}

// ---------- 注入状态（缓存感知） ----------

let lastInjectHash = "";
let lastInjectBlock = "";

function hashOf(uids: string[]): string {
  return uids.slice().sort().join(",");
}

function formatBlock(hits: any[]): string {
  let out = "<semantic-memory>\n以下是跨会话记忆库中与当前任务相关的条目（供参考，勿复述给用户）：\n";
  let used = out.length;
  for (const h of hits) {
    const line = `- [${h.kind}] ${h.content}${h.project ? ` (project: ${h.project})` : ""}\n`;
    if (used + line.length > BUDGET) break;
    out += line;
    used += line.length;
  }
  return out + "</semantic-memory>";
}

/** 从钩子事件中提取最近的用户文本（防御式取值，适配 v2 事件载荷演进） */
function recentUserText(event: any): string {
  const msgs = event?.messages ?? event?.session?.messages ?? [];
  for (let i = msgs.length - 1; i >= 0 && i >= msgs.length - 6; i--) {
    const m = msgs[i];
    if (m?.role !== "user") continue;
    const parts = m?.parts ?? m?.content;
    if (typeof parts === "string") return parts.slice(0, 2000);
    if (Array.isArray(parts)) {
      const t = parts.filter((p: any) => p?.type === "text").map((p: any) => p.text).join("\n");
      if (t) return t.slice(0, 2000);
    }
  }
  return "";
}

function projectOf(ctx: any): string | undefined {
  return ctx?.project?.id ?? ctx?.project?.name ?? ctx?.directory ?? undefined;
}

// ---------- 插件定义 ----------

export default Plugin.define({
  id: "semantic-memory",

  setup(ctx) {
    const project = projectOf(ctx);

    /** 召回 + 缓存感知注入。三个钩子共用同一实现（见文件头约束 3）。 */
    async function inject(event: any) {
      if (!(await ensureService())) return;
      const query = recentUserText(event);
      if (!query) return;
      const res = await post("/recall", { query, limit: LIMIT, project });
      const hits: any[] = res?.hits ?? [];
      if (!hits.length) return;
      const hash = hashOf(hits.map((h) => h.uid));
      if (hash === lastInjectHash) return; // 集合未变，跳过重复注入
      lastInjectHash = hash;
      lastInjectBlock = formatBlock(hits);
      event?.inject?.(lastInjectBlock) ?? event?.context?.push?.({
        role: "user",
        parts: [{ type: "text", text: lastInjectBlock }],
      });
    }

    ctx.session.hook("context", inject);
    ctx.session.hook("compaction", inject);
    ctx.session.hook("generate", inject);

    /** 压缩前落库：把将被压缩的会话内容浓缩存入（v2 compaction 丢弃 ≠ 记忆永存）。 */
    ctx.session.hook("compaction", async (event: any) => {
      if (!(await ensureService())) return;
      const text = recentUserText(event);
      if (text.length < 40) return;
      await post("/ingest", {
        content: text.slice(0, 1200),
        kind: "fact",
        source: "compaction",
        project,
        tags: ["auto-compaction"],
      });
      lastInjectHash = ""; // 压缩后强制下轮重新召回
    });

    /** 显式记忆信号：用户说"记住…"时自动落库（选择性，不做全量落库）。 */
    ctx.session.hook("chat.message", async (event: any) => {
      if (!(await ensureService())) return;
      const text = recentUserText(event);
      const m = text.match(/(?:记住|请记住|remember)[:：]?\s*(.{10,500})/i);
      if (!m) return;
      await post("/ingest", {
        content: m[1].trim(),
        kind: "preference",
        source: "chat",
        project,
        tags: ["explicit"],
      });
    });

    /** 工具产物捕获（白名单制，默认关闭）。例：MEMORY_PLUGIN_TOOL_CAPTURE=gis,bash */
    ctx.tool.hook("execute.after", async (event: any) => {
      const name: string = event?.tool ?? event?.name ?? "";
      if (!TOOL_CAPTURE.has(name) || !(await ensureService())) return;
      const output = String(event?.result ?? event?.output ?? "").slice(0, 800);
      if (!output) return;
      await post("/ingest", {
        content: `tool ${name} produced: ${output}`,
        kind: "artifact",
        source: "tool",
        project,
        tags: ["tool-capture", name],
      });
    });
  },
});
