/**
 * semantic-memory — opencode v2 插件（控制面，薄适配层）
 *
 * 安装：cp memory.ts ~/.config/opencode/plugins/
 * 依赖：零运行时依赖（仅用 fetch/AbortSignal，类型来自 @opencode/plugin）。
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
 * 4. 注入方式 = event.system.push({type:"text",...})：追加到 system 数组
 *    尾部（位于 opencode 自身 system 段之后、messages 之前），不改动已有
 *    前缀；召回集合未变化（uid 集合哈希）时不重复注入，保证注入块内容
 *    逐轮稳定，保护 prefix cache。
 *
 * 类型基线：@opencode/plugin@2.0.28（v2 API，tsc --strict 通过）；
 * 运行时兼容：stable（{id,setup}）+ dev（default.server()）双形态导出，
 * 已在 opencode 0.0.0-dev-202610130450 真实加载器验证。
 */
import { Plugin } from "@opencode/plugin";

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

function hashOf(uids: string[]): string {
  return uids.slice().sort().join(",");
}

interface Hit {
  uid: string;
  kind: string;
  content: string;
  project?: string | null;
}

function formatBlock(hits: Hit[]): string {
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

/** 从 context/compaction/generate 事件中提取最近的用户文本。 */
function recentUserText(event: { messages: Array<{ role: string; content: unknown }> }): string {
  const msgs = event.messages;
  for (let i = msgs.length - 1; i >= 0 && i >= msgs.length - 6; i--) {
    const m = msgs[i];
    if (m.role !== "user") continue;
    const t = partsText(m.content);
    if (t) return t.slice(0, 2000);
  }
  return "";
}

/** 防御式提取文本：兼容字符串 content 与 [{type:"text",text}] 部件数组。 */
function partsText(parts: unknown): string {
  if (typeof parts === "string") return parts;
  if (Array.isArray(parts)) {
    return parts
      .filter((p: any) => p?.type === "text" && typeof p?.text === "string")
      .map((p: any) => p.text as string)
      .join("\n");
  }
  return "";
}

const REMEMBER_RE = /(?:记住|请记住|remember)[:：]?\s*(.{10,500})/i;

/** 显式记忆捕获（两条加载线共用）。返回是否命中。 */
async function captureRemember(text: string, project: string | undefined): Promise<boolean> {
  const m = text.match(REMEMBER_RE);
  if (!m) return false;
  await post("/ingest", {
    content: m[1].trim(),
    kind: "preference",
    source: "chat",
    project,
    tags: ["explicit"],
  });
  return true;
}

/** 工具产物捕获（两条加载线共用）。 */
async function captureTool(name: string, output: string, project: string | undefined): Promise<void> {
  const text = output.slice(0, 800);
  if (!text || text === '""') return;
  await post("/ingest", {
    content: `tool ${name} produced: ${text}`,
    kind: "artifact",
    source: "tool",
    project,
    tags: ["tool-capture", name],
  });
}

// ---------- 插件定义 ----------

type StableCtx = Parameters<Parameters<typeof Plugin.define>[0]["setup"]>[0];

/**
 * 召回 + uid 哈希去重（两条加载线共用核心）。
 * query 为空或召回集合未变时返回 null，否则返回注入块文本。
 */
async function recallBlock(query: string, project: string | undefined): Promise<string | null> {
  if (!(await ensureService())) return null;
  if (!query) return null;
  const res = await post("/recall", { query: query.slice(0, 2000), limit: LIMIT, project });
  const hits: Hit[] = res?.hits ?? [];
  if (!hits.length) return null;
  const hash = hashOf(hits.map((h) => h.uid));
  if (hash === lastInjectHash) return null; // 集合未变，跳过重复注入
  lastInjectHash = hash;
  return formatBlock(hits);
}

/**
 * 控制面主体，按运行时能力自动选线：
 * - stable 线（@opencode/plugin 2.x 类型契约）：ctx.session.hook 可用，
 *   注册 context/compaction/generate/prompt + tool execute.after 五个钩子；
 * - legacy/dev 线（实测 0.0.0-dev-202610130450）：ctx 为 v1 形态
 *   {client, project, directory, worktree, ...}，无 session.hook，
 *   返回 v1 风格钩子映射（chat.message / experimental.session.compacting /
 *   tool.execute.after）。
 */
async function control(ctx: StableCtx): Promise<Record<string, unknown> | void> {
    const c: any = ctx;
    const project: string | undefined =
      c?.location?.project?.directory ?? c?.location?.directory
      ?? c?.project?.directory ?? c?.directory ?? c?.worktree ?? c?.project?.id ?? undefined;
    if (process.env.MEMORY_PLUGIN_DEBUG === "1") {
      console.log("[semantic-memory] ctx keys:", JSON.stringify(Object.keys(c ?? {})),
        "line:", typeof c?.session?.hook === "function" ? "stable" : "legacy",
        "project:", project);
    }

    // ===== stable 线：v2 类型化钩子 =====
    if (typeof c?.session?.hook === "function") {
      /** 召回注入。三个钩子共用同一实现（见文件头约束 3/4）。 */
      async function inject(event: { system: Array<{ type: "text"; text: string }>; messages: Array<{ role: string; content: unknown }> }) {
        const block = await recallBlock(recentUserText(event), project);
        if (block) event.system.push({ type: "text", text: block });
      }

      await ctx.session.hook("context", inject);
      await ctx.session.hook("compaction", inject);
      await ctx.session.hook("generate", inject);

      /** 压缩前落库：把将被压缩的用户意图浓缩存入（v2 compaction 丢弃 ≠ 记忆永存）。 */
      await ctx.session.hook("compaction", async (event) => {
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

      /** 显式记忆信号：用户说"记住…"时自动落库（prompt 钩子在用户提交时触发）。 */
      await ctx.session.hook("prompt", async (event) => {
        if (!(await ensureService())) return;
        await captureRemember(event.prompt?.text ?? "", project);
      });

      /** 工具产物捕获（白名单制，默认关闭）。例：MEMORY_PLUGIN_TOOL_CAPTURE=gis,bash */
      await ctx.tool.hook("execute.after", async (event) => {
        if (event.status !== "completed") return;
        if (!TOOL_CAPTURE.has(event.tool) || !(await ensureService())) return;
        const content = event.result?.content;
        const raw = typeof content === "string"
          ? content
          : JSON.stringify(event.result?.output ?? content ?? "");
        await captureTool(event.tool, raw, project);
      });
      return;
    }

    // ===== legacy/dev 线：v1 风格钩子映射 =====
    return {
      /** 用户消息：显式记忆捕获 + 合成 text part 注入。 */
      "chat.message": async (_input: any, output: any) => {
        if (!(await ensureService())) return;
        const text = partsText(output?.parts ?? output?.message?.content ?? "");
        if (text) await captureRemember(text, project);
        const block = await recallBlock(text, project);
        if (block && Array.isArray(output?.parts)) {
          output.parts.push({ type: "text", text: block, synthetic: true });
        }
      },
      /** 压缩：重置注入去重（dev 线拿不到会话文本，托管落库仅 stable 线）。 */
      "experimental.session.compacting": async () => {
        lastInjectHash = "";
      },
      /** 工具产物捕获：output.output 为字符串结果。 */
      "tool.execute.after": async (input: any, output: any) => {
        const name: string = input?.tool ?? "";
        if (!TOOL_CAPTURE.has(name) || !(await ensureService())) return;
        await captureTool(name, String(output?.output ?? ""), project);
      },
    };
}

/**
 * 双形态导出：
 * - stable 线（@opencode/plugin 2.x 运行时）：读取 { id, setup }；
 * - dev 线（0.0.0-dev-* 运行时）：要求 default export 带 server() 方法，
 *   签名为 server(ctx, options)，但 ctx 实为 v1 形态（control 内自动检测）。
 * 两种运行时各取所需，多余的键对彼此均为惰性。
 */
const plugin = Plugin.define({
  id: "semantic-memory",
  setup: async (ctx) => { await control(ctx); },
});

export default { ...plugin, server: (ctx: StableCtx) => control(ctx) };
