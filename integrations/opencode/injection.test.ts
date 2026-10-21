// 注入/会话过滤逻辑回归测试（bun test）：覆盖矩阵的缺口面——
// 召回去重（prefix cache 保护）、服务离线 fail-open、会话隔离、预算封顶、
// 冻结头块语义、文本提取防御、转录防回流、最近用户文本定位。
// 跑法：cd integrations/opencode && bun test injection.test.ts
import { beforeEach, describe, expect, test } from "bun:test"
import {
  buildHeadBlock, formatBlock, firstUserText, hashOf, partsText, recallBlock,
  recentUserText, stateOf, transcriptText,
} from "./memory.ts"

// ---------- 环境隔离：bun 测试内 fetch 全部走桩 ----------
const originalFetch = globalThis.fetch

function stubFetch(handler: (url: string, body: any) => any) {
  globalThis.fetch = (async (url: any, init: any) => {
    const body = init?.body ? JSON.parse(init.body) : {}
    const data = handler(String(url), body)
    if (data === null) throw new Error("offline")
    return new Response(JSON.stringify(data), { status: 200 })
  }) as any
}

beforeEach(() => { globalThis.fetch = originalFetch })

// ---------- 会话隔离与去重 ----------

describe("recallBlock：召回去重与会话隔离（prefix cache 保护）", () => {
  test("同会话同 uid 集合 → 第二次返回 null（不重注）", async () => {
    const st = stateOf("g1-dedup")
    stubFetch(() => ({ hits: [
      { uid: "u1", content: "偏好 uv", kind: "preference", project: "p" },
      { uid: "u2", content: "团队中文", kind: "decision", project: "p" },
    ] }))
    const first = await recallBlock("查询", "p", st)
    expect(first).not.toBeNull()
    expect(first!.block).toContain("偏好 uv")
    const second = await recallBlock("查询", "p", st)
    expect(second).toBeNull()
  })

  test("集合变化 → 重新注入", async () => {
    const st = stateOf("g1-change")
    let uids = ["u1", "u2"]
    stubFetch(() => ({ hits: uids.map((u) => ({ uid: u, content: `c${u}`, kind: "fact", project: "p" })) }))
    await recallBlock("q", "p", st)
    uids = ["u1", "u3"]
    const second = await recallBlock("q", "p", st)
    expect(second).not.toBeNull()
  })

  test("会话隔离：sA 的注入状态不影响 sB", async () => {
    stubFetch(() => ({ hits: [{ uid: "u1", content: "c", kind: "fact", project: "p" }] }))
    const sA = stateOf("g3-sA")
    const sB = stateOf("g3-sB")
    await recallBlock("q", "p", sA)
    const sBFirst = await recallBlock("q", "p", sB)
    expect(sBFirst).not.toBeNull()
  })

  test("服务离线 → null（fail-open，不注入不上报）", async () => {
    stubFetch(() => null)
    const st = stateOf("g2-offline")
    expect(await recallBlock("q", "p", st)).toBeNull()
  })

  test("空命中集合 → null", async () => {
    stubFetch(() => ({ hits: [] }))
    const st = stateOf("g2-empty")
    expect(await recallBlock("q", "p", st)).toBeNull()
  })
})

// ---------- 注入块格式与预算 ----------

describe("formatBlock：注入块格式与预算封顶", () => {
  test("预算封顶：超长命中行数截断，块以闭合标记收尾", () => {
    const hits = Array.from({ length: 50 }, (_, i) => ({
      uid: `u${i}`, content: `内容 ${i} `.repeat(200), kind: "fact", project: "p",
    }))
    const block = formatBlock(hits)
    expect(block.endsWith("</semantic-memory>")).toBe(true)
    // 预算 2000 默认：内容部分不得超过预算（头部+闭合标记不计入行截断）
    expect(block.length).toBeLessThanOrEqual(2000 + "</semantic-memory>".length + 100)
  })

  test("行格式：kind 前缀＋内容＋project 标注＋跨项目标注", () => {
    const block = formatBlock([{ uid: "u", content: "偏好", kind: "preference", project: "p", cross_kind: "co_injection" }])
    expect(block).toContain("[preference]")
    expect(block).toContain("(project: p)")
    expect(block).toContain("(cross-project co_injection)")
  })

  test("跨项目附加条目仍展示（尾注侧），但跨项目偏好不进冻结头块", async () => {
    // formatBlock 不过滤 cross_kind（尾注侧保留展示）；buildHeadBlock 排除（task11 不变量）
    stubFetch(() => ({ hits: [
      { uid: "a", content: "本域偏好", kind: "preference", project: "p", injections: 5, updated_at: 2 },
      { uid: "b", content: "他项目偏好", kind: "preference", project: "other", cross_kind: "co_injection", injections: 99, updated_at: 9 },
      { uid: "c", content: "本域事实", kind: "fact", project: "p", injections: 1, updated_at: 1 },
    ] }))
    const head = await buildHeadBlock("q", "p")
    expect(head).not.toBeNull()
    expect(head).toContain("本域偏好")
    expect(head).not.toContain("他项目偏好")  // cross_kind 排除
    expect(head).not.toContain("本域事实")    // 非 preference/decision 排除
  })
})

// ---------- 文本提取与转录 ----------

describe("partsText：防御式文本提取", () => {
  test("字符串直返", () => { expect(partsText("raw")).toBe("raw") })
  test("part 数组拼 text", () => {
    expect(partsText([{ type: "text", text: "a" }, { type: "tool", text: "ignore" }, { type: "text", text: "b" }])).toBe("a\nb")
  })
  test("非字符串非数组 → 空", () => { expect(partsText({})).toBe("") })
  test("脏 part（无 type/text 字符串）跳过", () => {
    expect(partsText([{ type: "text" }, { type: "text", text: "ok" }])).toBe("ok")
  })
})

describe("transcriptText：压缩托管防回流", () => {
  test("注入块行被过滤，用户/助手保留", () => {
    const msgs = [
      { role: "user", content: [{ type: "text", text: "真实问题" }] },
      { role: "assistant", content: [{ type: "text", text: "回答" }] },
      { role: "user", content: [{ type: "text", text: "<semantic-memory>\n历史\n</semantic-memory>" }] },
    ]
    const t = transcriptText({ messages: msgs })
    expect(t).toContain("真实问题")
    expect(t).not.toContain("<semantic-memory>")
  })
})

describe("recentUserText / firstUserText", () => {
  const msgs = [
    { role: "user", content: [{ type: "text", text: "第一条" }] },
    { role: "assistant", content: [{ type: "text", text: "答" }] },
    { role: "user", content: [{ type: "text", text: "最近一条" }] },
  ]
  test("recentUserText 取末条 user", () => { expect(recentUserText({ messages: msgs })).toBe("最近一条") })
  test("firstUserText 取首条 user", () => { expect(firstUserText({ messages: msgs })).toBe("第一条") })
})

describe("hashOf：注入集合指纹", () => {
  test("顺序无关，集合相同哈希相同", () => {
    expect(hashOf(["b", "a"])).toBe(hashOf(["a", "b"]))
  })
  test("集合不同哈希不同", () => {
    expect(hashOf(["a"])).not.toBe(hashOf(["a", "b"]))
  })
})
