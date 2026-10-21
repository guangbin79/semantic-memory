// TUI 侧边栏面板：显示**当前会话**实际注入进上下文的记忆块（semantic-memory 服务遥测）。
// 挂载点：sidebar.content 槽（与内置 Todo 面板同机制，@opencode/plugin/tui）。
// 数据源：GET /api/injections?session_id=<本会话>（注入上报携带的块全文），5s 轮询。
// 设计约束：纯只读展示——不往模型上下文加任何内容，服务离线时静默保持旧态。
//
// 安装（目录形态，TUI 发现只认目录/软链）：
//   ln -s ~/semantic-memory/integrations/opencode/tui-plugin ~/.config/opencode/plugins/semantic-memory-tui
// 入口约定：Host.resolve 把目录下 ./tui 解析为本文件（tui.tsx）。
import { Plugin } from "@opencode/plugin/tui"
import { createSignal, For, onCleanup, onMount, Show } from "solid-js"

const SERVICE = (process.env.MEMORY_PLUGIN_URL ?? "http://127.0.0.1:8765").replace(/\/$/, "")
const POLL_MS = 5000
const MAX_ROWS = 20

type Inj = { ts: number; hook: string; project: string | null; chars: number; block: string | null }

export function SidebarMemory(props: { context: Plugin.Context; sessionID: string }) {
  const [rows, setRows] = createSignal<Inj[]>([])
  const [view, updateView] = props.context.storage.store("open", { initial: { open: true } })
  const theme = props.context.theme

  const refresh = async () => {
    try {
      const res = await fetch(
        `${SERVICE}/api/injections?session_id=${encodeURIComponent(props.sessionID)}&limit=${MAX_ROWS}`,
      )
      if (!res.ok) return
      const data = await res.json()
      setRows(data.injections ?? [])
    } catch { /* 服务离线：保持旧态，下轮再试 */ }
  }
  onMount(() => {
    void refresh()
    const timer = setInterval(refresh, POLL_MS)
    onCleanup(() => clearInterval(timer))
  })

  const rel = (ts: number) => {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - ts))
    if (s < 60) return `${s}s`
    if (s < 3600) return `${Math.floor(s / 60)}m`
    if (s < 86400) return `${Math.floor(s / 3600)}h`
    return `${Math.floor(s / 86400)}d`
  }

  // 压缩语义：compaction-reset 行标记上下文重建点，其前的注入块已不在当前上下文——
  // 只展示最后一个标记之后的注入，标记行自身不展示。
  const visible = () => {
    const cutoff = rows().reduce((acc, r) => (r.hook === "compaction-reset" ? Math.max(acc, r.ts) : acc), 0)
    return rows().filter((r) => r.ts > cutoff && r.hook !== "compaction-reset")
  }

  return (
    <Show when={visible().length > 0}>
      <box>
        <box flexDirection="row" gap={1}>
          <text onMouseDown={() => void updateView((draft) => { draft.open = !draft.open }).catch(() => {})}>
            {view.open ? "▼" : "▶"}
          </text>
          <text fg={theme.text.base}><b>记忆注入</b></text>
          <text fg={theme.text.muted}>{visible().length}</text>
        </box>
        <Show when={view.open}>
          <For each={visible()}>{(inj) => (
            <box paddingLeft={2}>
              <text fg={theme.text.muted}>{rel(inj.ts)} · {inj.project ?? "-"} · {inj.hook} · {inj.chars}c</text>
              <text fg={theme.text.base}>{inj.block ?? ""}</text>
            </box>
          )}</For>
        </Show>
      </box>
    </Show>
  )
}

export default Plugin.define({
  id: "opencode.sidebar.memory",
  setup(context) {
    context.ui.slot({
      append: "sidebar.content",
      render: (props) => <SidebarMemory context={context} sessionID={props.sessionID} />,
    })
  },
})
