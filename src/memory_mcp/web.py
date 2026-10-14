"""内置浏览页：单文件自包含 HTML，由服务进程直接吐出（GET /）。

零构建、零框架、零外部资源——离线可用。功能刻意收敛：
统计总览 / 关键词检索 / 列表浏览（kind、project 过滤）/ 单条遗忘。
内容渲染一律 textContent，防记忆文本里的注入。
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>semantic-memory</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font: 14px/1.6 ui-monospace, "Cascadia Mono", Consolas, monospace;
         background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; max-width: 1100px; margin-inline: auto; }
  h1 { font-size: 18px; color: #58a6ff; margin: 0 0 4px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 16px; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; background: #161b22;
           border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
  .stat b { color: #58a6ff; font-size: 18px; display: block; }
  .stat span { color: #8b949e; font-size: 11px; }
  form { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  input, select { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
                  border-radius: 6px; padding: 6px 10px; font: inherit; }
  input#q { flex: 1; min-width: 220px; }
  button { background: #238636; border: 0; color: #fff; border-radius: 6px;
           padding: 6px 14px; cursor: pointer; font: inherit; }
  button.ghost { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; }
  button.danger { background: #6e2323; padding: 2px 8px; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262d; vertical-align: top; }
  th { color: #8b949e; font-size: 11px; text-transform: uppercase; }
  .kind { display: inline-block; padding: 0 6px; border-radius: 4px; font-size: 11px; background: #1f6feb33; color: #79c0ff; }
  .kind.preference { background: #9e6a0333; color: #e3b341; }
  .kind.decision { background: #8957e533; color: #d2a8ff; }
  .kind.artifact { background: #30363d; color: #8b949e; }
  .meta { color: #8b949e; font-size: 11px; white-space: nowrap; }
  .score { color: #3fb950; font-size: 11px; }
  #status { color: #8b949e; font-size: 12px; margin: 8px 0; }
  .maint { margin-bottom: 16px; display: flex; gap: 8px; align-items: center; }
  #maint-status { color: #3fb950; font-size: 12px; }
</style>
</head>
<body>
<h1>semantic-memory</h1>
<div class="sub">上下文相关性侧车 · 本地浏览（只读 + 单条遗忘）</div>
<div class="stats" id="stats"></div>
<div class="maint">
  <button class="ghost" id="btn-purge">运行清理（purge）</button>
  <button class="ghost" id="btn-backfill">向量补嵌（backfill）</button>
  <span id="maint-status"></span>
</div>
<details id="telemetry" open>
  <summary>注入统计（供给侧：实际注入次数与内容；模型是否采纳不可度量）</summary>
  <div class="stats" id="inj-stats"></div>
  <div class="sub" id="inj-dist"></div>
  <h2>反复被注入的记忆 Top10</h2>
  <table><thead><tr><th>次数</th><th>kind</th><th>内容</th></tr></thead>
    <tbody id="inj-top"></tbody></table>
  <h2>最近注入</h2>
  <table><thead><tr><th>时间</th><th>harness</th><th>钩子</th><th>触发查询</th><th>条目</th><th>字符</th></tr></thead>
    <tbody id="inj-rows"></tbody></table>
</details>
<form id="f">
  <input id="q" placeholder="检索（走服务端混合召回 RRF）…">
  <select id="kind"><option value="">全部 kind</option>
    <option>fact</option><option>decision</option><option>preference</option><option>artifact</option></select>
  <input id="project" placeholder="project 过滤" size="12">
  <button type="submit">检索</button>
  <button type="button" class="ghost" id="recent">最近</button>
</form>
<div id="status"></div>
<table><thead><tr>
  <th>更新/创建时间</th><th>kind</th><th>source</th><th>project</th><th>内容</th><th>tags</th><th></th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
let token = localStorage.getItem('memory_token') || '';
async function api(path, opts = {}) {
  opts.headers = Object.assign({'content-type': 'application/json'},
    token ? {authorization: 'Bearer ' + token} : {}, opts.headers || {});
  let r = await fetch(path, opts);
  if (r.status === 401) {
    token = prompt('服务要求 MEMORY_TOKEN：') || '';
    localStorage.setItem('memory_token', token);
    r = await fetch(path, opts);
  }
  if (!r.ok) throw new Error(r.status);
  return r.json();
}
const $ = id => document.getElementById(id);
const esc = (p, t) => { const e = document.createElement('td'); e.textContent = t ?? ''; p.appendChild(e); return e; };
const fmt = ts => ts ? new Date(ts * 1000).toLocaleString('zh-CN', {hour12: false}) : '';

async function loadStats() {
  const s = await api('/api/stats');
  $('stats').innerHTML = '';
  const items = [['总条目', s.total], ['已向量化', s.vectorized],
    ['库体积', (s.db_bytes / 1048576).toFixed(1) + ' MB'],
    ['向量通道', s.embedder ? (s.device || 'on') : 'BM25-only'],
    ...Object.entries(s.by_kind || {})];
  for (const [k, v] of items) {
    const d = document.createElement('div'); d.className = 'stat';
    const b = document.createElement('b'); b.textContent = v;
    const sp = document.createElement('span'); sp.textContent = k;
    d.append(b, sp); $('stats').appendChild(d);
  }
}

async function loadInjections() {
  const s = await api('/api/stats');
  const inj = s.injections || {total: 0, total_chars: 0, recent_7d: 0, by_harness: {}, by_hook: {}, top: []};
  $('inj-stats').innerHTML = '';
  const cards = [['累计注入', inj.total], ['近 7 天', inj.recent_7d],
    ['累计字符', inj.total_chars], ['≈tokens(粗估)', Math.round(inj.total_chars / 4)]];
  for (const [k, v] of cards) {
    const d = document.createElement('div'); d.className = 'stat';
    const b = document.createElement('b'); b.textContent = v;
    const sp = document.createElement('span'); sp.textContent = k;
    d.append(b, sp); $('inj-stats').appendChild(d);
  }
  const dist = [];
  for (const [k, v] of Object.entries(inj.by_harness)) dist.push('harness ' + (k || '?') + ' ×' + v);
  for (const [k, v] of Object.entries(inj.by_hook)) dist.push('钩子 ' + (k || '?') + ' ×' + v);
  $('inj-dist').textContent = dist.join(' · ') || '暂无注入记录——插件完成首次注入后此处有数';
  $('inj-top').innerHTML = '';
  for (const t of inj.top) {
    if (!t.content) continue;  // 已被遗忘的 uid
    const tr = document.createElement('tr');
    esc(tr, t.times);
    const k = document.createElement('span'); k.className = 'kind ' + t.kind; k.textContent = t.kind;
    const td = document.createElement('td'); td.appendChild(k); tr.appendChild(td);
    esc(tr, t.content);
    $('inj-top').appendChild(tr);
  }
  const list = (await api('/api/injections?limit=20')).injections;
  $('inj-rows').innerHTML = '';
  for (const r of list) {
    const tr = document.createElement('tr');
    esc(tr, fmt(r.ts)); esc(tr, r.harness || '?'); esc(tr, r.hook || '?');
    esc(tr, (r.query || '').slice(0, 60));
    esc(tr, JSON.parse(r.uids || '[]').length); esc(tr, r.chars);
    $('inj-rows').appendChild(tr);
  }
}

function render(hits, withScore) {
  $('rows').innerHTML = '';
  for (const h of hits) {
    const tr = document.createElement('tr');
    esc(tr, fmt(h.updated_at || h.created_at));
    const k = document.createElement('span'); k.className = 'kind ' + h.kind; k.textContent = h.kind;
    const kc = document.createElement('td'); kc.appendChild(k); tr.appendChild(kc);
    esc(tr, h.source); esc(tr, h.project || '—');
    const c = esc(tr, h.content);
    if (withScore) { const s = document.createElement('div'); s.className = 'score';
      s.textContent = 'rrf ' + h.score; c.appendChild(s); }
    esc(tr, JSON.parse(h.tags || '[]').join(', ')).className = 'meta';
    const b = document.createElement('button'); b.className = 'danger'; b.textContent = '遗忘';
    b.onclick = async () => { if (!confirm('遗忘此条？（软删除，purge 后彻底清除）')) return;
      await api('/api/memories/' + h.uid, {method: 'DELETE'}); refresh(); };
    const bc = document.createElement('td'); bc.appendChild(b); tr.appendChild(bc);
    $('rows').appendChild(tr);
  }
  $('status').textContent = hits.length + ' 条';
}

async function recent() {
  const p = new URLSearchParams();
  if ($('kind').value) p.set('kind', $('kind').value);
  if ($('project').value) p.set('project', $('project').value);
  p.set('limit', '50');
  render((await api('/api/memories?' + p)).memories, false);
}
async function search(q) {
  render((await api('/recall', {method: 'POST',
    body: JSON.stringify({query: q, limit: 20, project: $('project').value || undefined})})).hits, true);
}
async function refresh() { await loadStats(); await loadInjections(); await recent(); }
$('f').onsubmit = e => { e.preventDefault(); $('q').value.trim() ? search($('q').value.trim()) : recent(); };
$('recent').onclick = recent;

async function maintain(path, label, confirmText) {
  if (!confirm(confirmText)) return;
  $('maint-status').textContent = label + ' 运行中…';
  try {
    const r = await api(path, {method: 'POST'});
    if (r.error) { $('maint-status').textContent = r.error; return; }
    $('maint-status').textContent = label + ' 完成：' + JSON.stringify(r);
    refresh();
  } catch (e) {
    $('maint-status').textContent = label + ' 失败：' + e.message +
      '（未配置 MEMORY_TOKEN 时批量维护仅 CLI 可用）';
  }
}
$('btn-purge').onclick = () => maintain('/api/maintenance/purge', 'purge',
  '硬删软删除条目 + TTL 清理 + VACUUM。确认执行？');
$('btn-backfill').onclick = () => maintain('/api/maintenance/backfill', 'backfill',
  '为所有缺向量条目补算嵌入（可能耗时）。确认执行？');

refresh().catch(e => $('status').textContent = '服务不可达：' + e.message);
</script>
</body>
</html>
"""
