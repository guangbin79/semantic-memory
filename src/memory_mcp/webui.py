"""内置 Web 浏览页：单文件 HTML（内联 CSS/JS，零构建），由服务同端口吐出。

只读设计：浏览/搜索/过滤/详情，不提供删除入口（删除走 agent 单条 forget
或 CLI purge——破坏性操作不进浏览器面）。鉴权复用 MEMORY_TOKEN：
页面本身可加载，API 返回 401 时页面提示输入 token（存 localStorage）。
"""

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>semantic-memory · 编排历史</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --ink:#1c1c1e; --dim:#6b6b70; --line:#e4e4e0;
          --fact:#2f6fed; --decision:#0f9d6c; --preference:#b4237a; --artifact:#c77700; }
  * { box-sizing:border-box; margin:0; }
  body { font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
         background:var(--bg); color:var(--ink); }
  .wrap { max-width:960px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:18px; display:flex; align-items:baseline; gap:10px; }
  h1 small { color:var(--dim); font-weight:normal; font-size:12px; }
  .stats { display:flex; gap:10px; margin:16px 0; flex-wrap:wrap; }
  .chip { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:8px 14px; font-size:12px; color:var(--dim); }
  .chip b { color:var(--ink); font-size:16px; display:block; }
  .bar { display:flex; gap:8px; margin:14px 0; flex-wrap:wrap; }
  input,select { font:inherit; padding:8px 10px; border:1px solid var(--line);
                 border-radius:8px; background:var(--card); }
  #q { flex:1; min-width:200px; }
  .list { display:flex; flex-direction:column; gap:8px; }
  .item { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:12px 14px; cursor:pointer; }
  .item:hover { border-color:#c9c9c4; }
  .meta { display:flex; gap:8px; align-items:center; font-size:12px; color:var(--dim);
          margin-top:6px; flex-wrap:wrap; }
  .badge { font-size:11px; padding:1px 8px; border-radius:99px; color:#fff; }
  .badge.fact{background:var(--fact)} .badge.decision{background:var(--decision)}
  .badge.preference{background:var(--preference)} .badge.artifact{background:var(--artifact)}
  .score { margin-left:auto; font-family:ui-monospace,monospace; }
  .detail { white-space:pre-wrap; margin-top:10px; padding-top:10px;
            border-top:1px dashed var(--line); font-size:13.5px; }
  .empty { color:var(--dim); text-align:center; padding:48px 0; }
  .more { display:block; margin:16px auto; padding:8px 24px; border:1px solid var(--line);
          background:var(--card); border-radius:8px; cursor:pointer; }
  #tokenbar { display:none; background:#fff8e6; border:1px solid #f0dfa0;
              border-radius:8px; padding:10px 14px; margin:12px 0; gap:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>semantic-memory <small>编排历史 · 跨会话 / 跨 harness 记忆浏览</small></h1>
  <div id="tokenbar">需要访问令牌：<input id="token" placeholder="MEMORY_TOKEN" size="24">
    <button onclick="saveToken()">保存</button></div>
  <div class="stats" id="stats"></div>
  <div class="bar">
    <input id="q" placeholder="语义搜索（措辞不同也能命中）…">
    <select id="kind"><option value="">全部类型</option><option>fact</option>
      <option>decision</option><option>preference</option><option>artifact</option></select>
    <select id="project"><option value="">全部项目</option></select>
    <button class="more" style="margin:0" onclick="load(true)">查询</button>
  </div>
  <div class="list" id="list"></div>
  <div id="morewrap"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let offset = 0, total = 0, searching = false;
const KIND_CN = {fact:'事实', decision:'决策', preference:'偏好', artifact:'产物'};
const SRC_CN = {manual:'手动', chat:'对话', compaction:'压缩托管', tool:'工具'};
const authH = () => localStorage.memToken ? {authorization:'Bearer '+localStorage.memToken} : {};
const fmt = t => new Date(t*1000).toLocaleString('zh-CN', {hour12:false});
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function api(path) {
  const r = await fetch(path, {headers: authH()});
  if (r.status === 401) { $('tokenbar').style.display = 'flex'; throw new Error('401'); }
  return r.json();
}
function saveToken() { localStorage.memToken = $('token').value; $('tokenbar').style.display='none'; boot(); }

async function stats() {
  const s = await api('/api/stats');
  $('stats').innerHTML =
    `<div class="chip"><b>${s.total}</b>记忆总数</div>
     <div class="chip"><b>${s.vectorized}</b>已向量化</div>
     <div class="chip"><b>${s.embedder ? s.device||'?' : 'off'}</b>向量通道</div>
     <div class="chip"><b>${(s.db_bytes/1048576).toFixed(1)}MB</b>库体积</div>`;
  const ps = await api('/api/projects');
  ps.forEach(p => $('project').insertAdjacentHTML('beforeend', `<option>${esc(p)}</option>`));
}

function itemHtml(m, score) {
  return `<div class="item" onclick="toggle(this, '${m.uid}')">
    <div>${esc(m.content.slice(0, 140))}${m.content.length>140?'…':''}</div>
    <div class="meta">
      <span class="badge ${m.kind}">${KIND_CN[m.kind]||m.kind}</span>
      <span>${SRC_CN[m.source]||m.source}</span>
      ${m.project?`<span>📁 ${esc(m.project)}</span>`:''}
      <span>${fmt(m.created_at)}</span>
      ${score!=null?`<span class="score">RRF ${score}</span>`:''}
    </div><div class="detail" style="display:none"></div></div>`;
}

async function toggle(el, uid) {
  const d = el.querySelector('.detail');
  if (d.style.display === 'block') { d.style.display = 'none'; return; }
  if (!d.dataset.loaded) {
    const m = await api('/api/memories/' + uid);
    d.textContent = m.content + '\n\n— 更新于 ' + fmt(m.updated_at) + (m.tags && m.tags!=='[]' ? ' · tags: '+m.tags : '');
    d.dataset.loaded = 1;
  }
  d.style.display = 'block';
}

async function load(reset) {
  if (reset) { offset = 0; $('list').innerHTML = ''; }
  const q = $('q').value.trim(), kind = $('kind').value, project = $('project').value;
  searching = !!q;
  let data, hits;
  if (searching) {
    data = await api(`/api/search?q=${encodeURIComponent(q)}&limit=50${project?`&project=${encodeURIComponent(project)}`:''}`);
    hits = data.hits;
  } else {
    const p = new URLSearchParams({limit:'50', offset:String(offset)});
    if (kind) p.set('kind', kind);
    if (project) p.set('project', project);
    data = await api('/api/memories?' + p); total = data.total; hits = data.items;
  }
  $('list').insertAdjacentHTML('beforeend',
    hits.length ? hits.map(m => itemHtml(m, m.score)).join('') :
    (reset ? '<div class="empty">没有匹配的记忆</div>' : ''));
  offset += hits.length;
  $('morewrap').innerHTML = (!searching && offset < total)
    ? `<button class="more" onclick="load(false)">加载更多（${offset}/${total}）</button>` : '';
}

async function boot() { try { await stats(); await load(true); } catch(e) {} }
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') load(true); });
boot();
</script>
</body>
</html>"""
