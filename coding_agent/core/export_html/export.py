"""
Rich HTML export for agent sessions.

Renders a self-contained two-column page:
  - Left sidebar: search + filter tabs + message navigator
  - Right main: metadata grid, system prompt card, tools card, full message thread
  - Per-message copy (📋) and shareable anchor link (🔗) buttons
  - Download JSONL button + keyboard-driven think/tool visibility toggles
"""
from __future__ import annotations
import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path

import aiofiles

from ..session_manager import SessionManager, _header_to_dict, _entry_to_dict
from ..types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    LabelEntry,
    ModelChangeEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)
from .tool_renderer import ToolHtmlRenderer

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:#13141a;--surface:#1c1f26;--surface2:#21262d;--border:#30363d;
  --text:#c9d1d9;--muted:#8b949e;
  --blue:#58a6ff;--green:#3fb950;--purple:#a371f7;
  --yellow:#d29922;--red:#f85149;--cyan:#79c0ff;
}
*{box-sizing:border-box;}
html,body{height:100%;margin:0;padding:0;}
body{font-family:'Courier New',Courier,monospace;background:var(--bg);
  color:var(--text);font-size:14px;line-height:1.5;overflow:hidden;}
a{color:var(--blue);}

/* Two-column layout */
.layout{display:flex;height:100vh;}

/* Sidebar */
.sidebar{width:300px;flex-shrink:0;background:var(--surface);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  overflow:hidden;}
.sb-header{padding:.6rem .75rem .4rem;border-bottom:1px solid var(--border);
  flex-shrink:0;}
.sb-title{color:var(--muted);font-size:.72em;font-weight:bold;
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem;}
.sb-search{width:100%;background:var(--surface2);border:1px solid var(--border);
  color:var(--text);padding:.3rem .5rem;border-radius:3px;
  font-family:inherit;font-size:.78em;outline:none;}
.sb-search:focus{border-color:var(--blue);}
.sb-tabs{display:flex;gap:.2rem;padding:.35rem .5rem;flex-wrap:wrap;
  border-bottom:1px solid var(--border);flex-shrink:0;}
.tab{background:none;border:1px solid transparent;color:var(--muted);
  padding:.15rem .4rem;border-radius:3px;cursor:pointer;font-family:inherit;
  font-size:.72em;white-space:nowrap;}
.tab:hover{color:var(--text);border-color:var(--border);}
.tab.active{background:var(--surface2);border-color:var(--border);
  color:var(--blue);}
.sb-list{flex:1;overflow-y:auto;padding:.25rem 0;}
.sb-item{padding:.3rem .75rem;cursor:pointer;border-left:2px solid transparent;
  display:flex;align-items:flex-start;gap:.4rem;font-size:.76em;line-height:1.4;}
.sb-item:hover{background:var(--surface2);border-left-color:var(--border);}
.sb-item.sb-user{border-left-color:transparent;}
.sb-item.sb-user:hover{border-left-color:var(--blue);}
.sb-item.sb-asst:hover{border-left-color:var(--green);}
.sb-item.sb-ev{opacity:.6;}
.sb-badge{flex-shrink:0;font-size:.68em;font-weight:bold;text-transform:uppercase;
  padding:.05rem .3rem;border-radius:2px;margin-top:.1rem;}
.sb-badge-user{color:var(--blue);background:rgba(88,166,255,.12);}
.sb-badge-asst{color:var(--green);background:rgba(63,185,80,.12);}
.sb-badge-tool{color:var(--yellow);background:rgba(210,153,34,.12);}
.sb-badge-ev{color:var(--muted);background:var(--surface2);}
.sb-preview{color:var(--muted);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex:1;min-width:0;}

/* Main panel */
.main{flex:1;overflow-y:auto;padding:1.5rem 2rem 4rem;min-width:0;}

/* Header */
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:.9rem;flex-wrap:wrap;gap:.5rem;}
.sid{color:var(--cyan);font-size:1em;font-weight:bold;}
.sname{color:var(--muted);font-size:.82em;margin-top:.15rem;}
.hdr-btns{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap;}
.btn{background:var(--surface2);border:1px solid var(--border);color:var(--text);
  padding:.18rem .55rem;border-radius:3px;cursor:pointer;font-family:inherit;
  font-size:.78em;text-decoration:none;user-select:none;}
.btn:hover{background:var(--border);}
.btn-dl{border-color:var(--blue);color:var(--blue);}
.hints{color:var(--muted);font-size:.75em;margin-bottom:1rem;}
.hints kbd{background:var(--surface2);border:1px solid var(--border);
  border-radius:2px;padding:0 .3rem;font-size:.9em;}

/* Metadata grid */
.meta{display:grid;grid-template-columns:max-content 1fr;
  gap:.3rem 1.4rem;font-size:.84em;margin-bottom:1.3rem;}
.mk{color:var(--muted);}
.mv{color:var(--text);}
.ti{color:var(--blue);}
.to{color:var(--green);}
.tr{color:var(--muted);}

/* Cards */
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:6px;padding:.9rem 1.1rem;margin-bottom:.9rem;}
.ctitle{color:var(--purple);font-size:.76em;font-weight:bold;
  letter-spacing:.09em;text-transform:uppercase;margin-bottom:.65rem;}

/* System prompt */
.sys-body{color:var(--muted);font-size:.82em;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;margin:0;}
.sys-collapsed{max-height:7.5em;overflow:hidden;}
.sys-btn{color:var(--blue);cursor:pointer;font-style:italic;background:none;
  border:none;font-family:inherit;font-size:.78em;padding:.25rem 0 0;
  display:block;}

/* Tools */
.tool-row{font-size:.82em;margin-bottom:.3rem;}
.tool-row summary{list-style:none;cursor:pointer;line-height:1.6;}
.tool-row summary::-webkit-details-marker{display:none;}
.tn{color:var(--blue);font-weight:bold;}
.td{color:var(--muted);}
.tp{color:var(--green);font-style:italic;}
.tparams{background:var(--surface2);border:1px solid var(--border);
  border-radius:3px;padding:.4rem .7rem;margin-top:.3rem;
  color:var(--muted);font-size:.86em;white-space:pre-wrap;}

/* Messages */
.msgs{margin-top:1.3rem;}
.msg{margin:.45rem 0;border-radius:4px;overflow:hidden;scroll-margin-top:.5rem;}
.msg-user{background:var(--surface);border-left:3px solid var(--blue);}
.msg-asst{border-left:3px solid var(--green);}
.mhead{display:flex;align-items:center;gap:.6rem;padding:.28rem .75rem;
  border-bottom:1px solid var(--border);}
.mrole{font-size:.7em;font-weight:bold;text-transform:uppercase;}
.ru{color:var(--blue);}
.ra{color:var(--green);}
.mts{font-size:.68em;color:var(--muted);margin-left:auto;}
.mactions{display:flex;gap:.2rem;align-items:center;margin-left:.4rem;}
.mact{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:.82em;padding:.05rem .2rem;border-radius:2px;line-height:1;
  font-family:inherit;opacity:.5;}
.mact:hover{opacity:1;background:var(--surface2);}
.mbody{padding:.5rem .75rem .6rem;}
pre.mtext{white-space:pre-wrap;word-break:break-word;
  font-size:.87em;line-height:1.65;margin:0;}

/* Tool call blocks */
.tblock{background:var(--surface2);border:1px solid var(--border);
  border-radius:4px;margin:.32rem 0;}
.tblock>summary{list-style:none;padding:.32rem .55rem;cursor:pointer;
  color:var(--yellow);font-size:.82em;display:flex;align-items:flex-start;gap:.35rem;}
.tblock>summary::-webkit-details-marker{display:none;}
.ticon{color:var(--muted);font-size:.78em;flex-shrink:0;padding-top:.1rem;}
.trwrap{padding:.32rem .55rem;border-top:1px solid var(--border);
  max-height:22em;overflow-y:auto;}
pre.tr{font-size:.76em;white-space:pre-wrap;word-break:break-word;
  margin:0;color:var(--green);}
pre.tr.err{color:var(--red);}

/* Thinking */
.thblock{border-left:2px solid var(--border);margin:.3rem 0;}
.thblock>summary{list-style:none;padding:.28rem .55rem;cursor:pointer;
  color:var(--muted);font-size:.78em;font-style:italic;}
.thblock>summary::-webkit-details-marker{display:none;}
.thtext{padding:.3rem .55rem;color:var(--muted);
  font-size:.78em;white-space:pre-wrap;line-height:1.5;}

/* Tool-call section inside assistant messages */
.tools-section{margin:.65rem 0;border:1px solid var(--border);
  border-left:3px solid var(--yellow);border-radius:4px;
  background:var(--surface);}
.tools-section-hdr{padding:.22rem .65rem;font-size:.71em;color:var(--yellow);
  font-weight:600;letter-spacing:.04em;border-bottom:1px solid var(--border);}
/* Response-text section (text + thinking blocks) */
.resp-section{padding:.1rem 0;}
/* Event bars */
.ev{padding:.18rem 0;font-size:.74em;color:var(--muted);margin:.22rem 0;}
.ev span{color:var(--purple);}
.compact-bar{background:var(--surface);border:1px solid var(--yellow);
  color:var(--yellow);padding:.3rem .75rem;margin:.5rem 0;
  font-size:.76em;border-radius:4px;}

/* Skills-loaded card */
.skills-loaded-card{background:var(--surface);border:1px solid var(--border);
  border-left:3px solid var(--green);border-radius:4px;
  padding:.45rem .75rem;margin:.5rem 0;font-size:.76em;}
.skills-loaded-title{color:var(--green);font-weight:600;margin-bottom:.35rem;}
.skill-row{display:flex;gap:.6rem;align-items:baseline;
  padding:.1rem 0;border-top:1px solid var(--border);}
.skill-row:first-of-type{border-top:none;}
.skill-name{color:var(--blue);font-weight:600;min-width:9rem;flex-shrink:0;}
.skill-desc{color:var(--text);flex:1;}
.skill-path{color:var(--muted);font-size:.88em;word-break:break-all;}

/* Visibility toggles */
.thinking-hidden .thblock{display:none;}
.tools-hidden .tblock{display:none;}

/* Toast notification */
.toast{position:fixed;bottom:1.5rem;right:1.5rem;background:var(--surface2);
  border:1px solid var(--border);color:var(--text);padding:.4rem .9rem;
  border-radius:4px;font-size:.8em;opacity:0;pointer-events:none;
  transition:opacity .2s;z-index:999;}
.toast.show{opacity:1;}
"""

# ── JS ────────────────────────────────────────────────────────────────────────

_JS = """
// ── Visibility toggles ──────────────────────────────────────────────────────
var _thHide=false,_tlHide=false;
function toggleThinking(){
  _thHide=!_thHide;
  document.getElementById('main').classList.toggle('thinking-hidden',_thHide);
}
function toggleTools(){
  _tlHide=!_tlHide;
  document.getElementById('main').classList.toggle('tools-hidden',_tlHide);
}
function toggleSys(btn){
  var el=document.getElementById('sys-body');
  var col=el.classList.toggle('sys-collapsed');
  btn.textContent=col?btn.dataset.more:'(click to collapse)';
}
function dlJsonl(){
  var b64=document.getElementById('jdata').value;
  var bin=atob(b64);
  var arr=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
  var blob=new Blob([arr],{type:'application/x-ndjson'});
  var a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='session.jsonl';a.click();
}

// ── Toast ────────────────────────────────────────────────────────────────────
var _toastTimer=null;
function showToast(msg){
  var t=document.getElementById('toast');
  t.textContent=msg;t.classList.add('show');
  if(_toastTimer)clearTimeout(_toastTimer);
  _toastTimer=setTimeout(function(){t.classList.remove('show');},1600);
}

// ── Per-message actions ──────────────────────────────────────────────────────
function copyMsg(btn){
  var id=btn.dataset.msgid;
  var el=document.getElementById(id);
  if(!el)return;
  var texts=[];
  el.querySelectorAll('pre.mtext').forEach(function(p){texts.push(p.textContent);});
  navigator.clipboard.writeText(texts.join('\\n')).then(function(){
    showToast('Copied message text');
  });
}
function copyLink(btn){
  var id=btn.dataset.msgid;
  var url=window.location.href.split('#')[0]+'#'+id;
  navigator.clipboard.writeText(url).then(function(){
    showToast('Link copied');
  });
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function scrollToMsg(id){
  var el=document.getElementById(id);
  if(!el)return;
  el.scrollIntoView({behavior:'smooth',block:'start'});
  // Briefly highlight
  el.style.outline='1px solid var(--blue)';
  setTimeout(function(){el.style.outline='';},1200);
}

function applyFilter(){
  var search=(document.getElementById('sb-search').value||'').toLowerCase();
  var activeTab=document.querySelector('.tab.active');
  var filter=activeTab?activeTab.dataset.filter:'default';
  document.querySelectorAll('.sb-item').forEach(function(item){
    var role=item.dataset.role||'';
    var labeled=item.dataset.labeled==='1';
    var hastools=item.dataset.hastools==='1';
    var preview=(item.querySelector('.sb-preview')||{}).textContent||'';
    var matchSearch=!search||preview.toLowerCase().indexOf(search)!==-1;
    var matchFilter=true;
    if(filter==='default'){matchFilter=role==='user'||role==='assistant'||role==='ev';}
    else if(filter==='notools'){matchFilter=(role==='user'||(role==='assistant'&&!hastools));}
    else if(filter==='user'){matchFilter=role==='user';}
    else if(filter==='labeled'){matchFilter=labeled;}
    // 'all' shows everything
    item.style.display=(matchSearch&&matchFilter)?'':'none';
  });
}

document.addEventListener('DOMContentLoaded',function(){
  document.getElementById('sb-search').addEventListener('input',applyFilter);
  document.querySelectorAll('.tab').forEach(function(btn){
    btn.addEventListener('click',function(){
      document.querySelectorAll('.tab').forEach(function(b){b.classList.remove('active');});
      btn.classList.add('active');
      applyFilter();
    });
  });
  applyFilter();
  // Handle initial anchor (page load with #msg-xxx)
  if(window.location.hash){
    var el=document.getElementById(window.location.hash.slice(1));
    if(el)setTimeout(function(){el.scrollIntoView({block:'start'});},120);
  }
});

document.addEventListener('keydown',function(e){
  if(e.ctrlKey&&e.key==='t'){e.preventDefault();toggleThinking();}
  if(e.ctrlKey&&e.key==='o'){e.preventDefault();toggleTools();}
});
"""

# ── Template ──────────────────────────────────────────────────────────────────

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Session {{SID}}</title>
<style>
{{CSS}}
</style>
</head>
<body>
<div class="layout">
<div class="sidebar">
  <div class="sb-header">
    <div class="sb-title">Messages</div>
    <input class="sb-search" id="sb-search" type="text" placeholder="Search..." autocomplete="off" />
  </div>
  <div class="sb-tabs">
    <button class="tab active" data-filter="default">Default</button>
    <button class="tab" data-filter="notools">No-tools</button>
    <button class="tab" data-filter="user">User</button>
    <button class="tab" data-filter="labeled">Labeled</button>
    <button class="tab" data-filter="all">All</button>
  </div>
  <div class="sb-list" id="sb-list">
{{SIDEBAR}}
  </div>
</div>
<div class="main" id="main">
{{CONTENT}}
</div>
</div>
<div class="toast" id="toast"></div>
<textarea id="jdata" style="display:none">{{JDATA}}</textarea>
<script>
{{JS}}
</script>
</body>
</html>"""


# ── Small utilities ───────────────────────────────────────────────────────────

def _attr(obj, key, default=None):
    """Uniform attribute access for both dict and dataclass/object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _usage_int(usage, *keys) -> int:
    for k in keys:
        v = _attr(usage, k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%-m/%-d/%Y, %-I:%M:%S %p")
    except Exception:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%m/%d/%Y, %I:%M:%S %p")
        except Exception:
            return ts


def _fmt_ts_short(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%I:%M:%S %p").lstrip("0")
    except Exception:
        return ""


def _num(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 10000:
        return f"{n / 1000:.1f}k"
    return f"{n // 1000}k"


# ── Statistics ────────────────────────────────────────────────────────────────

def _extract_stats(sm: SessionManager) -> dict:
    stats: dict = {
        "models": [],
        "user_msgs": 0,
        "asst_msgs": 0,
        "tool_results": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    for entry in sm._entries:
        if isinstance(entry, ModelChangeEntry):
            m = f"{entry.provider}/{entry.model_id}" if entry.provider else entry.model_id
            if m and m not in stats["models"]:
                stats["models"].append(m)
        elif isinstance(entry, SessionMessageEntry) and entry.message is not None:
            msg = entry.message
            role = _attr(msg, "role", "")
            if role == "user":
                stats["user_msgs"] += 1
            elif role == "assistant":
                stats["asst_msgs"] += 1
                content = _attr(msg, "content", []) or []
                for item in (content if isinstance(content, list) else []):
                    if _attr(item, "type", "") in ("toolCall", "tool_use"):
                        stats["tool_calls"] += 1
                usage = _attr(msg, "usage", None)
                if usage:
                    stats["input_tokens"] += _usage_int(usage, "inputTokens", "input_tokens")
                    stats["output_tokens"] += _usage_int(usage, "outputTokens", "output_tokens")
                    stats["cache_read"] += _usage_int(usage, "cacheReadTokens", "cache_read_tokens")
                    stats["cache_write"] += _usage_int(
                        usage, "cacheCreationTokens", "cache_write_tokens", "cacheWriteTokens"
                    )
            elif role == "toolResult":
                stats["tool_results"] += 1
    return stats


# ── Labeled IDs ───────────────────────────────────────────────────────────────

def _collect_labeled_ids(sm: SessionManager) -> set[str]:
    """Return the set of entry IDs that have a LabelEntry pointing at them."""
    labeled: set[str] = set()
    for entry in sm._entries:
        if isinstance(entry, LabelEntry) and entry.target_id:
            labeled.add(entry.target_id)
    return labeled


# ── System-prompt tool parser ─────────────────────────────────────────────────

def _parse_tools_from_prompt(prompt: str) -> list[dict]:
    """Extract tool names + descriptions from an '## Available Tools' section."""
    tools: list[dict] = []
    if not prompt:
        return tools
    m = re.search(r"## Available Tools\n(.*?)(?=\n##|\Z)", prompt, re.DOTALL)
    if not m:
        return tools
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # "- **name**: description"
        mm = re.match(r"-\s+\*\*(\S+?)\*\*:\s*(.*)", line)
        if not mm:
            # "- name: description"
            mm = re.match(r"-\s+(\S+?):\s*(.*)", line)
        if mm:
            tools.append({"name": mm.group(1), "description": mm.group(2).strip()})
    return tools


# ── HTML section renderers ────────────────────────────────────────────────────

def _render_header(sm: SessionManager) -> str:
    sid = html.escape(sm.get_session_id())
    name = sm.get_session_name()
    name_html = f'<div class="sname">{html.escape(name)}</div>' if name else ""
    return (
        f'<div class="hdr">'
        f'<div><div class="sid">Session: {sid}</div>{name_html}</div>'
        f'<div class="hdr-btns">'
        f'<button class="btn" onclick="toggleThinking()">Ctrl+T toggle thinking</button>'
        f'<button class="btn" onclick="toggleTools()">Ctrl+O toggle tools</button>'
        f'<button class="btn btn-dl" onclick="dlJsonl()">&#8595; JSONL</button>'
        f'</div>'
        f'</div>'
        f'<div class="hints">'
        f'<kbd>Ctrl+T</kbd> toggle thinking &nbsp;&middot;&nbsp; '
        f'<kbd>Ctrl+O</kbd> toggle tools'
        f'</div>'
    )


def _render_metadata(sm: SessionManager, stats: dict) -> str:
    ts = _fmt_ts(sm._header.timestamp)
    models_str = html.escape(", ".join(stats["models"]) or "—")
    msgs_str = (
        f"{stats['user_msgs']} user, {stats['asst_msgs']} assistant"
        + (f", {stats['tool_results']} tool results" if stats["tool_results"] else "")
    )
    tok_in, tok_out, tok_r = stats["input_tokens"], stats["output_tokens"], stats["cache_read"]
    if tok_in or tok_out or tok_r:
        tok_html = (
            f'<span class="ti">&#8593;{_num(tok_in)}</span> '
            f'<span class="to">&#8595;{_num(tok_out)}</span>'
        )
        if tok_r:
            tok_html += f' <span class="tr">R{_num(tok_r)}</span>'
    else:
        tok_html = "—"

    rows = [
        ("Date", html.escape(ts)),
        ("Models", models_str),
        ("Messages", html.escape(msgs_str)),
        ("Tool Calls", str(stats["tool_calls"]) if stats["tool_calls"] else "—"),
        ("Tokens", tok_html),
    ]
    inner = "".join(
        f'<span class="mk">{html.escape(k)}:</span><span class="mv">{v}</span>'
        for k, v in rows
    )
    return f'<div class="meta">{inner}</div>'


def _render_system_prompt_card(system_prompt: str) -> str:
    if not system_prompt:
        return ""
    lines = system_prompt.splitlines()
    visible_lines = 8
    more_count = max(0, len(lines) - visible_lines)
    escaped = html.escape(system_prompt)
    collapsed_cls = " sys-collapsed" if more_count else ""
    btn_html = ""
    if more_count:
        more_label = f"... ({more_count} more lines, click to expand)"
        btn_html = (
            f'<button class="sys-btn" onclick="toggleSys(this)"'
            f' data-more="{html.escape(more_label)}">'
            f'{html.escape(more_label)}</button>'
        )
    return (
        f'<div class="card">'
        f'<div class="ctitle">System Prompt</div>'
        f'<pre class="sys-body{collapsed_cls}" id="sys-body">{escaped}</pre>'
        f'{btn_html}'
        f'</div>'
    )


def _render_tools_card(tools: list[dict]) -> str:
    if not tools:
        return ""
    rows = []
    for t in tools:
        name = html.escape(t.get("name", ""))
        desc = html.escape(t.get("description", ""))
        params = t.get("parameters")
        if params:
            params_html = f'<pre class="tparams">{html.escape(json.dumps(params, indent=2))}</pre>'
            rows.append(
                f'<details class="tool-row">'
                f'<summary><span class="tn">{name}</span>'
                f' <span class="td">–</span> <span class="td">{desc}</span>'
                f' <span class="tp">[click to show parameters]</span></summary>'
                f'{params_html}'
                f'</details>'
            )
        else:
            rows.append(
                f'<div class="tool-row">'
                f'<span class="tn">{name}</span>'
                f' <span class="td">–</span> <span class="td">{desc}</span>'
                f'</div>'
            )
    return (
        f'<div class="card">'
        f'<div class="ctitle">Available Tools</div>'
        + "".join(rows) +
        f'</div>'
    )


# ── Message rendering ─────────────────────────────────────────────────────────

def _collect_tool_results(sm: SessionManager) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for entry in sm._entries:
        if not isinstance(entry, SessionMessageEntry):
            continue
        msg = entry.message
        if msg is None:
            continue
        if _attr(msg, "role", "") != "toolResult":
            continue
        tc_id = _attr(msg, "tool_call_id", "") or _attr(msg, "toolCallId", "") or ""
        is_err = bool(_attr(msg, "is_error", False) or _attr(msg, "isError", False))
        raw = _attr(msg, "content", []) or []
        texts: list[str] = []
        if isinstance(raw, list):
            for c in raw:
                t = _attr(c, "text", "")
                if t:
                    texts.append(t)
        elif isinstance(raw, str):
            texts = [raw]
        results[tc_id] = {"text": "\n".join(texts), "is_error": is_err}
    return results


def _render_tool_block(item, tool_results: dict, renderer: ToolHtmlRenderer) -> str:
    tid = _attr(item, "id", "") or ""
    tname = _attr(item, "name", "") or "?"
    args = _attr(item, "arguments", None) or _attr(item, "input", None) or {}

    call_html = None
    if renderer:
        call_html = renderer.render_call(tid, tname, args if isinstance(args, dict) else {})
    if call_html is None:
        try:
            astr = json.dumps(args, default=str) if isinstance(args, dict) else str(args)
        except Exception:
            astr = str(args)
        call_html = (
            f'<code><span class="tn">{html.escape(tname)}</span>'
            f'({html.escape(astr[:160])})</code>'
        )

    res = tool_results.get(tid, {})
    res_text = res.get("text", "")
    is_err = res.get("is_error", False)
    err_cls = " err" if is_err else ""

    res_html = None
    if renderer:
        rd = renderer.render_result(tid, tname, res_text, None, is_err)
        if rd:
            res_html = rd.get("expanded", "")
    if res_html is None:
        res_html = f'<pre class="tr{err_cls}">{html.escape(res_text)}</pre>'

    return (
        f'<details class="tblock">'
        f'<summary><span class="ticon">&#9654;</span>{call_html}</summary>'
        f'<div class="trwrap">{res_html}</div>'
        f'</details>'
    )


def _render_message(
    msg,
    tool_results: dict,
    renderer: ToolHtmlRenderer,
    ts: str = "",
    entry_id: str = "",
) -> str:
    role = _attr(msg, "role", "user")
    content = _attr(msg, "content", []) or []
    msg_cls = "msg-user" if role == "user" else "msg-asst"
    role_css = "ru" if role == "user" else "ra"
    label = "You" if role == "user" else "Assistant"
    ts_html = f'<span class="mts">{html.escape(ts)}</span>' if ts else ""

    # Per-message action buttons (copy text + copy anchor link)
    anchor_id = f"msg-{entry_id}" if entry_id else ""
    actions_html = ""
    if anchor_id:
        actions_html = (
            f'<span class="mactions">'
            f'<button class="mact" title="Copy message text"'
            f' data-msgid="{anchor_id}" onclick="copyMsg(this)">&#x1F4CB;</button>'
            f'<button class="mact" title="Copy shareable link"'
            f' data-msgid="{anchor_id}" onclick="copyLink(this)">&#x1F517;</button>'
            f'</span>'
        )

    # ── Build content segments ────────────────────────────────────────────────
    # Group adjacent blocks into alternating "resp" (text/thinking) and "tool"
    # runs so we can wrap each tool run in a visually distinct section.
    # This preserves interleaved order: text → tools → more text renders as
    #   [response]  [tool section]  [response]
    # User messages never have tool calls but the same logic applies cleanly.

    # Each segment: {"kind": "resp"|"tool", "items": [...raw content items]}
    segments: list[dict] = []

    raw_items: list = []
    if isinstance(content, str):
        raw_items = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        raw_items = list(content)

    for item in raw_items:
        itype = _attr(item, "type", "")
        kind = "tool" if itype in ("toolCall", "tool_use") else "resp"
        if segments and segments[-1]["kind"] == kind:
            segments[-1]["items"].append(item)
        else:
            segments.append({"kind": kind, "items": [item]})

    # ── Render each segment ───────────────────────────────────────────────────
    blocks: list[str] = []
    for seg in segments:
        if seg["kind"] == "resp":
            resp_parts: list[str] = []
            for item in seg["items"]:
                itype = _attr(item, "type", "")
                if itype == "text":
                    text = _attr(item, "text", "") or ""
                    if text:
                        resp_parts.append(f'<pre class="mtext">{html.escape(text)}</pre>')
                elif itype == "thinking":
                    thinking = _attr(item, "thinking", "") or ""
                    preview = thinking[:60].replace("\n", " ")
                    title = (
                        f"\U0001f4ad Thinking: {html.escape(preview)}\u2026"
                        if preview
                        else "\U0001f4ad Thinking\u2026"
                    )
                    resp_parts.append(
                        f'<details class="thblock">'
                        f'<summary>{title}</summary>'
                        f'<div class="thtext">{html.escape(thinking)}</div>'
                        f'</details>'
                    )
            if resp_parts:
                blocks.append(
                    f'<div class="resp-section">{"".join(resp_parts)}</div>'
                )
        else:  # tool run
            tool_parts = [
                _render_tool_block(item, tool_results, renderer)
                for item in seg["items"]
            ]
            n = len(tool_parts)
            hdr = "🔧 Tool call" if n == 1 else f"🔧 Tool calls ({n})"
            blocks.append(
                f'<div class="tools-section">'
                f'<div class="tools-section-hdr">{hdr}</div>'
                + "".join(tool_parts)
                + f'</div>'
            )

    body = "".join(blocks)
    id_attr = f' id="{anchor_id}"' if anchor_id else ""
    return (
        f'<div class="msg {msg_cls}"{id_attr}>'
        f'<div class="mhead">'
        f'<span class="mrole {role_css}">{html.escape(label)}</span>'
        f'{ts_html}'
        f'{actions_html}'
        f'</div>'
        f'<div class="mbody">{body}</div>'
        f'</div>'
    )


def _sidebar_preview(msg) -> tuple[str, bool]:
    """Return (preview_text, has_tools) for a message."""
    role = _attr(msg, "role", "user")
    content = _attr(msg, "content", []) or []
    has_tools = False
    texts: list[str] = []
    tool_names: list[str] = []

    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            itype = _attr(item, "type", "")
            if itype == "text":
                t = _attr(item, "text", "") or ""
                if t:
                    texts.append(t)
            elif itype in ("toolCall", "tool_use"):
                has_tools = True
                tname = _attr(item, "name", "") or "tool"
                tool_names.append(tname)

    if texts:
        raw = texts[0].strip().replace("\n", " ")
        return raw[:80], has_tools
    if tool_names:
        return "[" + ", ".join(tool_names[:3]) + "]", True
    return "(empty)", has_tools


def _render_messages_and_sidebar(
    sm: SessionManager,
    renderer: ToolHtmlRenderer,
    labeled_ids: set[str],
) -> tuple[str, str]:
    """
    Single pass through entries, returns (sidebar_html, messages_html).
    """
    tool_results = _collect_tool_results(sm)
    msg_parts: list[str] = ['<div class="msgs">']
    sb_parts: list[str] = []

    for entry in sm._entries:
        if isinstance(entry, SessionMessageEntry) and entry.message is not None:
            msg = entry.message
            role = _attr(msg, "role", "")
            if role == "toolResult":
                continue

            entry_id = getattr(entry, "id", "") or ""
            anchor_id = f"msg-{entry_id}" if entry_id else ""
            is_labeled = entry_id in labeled_ids

            ts = _fmt_ts_short(getattr(entry, "timestamp", "") or "")
            msg_parts.append(
                _render_message(msg, tool_results, renderer, ts, entry_id)
            )

            # Sidebar item
            preview, has_tools = _sidebar_preview(msg)
            sb_role_attr = "user" if role == "user" else "assistant"
            sb_cls = "sb-user" if role == "user" else "sb-asst"
            badge_cls = "sb-badge-user" if role == "user" else "sb-badge-asst"
            if role == "user":
                badge_label = "user"
            elif has_tools:
                # Count tool calls for the badge
                _tc = sum(
                    1 for _i in (_attr(msg, "content", []) or [])
                    if _attr(_i, "type", "") in ("toolCall", "tool_use")
                )
                badge_label = f"asst+{_tc}🔧"
            else:
                badge_label = "asst"
            labeled_attr = "1" if is_labeled else "0"
            hastools_attr = "1" if has_tools else "0"
            click_attr = f'onclick="scrollToMsg(\'{anchor_id}\')"' if anchor_id else ""
            sb_parts.append(
                f'<div class="sb-item {sb_cls}" data-role="{sb_role_attr}"'
                f' data-labeled="{labeled_attr}" data-hastools="{hastools_attr}"'
                f' {click_attr}>'
                f'<span class="sb-badge {badge_cls}">{badge_label}</span>'
                f'<span class="sb-preview">{html.escape(preview)}</span>'
                f'</div>'
            )

        elif isinstance(entry, ModelChangeEntry):
            label = (
                f"{entry.provider}/{entry.model_id}" if entry.provider else entry.model_id
            )
            msg_parts.append(
                f'<div class="ev">Switched to model: <span>{html.escape(label)}</span></div>'
            )
            sb_parts.append(
                f'<div class="sb-item sb-ev" data-role="ev" data-labeled="0" data-hastools="0">'
                f'<span class="sb-badge sb-badge-ev">ev</span>'
                f'<span class="sb-preview">model: {html.escape(label)}</span>'
                f'</div>'
            )

        elif isinstance(entry, ThinkingLevelChangeEntry):
            msg_parts.append(
                f'<div class="ev">Thinking level: <span>{html.escape(entry.thinking_level)}</span></div>'
            )
            sb_parts.append(
                f'<div class="sb-item sb-ev" data-role="ev" data-labeled="0" data-hastools="0">'
                f'<span class="sb-badge sb-badge-ev">ev</span>'
                f'<span class="sb-preview">thinking: {html.escape(entry.thinking_level)}</span>'
                f'</div>'
            )

        elif isinstance(entry, CompactionEntry):
            short = entry.summary[:300] + ("…" if len(entry.summary) > 300 else "")
            msg_parts.append(
                f'<div class="compact-bar">&#x25C6; Conversation compacted — {html.escape(short)}</div>'
            )
            sb_parts.append(
                f'<div class="sb-item sb-ev" data-role="ev" data-labeled="0" data-hastools="0">'
                f'<span class="sb-badge sb-badge-ev">ev</span>'
                f'<span class="sb-preview">compacted</span>'
                f'</div>'
            )

        elif isinstance(entry, BranchSummaryEntry):
            short = entry.summary[:300] + ("…" if len(entry.summary) > 300 else "")
            msg_parts.append(
                f'<div class="compact-bar">&#x25C6; Branch summary — {html.escape(short)}</div>'
            )
            sb_parts.append(
                f'<div class="sb-item sb-ev" data-role="ev" data-labeled="0" data-hastools="0">'
                f'<span class="sb-badge sb-badge-ev">ev</span>'
                f'<span class="sb-preview">branch summary</span>'
                f'</div>'
            )

        elif isinstance(entry, CustomMessageEntry) and entry.display:
            if entry.custom_type == "skills_loaded":
                details = entry.details if isinstance(entry.details, list) else []
                count = len(details)
                rows = ""
                for sk in details:
                    name = html.escape(str(sk.get("name", "")))
                    desc = html.escape(str(sk.get("description", "")))
                    path = html.escape(str(sk.get("path", "")))
                    rows += (
                        f'<div class="skill-row">'
                        f'<span class="skill-name">/{name}</span>'
                        f'<span class="skill-desc">{desc}</span>'
                        f'<span class="skill-path">{path}</span>'
                        f'</div>'
                    )
                label = "⚡ Skills loaded" if count else "⚡ Skills loaded (none)"
                if count:
                    label = f"⚡ Skills loaded ({count})"
                msg_parts.append(
                    f'<div class="skills-loaded-card">'
                    f'<div class="skills-loaded-title">{label}</div>'
                    f'{rows}'
                    f'</div>'
                )
                sb_parts.append(
                    f'<div class="sb-item sb-ev" data-role="ev"'
                    f' data-labeled="0" data-hastools="0">'
                    f'<span class="sb-badge sb-badge-ev">ev</span>'
                    f'<span class="sb-preview">skills ({count})</span>'
                    f'</div>'
                )
            else:
                content_str = (
                    entry.content
                    if isinstance(entry.content, str)
                    else json.dumps(entry.content)
                )
                msg_parts.append(
                    f'<div class="ev"><span>[system]</span> {html.escape(content_str[:300])}</div>'
                )

    msg_parts.append("</div>")
    return "\n".join(sb_parts), "\n".join(msg_parts)


# ── JSONL embed ───────────────────────────────────────────────────────────────

def _build_jsonl_b64(sm: SessionManager) -> str:
    lines: list[str] = [json.dumps(_header_to_dict(sm._header))]
    for entry in sm._entries:
        try:
            lines.append(json.dumps(_entry_to_dict(entry)))
        except Exception:
            pass
    raw = "\n".join(lines).encode("utf-8")
    return base64.b64encode(raw).decode()


# ── Public API ────────────────────────────────────────────────────────────────

async def export_session_to_html(
    session_manager: SessionManager,
    output_path: str | None = None,
    theme_name: str | None = None,
    tool_renderer: ToolHtmlRenderer | None = None,
    system_prompt: str | None = None,
    tool_definitions: list[dict] | None = None,
) -> str:
    """Export the session to a rich self-contained HTML file. Returns output path."""
    renderer = tool_renderer or ToolHtmlRenderer()
    stats = _extract_stats(session_manager)
    labeled_ids = _collect_labeled_ids(session_manager)

    # Resolve tools
    tools: list[dict] = tool_definitions or []
    if not tools and system_prompt:
        tools = _parse_tools_from_prompt(system_prompt)

    # Build main content sections
    sidebar_html, messages_html = _render_messages_and_sidebar(
        session_manager, renderer, labeled_ids
    )

    sections = [
        _render_header(session_manager),
        _render_metadata(session_manager, stats),
        _render_system_prompt_card(system_prompt or ""),
        _render_tools_card(tools),
        messages_html,
    ]
    content = "\n".join(s for s in sections if s)

    sid = session_manager.get_session_id()
    final_html = (
        _TEMPLATE
        .replace("{{SID}}", html.escape(sid))
        .replace("{{CSS}}", _CSS)
        .replace("{{JS}}", _JS)
        .replace("{{SIDEBAR}}", sidebar_html)
        .replace("{{CONTENT}}", content)
        .replace("{{JDATA}}", _build_jsonl_b64(session_manager))
    )

    if output_path is None:
        output_path = f"session-{sid[:8]}.html"

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(final_html)

    return output_path


async def export_from_file(input_path: str, output_path: str | None = None) -> str:
    """Open a JSONL session file and export it to HTML."""
    sm = SessionManager.open(input_path)
    return await export_session_to_html(sm, output_path)
