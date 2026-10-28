"""L9 static HTML viewer: one self-contained Go-broadcast board page."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .commentary import turn_values

_YT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MARKS = ("!", "?", "!?")


def _dataset_credit(item_id):
    if not isinstance(item_id, str) or not item_id.startswith("casino-"):
        return None
    try:
        p = Path(__file__).with_name("references.json")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        ds = data.get("datasets", []) or []
        if ds and isinstance(ds[0], dict):
            return dict(ds[0])
    except Exception:
        return None
    return None


def _text_and_chunk(t):
    if isinstance(t, dict):
        text = t.get("text", "")
        chunk = t.get("chunk", 0)
    else:
        text = getattr(t, "text", "")
        chunk = getattr(t, "chunk", 0)
    if not isinstance(text, str):
        text = str(text)
    try:
        chunk = int(chunk)
    except (TypeError, ValueError):
        chunk = 0
    return text, chunk


def _compute_t_list(turns, n_view, youtube_id):
    if not youtube_id or not turns:
        return [None] * n_view
    try:
        lst = list(turns)
    except TypeError:
        return [None] * n_view
    if not lst:
        return [None] * n_view
    texts: list[str] = []
    chunks: list[int] = []
    for t in lst:
        text, chunk = _text_and_chunk(t)
        texts.append(text)
        chunks.append(chunk)
    totals: dict[int, int] = {}
    for text, chunk in zip(texts, chunks):
        totals[chunk] = totals.get(chunk, 0) + len(text)
    t_vals: list[int] = []
    for i, (text, chunk) in enumerate(zip(texts, chunks)):
        total = totals.get(chunk, 0)
        if total <= 0:
            off = 0
        else:
            earlier = sum(len(texts[j]) for j in range(i) if chunks[j] == chunk)
            off = int(540 * earlier / total)
        t_vals.append(chunk * 540 + off)
    out: list[int | None] = []
    for pos in range(n_view):
        out.append(t_vals[pos] if pos < len(t_vals) else None)
    return out


def _triplet(value):
    if value is None:
        return None
    try:
        lst = list(value)
    except TypeError:
        return None
    if len(lst) != 3:
        return None
    try:
        return [float(lst[0]), float(lst[1]), float(lst[2])]
    except (TypeError, ValueError):
        return None


def _validity_subset(outcome_stats):
    if outcome_stats is None or not isinstance(outcome_stats, dict):
        return None
    c1 = _triplet(outcome_stats.get("C1"))
    c2 = _triplet(outcome_stats.get("C2"))
    if c1 is None and c2 is None:
        return None
    out: dict = {}
    if c1 is not None:
        out["C1"] = c1
    if c2 is not None:
        out["C2"] = c2
    n = outcome_stats.get("n_dialogues", outcome_stats.get("n", None))
    try:
        n_int = int(n) if n is not None else None
    except (TypeError, ValueError):
        n_int = None
    if n_int is not None:
        out["n"] = n_int
        out["n_dialogues"] = n_int
    try:
        npar = outcome_stats.get("n_participants")
        if npar is not None:
            out["n_participants"] = int(npar)
    except (TypeError, ValueError):
        pass
    return out


def _final_of(curves, side) -> float:
    try:
        arr = curves.get(side, []) if isinstance(curves, dict) else []
        if not arr:
            return 0.0
        return float(arr[-1])
    except (TypeError, ValueError):
        return 0.0


def build_view(item, commentary, outcomes=None, outcome_stats=None,
               youtube_id=None, turns=None, stakes=None) -> dict:
    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    item_id = item.get("id", "")
    if isinstance(item_id, str) and item_id:
        title = "Dialogue " + item_id + " \u2014 \ud615\uc138 board"
    elif item_id:
        title = "Dialogue " + str(item_id) + " \u2014 \ud615\uc138 board"
    else:
        title = "Dialogue \u2014 \ud615\uc138 board"
    norm = commentary.get("norm", {})
    tv = turn_values(item, norm)
    curves = commentary.get("curves")
    if not isinstance(curves, dict):
        curves = {"A": [], "B": [], "balance": []}
    pivots = commentary.get("turns", [])
    if not isinstance(pivots, list):
        pivots = []
    mark_by_index: dict[int, str] = {}
    for p in pivots:
        if not isinstance(p, dict):
            continue
        idx = p.get("index")
        mark = p.get("mark")
        if isinstance(idx, int) and mark in _MARKS and idx not in mark_by_index:
            mark_by_index[idx] = mark
    yt = youtube_id.strip() if isinstance(youtube_id, str) else youtube_id
    yt_out = yt if isinstance(yt, str) and yt else None
    if yt_out is None and isinstance(youtube_id, str) and not youtube_id.strip():
        yt_out = None
    t_list = _compute_t_list(turns, len(tv), yt_out)
    turns_out: list[dict] = []
    for pos, e in enumerate(tv):
        idx = e.get("index", pos)
        turns_out.append({
            "index": idx,
            "speaker": e.get("speaker", ""),
            "text": e.get("text", ""),
            "moves": list(e.get("moves", []) or []),
            "value": float(e.get("value", 0.0)),
            "z": float(e.get("z", 0.0)),
            "mark": mark_by_index.get(idx),
            "t": t_list[pos] if pos < len(t_list) else None,
        })
        if turns_out[-1]["mark"] is None:
            turns_out[-1]["mark"] = None
    global_counts = {"!": 0, "?": 0, "!?": 0}
    side_counts = {"A": {"!": 0, "?": 0, "!?": 0},
                   "B": {"!": 0, "?": 0, "!?": 0}}
    for p in pivots:
        if not isinstance(p, dict):
            continue
        mark = p.get("mark")
        if mark not in _MARKS:
            continue
        global_counts[mark] += 1
        speaker = p.get("speaker")
        if speaker in ("A", "B"):
            side_counts[speaker][mark] += 1
    n_a = sum(1 for e in tv if e.get("speaker") == "A")
    n_b = sum(1 for e in tv if e.get("speaker") == "B")
    final_a = _final_of(curves, "A")
    final_b = _final_of(curves, "B")

    def _side(side, final, n_turns):
        counts = dict(side_counts[side])
        return {
            "final": final,
            "value": final,
            "counts": counts,
            "!": counts["!"],
            "?": counts["?"],
            "!?": counts["!?"],
            "n_turns": n_turns,
            "n": n_turns,
            "turns": n_turns,
        }

    score = {
        "A": _side("A", final_a, n_a),
        "B": _side("B", final_b, n_b),
        "counts": dict(global_counts),
        "n_turns": len(tv),
        "n": len(tv),
    }
    validity = _validity_subset(outcome_stats)
    dataset = _dataset_credit(item_id)
    stakes_out = None
    if isinstance(stakes, dict):
        rec = stakes.get(item_id)
        if isinstance(rec, dict):
            track = rec.get("track", [])
            offers = rec.get("offers", {})
            stakes_out = {
                "track": track if isinstance(track, list) else [],
                "offers": offers if isinstance(offers, dict) else {},
            }
    return {
        "id": item_id,
        "title": title,
        "turns": turns_out,
        "curves": curves,
        "pivots": pivots,
        "score": score,
        "outcomes": outcomes if outcomes is not None else None,
        "validity": validity,
        "youtube_id": yt_out,
        "dataset": dataset,
        "stakes": stakes_out,
    }


_CSS = """
:root{
  --bg:#f4f6f8; --fg:#1f2328; --card:#ffffff; --muted:#5b6470; --border:#d8dee6;
  --line-a:#1f6feb; --line-b:#0a7d62; --line-bal:#75808c; --zero:#c2c9d2;
  --good:#1a7f37; --good-bg:#e5f3e9; --bad:#c92a33; --bad-bg:#fbe9ea;
  --mid:#96690a; --mid-bg:#fdf3d7; --accent:#1f6feb;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d1117; --fg:#e6edf3; --card:#161b22; --muted:#9aa4b2; --border:#30363d;
    --line-a:#7fb3ff; --line-b:#56d4a0; --line-bal:#9aa4b2; --zero:#4a525c;
    --good:#3fb950; --good-bg:#122619; --bad:#f0606a; --bad-bg:#2c1518;
    --mid:#d29922; --mid-bg:#2b230a; --accent:#7fb3ff;
  }
}
*{box-sizing:border-box}
body{max-width:1100px;margin:0 auto;padding:0 16px;background:var(--bg);color:var(--fg);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;line-height:1.55}
header{padding:20px 0 8px}
h1{font-size:22px;margin:0 0 4px;overflow-wrap:break-word}
h2{font-size:16px;margin:0 0 10px}
.credit{color:var(--muted);font-size:13px;margin:0 0 2px;overflow-wrap:break-word}
.note{color:var(--muted);font-size:13px;margin:0 0 10px}
.lang{display:flex;gap:8px;margin:6px 0 4px}
.lang button{border:1px solid var(--border);background:var(--card);color:var(--fg);
  border-radius:8px;padding:4px 12px;font-size:13px;cursor:pointer}
.lang button.active{background:var(--accent);border-color:var(--accent);color:#fff}
@media (prefers-color-scheme: dark){.lang button.active{color:#0d1117}}
.layout{display:flex;flex-direction:column;gap:14px;padding-bottom:28px}
@media (min-width:900px){
  .layout{flex-direction:row;align-items:flex-start}
  .col-main{flex:1 1 60%;min-width:0}
  .col-side{flex:1 1 40%;min-width:0}
}
.col-main,.col-side{display:flex;flex-direction:column;gap:14px;min-width:0}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px;min-width:0;overflow:hidden}
#chart{width:100%;height:auto;display:block;max-width:100%}
#stakes-chart{width:100%;height:auto;display:block;max-width:100%}
.line-a{fill:none;stroke:var(--line-a);stroke-width:2}
.line-b{fill:none;stroke:var(--line-b);stroke-width:2}
.line-bal{fill:none;stroke:var(--line-bal);stroke-width:1.5;stroke-dasharray:4 3}
.zero{stroke:var(--zero);stroke-width:1}
.axis-tx{fill:var(--muted);font-size:10px}
.marker{cursor:pointer}
.marker-bg{fill:var(--card);stroke-width:2}
.marker-bg[data-mark="!"]{stroke:var(--good)}
.marker-bg[data-mark="?"]{stroke:var(--bad)}
.marker-bg[data-mark="!?"]{stroke:var(--mid)}
.marker-tx{font-size:10px;font-weight:700;fill:var(--fg)}
.legend{display:flex;gap:12px;margin-top:6px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.lg{display:inline-flex;align-items:center;gap:5px}
.sw{width:18px;height:0;border-top:2px solid var(--line-a);display:inline-block}
.sw.b{border-color:var(--line-b)}
.sw.bal{border-color:var(--line-bal);border-top-style:dashed}
.video{position:relative;width:100%;aspect-ratio:16/9}
.video iframe{position:absolute;inset:0;width:100%;height:100%;border:0;border-radius:8px}
#turns{display:flex;flex-direction:column;gap:10px}
.turn{border:1px solid var(--border);border-radius:10px;padding:10px;min-width:0}
.turn.clickable{cursor:pointer}
.turn.flash{background:var(--mid-bg);outline:2px solid var(--mid)}
.turn-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.turn-idx{font-weight:700}
.turn-sp{color:var(--muted)}
.turn-t{color:var(--muted);margin-left:auto}
.turn-text{margin:6px 0;overflow-wrap:break-word;word-break:break-word}
.moves{display:flex;flex-wrap:wrap;gap:4px}
.chip{display:inline-block;background:var(--bg);border:1px solid var(--border);
  border-radius:999px;padding:0 8px;font-size:12px;color:var(--fg)}
.badge{display:inline-block;min-width:1.7em;text-align:center;border-radius:6px;
  padding:0 6px;font-weight:700;font-size:13px;border:1px solid var(--border)}
.badge[data-mark="!"]{background:var(--good-bg);color:var(--good);border-color:var(--good)}
.badge[data-mark="?"]{background:var(--bad-bg);color:var(--bad);border-color:var(--bad)}
.badge[data-mark="!?"]{background:var(--mid-bg);color:var(--mid);border-color:var(--mid)}
.value-wrap{display:flex;align-items:center;gap:8px;margin-top:6px}
.value-bar{position:relative;height:8px;width:140px;max-width:45%;
  background:var(--bg);border:1px solid var(--border);border-radius:5px;overflow:hidden;flex:none}
.value-fill{position:absolute;top:0;bottom:0;background:var(--accent)}
.value-zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--muted)}
.value-num{font-size:12px;color:var(--muted)}
#pivots{display:flex;flex-direction:column;gap:10px}
.pivot{border:1px solid var(--border);border-radius:10px;padding:10px;cursor:pointer;min-width:0}
.pivot-top{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pivot-label{font-weight:700}
.pivot-idx{color:var(--muted);font-size:12px;margin-left:auto}
.pivot-line{margin:6px 0;overflow-wrap:break-word;word-break:break-word}
.pivot-ref{margin:0;color:var(--muted);font-size:12px;overflow-wrap:break-word}
#score{display:flex;gap:10px;flex-wrap:wrap}
.score-side{flex:1 1 140px;border:1px solid var(--border);border-radius:10px;padding:8px 10px;min-width:0}
.score-side h3{margin:0 0 4px;font-size:14px}
.score-side p{margin:2px 0;font-size:13px}
.out-note{color:var(--muted);font-size:12px;margin:10px 0 2px}
.out-line{margin:2px 0;font-size:13px;overflow-wrap:break-word}
.validity{font-size:12px;color:var(--muted);margin:10px 0 0;overflow-wrap:break-word}
.muted{color:var(--muted)}
"""

_JS_HEAD = """(function(){
"use strict";
var dataEl=document.getElementById("data");
var view=JSON.parse(dataEl.textContent);
var lang="ko";
function el(tag,cls,text){var e=document.createElement(tag);if(cls)e.className=cls;if(text!==null&&text!==undefined)e.textContent=text;return e;}
function fmtT(t){if(t===null||t===undefined)return "";t=Math.max(0,Math.floor(t));var m=Math.floor(t/60);var s=t%60;return "≈"+m+":"+(s<10?"0":"")+s;}
"""

_JS_TAIL = """
function renderHeader(){
  var t=document.getElementById("title");
  t.textContent=view.title||view.id||"";
  try{document.title=view.title||"Uptake viewer";}catch(e){}
  var cr=document.getElementById("credit");
  while(cr.firstChild)cr.removeChild(cr.firstChild);
  if(view.dataset){
    var ds=view.dataset;
    var tx=(ds.author||"")+" ("+(ds.year!==null&&ds.year!==undefined?ds.year:"")+")"+", "+(ds.title||"");
    if(ds.venue)tx+=" — "+ds.venue;
    cr.textContent=tx;
  }
}
function renderChart(){
  var svg=document.getElementById("chart");
  var NS=svg.namespaceURI;
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  var curves=view.curves||{};
  var A=curves.A||[],B=curves.B||[],Bal=curves.balance||[];
  var turns=view.turns||[];
  var n=Math.max(A.length,B.length,Bal.length,turns.length);
  if(!n)return;
  var all=[0];
  var k,i;
  for(i=0;i<A.length;i++)all.push(A[i]);
  for(i=0;i<B.length;i++)all.push(B[i]);
  for(i=0;i<Bal.length;i++)all.push(Bal[i]);
  var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all);
  if(mx===mn)mx=mn+1;
  var W=600,H=220,padL=38,padR=14,padT=14,padB=24;
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  svg.setAttribute("role","img");
  svg.setAttribute("aria-label","position curve");
  function X(i){if(n<=1)return padL+(W-padL-padR)/2;return padL+(i/(n-1))*(W-padL-padR);}
  function Y(v){return padT+(1-(v-mn)/(mx-mn))*(H-padT-padB);}
  var zl=document.createElementNS(NS,"line");
  zl.setAttribute("x1",padL);zl.setAttribute("x2",W-padR);
  zl.setAttribute("y1",Y(0));zl.setAttribute("y2",Y(0));
  zl.setAttribute("class","zero");
  svg.appendChild(zl);
  var yticks=[mn,0,mx];
  for(var yi=0;yi<yticks.length;yi++){
    var ylab=document.createElementNS(NS,"text");
    ylab.setAttribute("x",padL-4);ylab.setAttribute("y",Y(yticks[yi])+3);
    ylab.setAttribute("text-anchor","end");
    ylab.setAttribute("class","axis-tx");
    ylab.textContent=Number(yticks[yi]).toFixed(1);
    svg.appendChild(ylab);
  }
  function draw(arr,cls){
    if(!arr||!arr.length)return;
    var pts=[];
    for(var j=0;j<arr.length;j++)pts.push(X(j).toFixed(1)+","+Y(arr[j]).toFixed(1));
    var pl=document.createElementNS(NS,"polyline");
    pl.setAttribute("points",pts.join(" "));
    pl.setAttribute("class",cls);
    svg.appendChild(pl);
  }
  draw(A,"line-a");draw(B,"line-b");draw(Bal,"line-bal");
  var step=Math.max(1,Math.floor(n/6));
  for(var t=0;t<n;t+=step){
    var lab=document.createElementNS(NS,"text");
    lab.setAttribute("x",X(t));lab.setAttribute("y",H-6);
    lab.setAttribute("text-anchor","middle");
    lab.setAttribute("class","axis-tx");
    var idx=(t<turns.length&&turns[t].index!==undefined)?turns[t].index:t;
    lab.textContent=""+idx;
    svg.appendChild(lab);
  }
  var posByIndex={};
  for(var p=0;p<turns.length;p++)posByIndex[turns[p].index]=p;
  var pivs=view.pivots||[];
  for(k=0;k<pivs.length;k++){
    (function(pv){
      var pos=posByIndex[pv.index];
      if(pos===null||pos===undefined)return;
      var yv=0;
      if(pv.speaker==="A"&&A.length>pos)yv=A[pos];
      else if(pv.speaker==="B"&&B.length>pos)yv=B[pos];
      else if(Bal.length>pos)yv=Bal[pos];
      var cx=X(pos),cy=Y(yv);
      var g=document.createElementNS(NS,"g");
      g.setAttribute("class","marker");
      var c=document.createElementNS(NS,"circle");
      c.setAttribute("cx",cx);c.setAttribute("cy",cy);c.setAttribute("r","10");
      c.setAttribute("class","marker-bg");
      if(pv.mark)c.setAttribute("data-mark",pv.mark);
      g.appendChild(c);
      var tx=document.createElementNS(NS,"text");
      tx.setAttribute("x",cx);tx.setAttribute("y",cy+4);
      tx.setAttribute("text-anchor","middle");
      tx.setAttribute("class","marker-tx");
      tx.textContent=pv.mark||"";
      g.appendChild(tx);
      g.addEventListener("click",function(){
        var tgt=document.getElementById("turn-"+pv.index);
        if(tgt){tgt.scrollIntoView({block:"center"});tgt.classList.add("flash");
          setTimeout(function(){tgt.classList.remove("flash");},1600);}
      });
      svg.appendChild(g);
    })(pivs[k]);
  }
}
function renderStakes(){
  var svg=document.getElementById("stakes-chart");
  if(!svg)return;
  var st=view.stakes;
  var card=document.getElementById("stakes-card");
  if(!st||!st.track||!st.track.length){
    if(card)card.style.display="none";
    return;
  }
  if(card)card.style.display="";
  var NS=svg.namespaceURI;
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  var curves=view.curves||{};
  var A=curves.A||[],B=curves.B||[],Bal=curves.balance||[];
  var turns=view.turns||[];
  var n=Math.max(A.length,B.length,Bal.length,turns.length,st.track.length);
  if(!n)return;
  var W=600,H=140,padL=38,padR=14,padT=10,padB=20;
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  svg.setAttribute("role","img");
  svg.setAttribute("aria-label","stake curve");
  function X(i){if(n<=1)return padL+(W-padL-padR)/2;return padL+(i/(n-1))*(W-padL-padR);}
  function Y(v){return padT+(1-(v/36))*(H-padT-padB);}
  var sticks=[0,12,24,36];
  for(var ti=0;ti<sticks.length;ti++){
    var gl=document.createElementNS(NS,"line");
    gl.setAttribute("x1",padL);gl.setAttribute("x2",W-padR);
    gl.setAttribute("y1",Y(sticks[ti]));gl.setAttribute("y2",Y(sticks[ti]));
    gl.setAttribute("class","zero");
    svg.appendChild(gl);
    var slab=document.createElementNS(NS,"text");
    slab.setAttribute("x",padL-4);slab.setAttribute("y",Y(sticks[ti])+3);
    slab.setAttribute("text-anchor","end");
    slab.setAttribute("class","axis-tx");
    slab.textContent=""+sticks[ti];
    svg.appendChild(slab);
  }
  function drawSide(key,cls){
    var run=[];
    function flush(){
      if(run.length>1){
        var pl=document.createElementNS(NS,"polyline");
        pl.setAttribute("points",run.join(" "));
        pl.setAttribute("class",cls);
        svg.appendChild(pl);
      }else if(run.length===1){
        var xy=run[0].split(",");
        var c=document.createElementNS(NS,"circle");
        c.setAttribute("cx",xy[0]);c.setAttribute("cy",xy[1]);
        c.setAttribute("r","2.5");
        c.style.fill=(key==="A")?"var(--line-a)":"var(--line-b)";
        svg.appendChild(c);
      }
      run=[];
    }
    for(var i=0;i<st.track.length;i++){
      var e=st.track[i];
      var v=e?e[key]:null;
      if(v===null||v===undefined){flush();}
      else{run.push(X(i).toFixed(1)+","+Y(v).toFixed(1));}
    }
    flush();
  }
  drawSide("A","line-a");drawSide("B","line-b");
}
function renderTurns(){
  var box=document.getElementById("turns");
  while(box.firstChild)box.removeChild(box.firstChild);
  var turns=view.turns||[];
  var maxAbs=0,i;
  for(i=0;i<turns.length;i++){var a=Math.abs(turns[i].value||0);if(a>maxAbs)maxAbs=a;}
  if(maxAbs===0)maxAbs=1;
  for(i=0;i<turns.length;i++){
    (function(turn,pos){
      var d=el("div","turn");
      d.setAttribute("id","turn-"+turn.index);
      var head=el("div","turn-head");
      head.appendChild(el("span","turn-idx","#"+turn.index));
      head.appendChild(el("span","turn-sp",turn.speaker||""));
      if(turn.mark){
        var b=el("span","badge",turn.mark);
        b.setAttribute("data-mark",turn.mark);
        head.appendChild(b);
      }
      if(turn.t!==null&&turn.t!==undefined)head.appendChild(el("span","turn-t",fmtT(turn.t)));
      d.appendChild(head);
      d.appendChild(el("p","turn-text",turn.text||""));
      if(turn.moves&&turn.moves.length){
        var mc=el("div","moves");
        for(var m=0;m<turn.moves.length;m++)mc.appendChild(el("span","chip",turn.moves[m]));
        d.appendChild(mc);
      }
      var vst=view.stakes;
      if(vst&&vst.offers&&vst.offers[turn.index]!==undefined){
        var tri=(vst.track||[])[pos];
        if(!(tri&&tri.A!==null&&tri.A!==undefined&&tri.B!==null&&tri.B!==undefined)){
          var trs=vst.track||[];
          for(var q=0;q<trs.length;q++){
            if(trs[q]&&trs[q].offer_turn===turn.index){tri=trs[q];break;}
          }
        }
        if(tri&&tri.A!==null&&tri.A!==undefined&&tri.B!==null&&tri.B!==undefined){
          var oc=el("div","moves");
          oc.appendChild(el("span","chip","offer A "+tri.A+" · B "+tri.B));
          d.appendChild(oc);
        }
      }
      var wrap=el("div","value-wrap");
      var bar=el("div","value-bar");
      var fill=document.createElement("div");
      fill.className="value-fill";
      var v=turn.value||0;
      var w=Math.abs(v)/maxAbs*50;
      if(v>=0){fill.style.left="50%";fill.style.width=w+"%";}
      else{fill.style.left=(50-w)+"%";fill.style.width=w+"%";}
      if(turn.mark==="!")fill.style.background="var(--good)";
      else if(turn.mark==="?")fill.style.background="var(--bad)";
      else if(turn.mark==="!?")fill.style.background="var(--mid)";
      bar.appendChild(fill);
      var z=document.createElement("div");z.className="value-zero";bar.appendChild(z);
      wrap.appendChild(bar);
      wrap.appendChild(el("span","value-num",(v>=0?"+":"")+Number(v).toFixed(1)));
      d.appendChild(wrap);
      if(turn.t!==null&&turn.t!==undefined&&view.youtube_id){
        d.classList.add("clickable");
        d.addEventListener("click",function(){maybeSeek(turn.t);});
      }
      box.appendChild(d);
    })(turns[i],i);
  }
}
function renderPivots(){
  var box=document.getElementById("pivots");
  while(box.firstChild)box.removeChild(box.firstChild);
  var pivs=view.pivots||[];
  var i;
  for(i=0;i<pivs.length;i++){
    (function(pv){
      var c=el("article","pivot");
      var top=el("div","pivot-top");
      var b=el("span","badge",pv.mark||"");
      if(pv.mark)b.setAttribute("data-mark",pv.mark);
      top.appendChild(b);
      top.appendChild(el("span","pivot-label",lang==="ko"?(pv.label_ko||""):(pv.label_en||"")));
      top.appendChild(el("span","pivot-idx","#"+pv.index+" "+(pv.speaker||"")));
      c.appendChild(top);
      c.appendChild(el("p","pivot-line",lang==="ko"?(pv.line_ko||""):(pv.line_en||"")));
      var ref=pv.reference||{};
      var yr=(ref.year!==null&&ref.year!==undefined)?ref.year:"";
      c.appendChild(el("p","pivot-ref",(ref.author||"")+" ("+yr+"), "+(ref.title||"")));
      c.addEventListener("click",function(){
        var tgt=document.getElementById("turn-"+pv.index);
        if(tgt){tgt.scrollIntoView({block:"center"});tgt.classList.add("flash");
          setTimeout(function(){tgt.classList.remove("flash");},1600);}
      });
      box.appendChild(c);
    })(pivs[i]);
  }
  if(!pivs.length)box.appendChild(el("p","muted",lang==="ko"?"결정적 장면 없음":"No pivotal turns"));
}
function renderScore(){
  var box=document.getElementById("score");
  while(box.firstChild)box.removeChild(box.firstChild);
  var sc=view.score||{};
  var sides=["A","B"];
  for(var i=0;i<sides.length;i++){
    var s=sides[i];
    var info=sc[s]||{};
    var d=el("div","score-side");
    d.appendChild(el("h3","score-name",s));
    var fin=(info.final!==null&&info.final!==undefined)?info.final:((info.value!==null&&info.value!==undefined)?info.value:0);
    d.appendChild(el("p","score-final",(lang==="ko"?"최종 ":"final ")+Number(fin).toFixed(1)));
    var counts=info.counts||{};
    var g=(counts["!"]!==null&&counts["!"]!==undefined)?counts["!"]:((info["!"]!==null&&info["!"]!==undefined)?info["!"]:0);
    var q=(counts["?"]!==null&&counts["?"]!==undefined)?counts["?"]:((info["?"]!==null&&info["?"]!==undefined)?info["?"]:0);
    var m=(counts["!?"]!==null&&counts["!?"]!==undefined)?counts["!?"]:((info["!?"]!==null&&info["!?"]!==undefined)?info["!?"]:0);
    d.appendChild(el("p","score-counts","! "+g+" · ? "+q+" · !? "+m));
    var nt=(info.n_turns!==null&&info.n_turns!==undefined)?info.n_turns:((info.n!==null&&info.n!==undefined)?info.n:((info.turns!==null&&info.turns!==undefined)?info.turns:0));
    d.appendChild(el("p","score-turns",(lang==="ko"?"턴 ":"turns ")+nt));
    box.appendChild(d);
  }
  if(view.stakes&&view.stakes.track){
    var trk=view.stakes.track;
    var endA=null,endB=null;
    for(var s=trk.length-1;s>=0;s--){
      if(trk[s]&&trk[s].A!==null&&trk[s].A!==undefined&&trk[s].B!==null&&trk[s].B!==undefined){
        endA=trk[s].A;endB=trk[s].B;break;
      }
    }
    box.appendChild(el("p","out-line","on-table points at the end: A "+(endA===null?"–":endA)+" / B "+(endB===null?"–":endB)));
  }
  var ocBox=document.getElementById("outcomes");
  while(ocBox.firstChild)ocBox.removeChild(ocBox.firstChild);
  if(view.outcomes){
    ocBox.appendChild(el("p","out-note","reported by participants after the negotiation"));
    var s2;
    for(var j=0;j<sides.length;j++){
      s2=sides[j];
      var r=(view.outcomes||{})[s2];
      if(!r)continue;
      ocBox.appendChild(el("p","out-line",s2+": points "+r.points+", satisfaction "+r.satisfaction+", likeness "+r.likeness));
    }
  }
  var vEl=document.getElementById("validity");
  while(vEl.firstChild)vEl.removeChild(vEl.firstChild);
  if(view.validity){
    var vv=view.validity;
    var n=(vv.n!==null&&vv.n!==undefined)?vv.n:((vv.n_dialogues!==null&&vv.n_dialogues!==undefined)?vv.n_dialogues:"?");
    var c1=vv.C1,c2=vv.C2,txt="";
    if(c1&&c1.length===3){
      txt+="on "+n+" held-out dialogues the curve tracks how much the other side liked the speaker: ρ = "+Number(c1[0]).toFixed(3)+" ["+Number(c1[1]).toFixed(3)+", "+Number(c1[2]).toFixed(3)+"]";
    }
    if(c2&&c2.length===3){
      if(txt)txt+="; ";
      txt+="dyad vs satisfaction: ρ = "+Number(c2[0]).toFixed(3)+" ["+Number(c2[1]).toFixed(3)+", "+Number(c2[2]).toFixed(3)+"]";
    }
    if(txt)vEl.textContent=txt;
  }
}
function applyLang(){
  var nodes=document.querySelectorAll("[data-ko]");
  for(var i=0;i<nodes.length;i++){
    var nd=nodes[i];
    nd.textContent=(lang==="ko")?nd.getAttribute("data-ko"):nd.getAttribute("data-en");
  }
  var bk=document.getElementById("lang-ko"),be=document.getElementById("lang-en");
  if(bk&&be){
    if(lang==="ko"){bk.classList.add("active");be.classList.remove("active");}
    else{be.classList.add("active");bk.classList.remove("active");}
  }
  renderPivots();
  renderScore();
}
document.getElementById("lang-ko").addEventListener("click",function(){lang="ko";applyLang();});
document.getElementById("lang-en").addEventListener("click",function(){lang="en";applyLang();});
renderHeader();
renderChart();
renderStakes();
renderTurns();
applyLang();
})();
"""

_SEEK_NONE = "function maybeSeek(t){}\n"
_SEEK_YT = (
    "function maybeSeek(t){var f=document.getElementById(\"yt\");"
    "if(!f||t===null||t===undefined)return;"
    "var yid=view.youtube_id;"
    "if(typeof yid!==\"string\"||!/^[A-Za-z0-9_-]{1,64}$/.test(yid))return;"
    "f.setAttribute(\"src\",\"https://www.youtube-nocookie.com/embed/\"+yid+\"?start=\"+Math.floor(t)+\"&rel=0\");}\n"
)


def render_html(view) -> str:
    view = view if isinstance(view, dict) else {}
    blob = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("<", "\\u003c")
    yt = view.get("youtube_id")
    yt = yt.strip() if isinstance(yt, str) else None
    yt_valid = yt if yt and _YT_RE.fullmatch(yt) else None
    if yt_valid:
        iframe = (
            '<section class="card"><h2 data-ko="\uc601\uc0c1" data-en="Video">\uc601\uc0c1</h2>'
            '<div class="video"><iframe id="yt" src="https://www.youtube-nocookie.com/embed/'
            + yt_valid
            + '?rel=0" title="Video" loading="lazy" '
            'allow="accelerometer; encrypted-media; picture-in-picture" allowfullscreen>'
            "</iframe></div></section>"
        )
        seek_js = _SEEK_YT
    else:
        iframe = ""
        seek_js = _SEEK_NONE
    js = _JS_HEAD + seek_js + _JS_TAIL
    stakes_section = ""
    if isinstance(view.get("stakes"), dict):
        stakes_section = (
            '<section class="card" id="stakes-card"><h2>판돈: 테이블 위 점수 / '
            'Stake: points on the table</h2>'
            '<svg id="stakes-chart"></svg>'
            '<p class="note">points each side would get if the latest offer '
            'were accepted (CaSiNo point values)</p></section>\n'
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Uptake viewer \u2014 \ud615\uc138 board</title>\n"
        "<style>" + _CSS + "</style>\n</head>\n<body>\n"
        "<header>\n"
        '<h1 id="title"></h1>\n'
        '<p id="credit" class="credit"></p>\n'
        '<p class="note">model reading of moves, not a judgment of people</p>\n'
        '<div class="lang"><button id="lang-ko" type="button">\ud55c\uad6d\uc5b4</button>'
        '<button id="lang-en" type="button">English</button></div>\n'
        "</header>\n"
        '<main class="layout">\n'
        '<div class="col-main">\n'
        '<section class="card"><h2 data-ko="\ud615\uc138" data-en="Position">\ud615\uc138</h2>'
        '<svg id="chart"></svg>'
        '<div class="legend"><span class="lg"><span class="sw"></span>A</span>'
        '<span class="lg"><span class="sw b"></span>B</span>'
        '<span class="lg"><span class="sw bal"></span>A\u2212B</span></div></section>\n'
        + stakes_section
        + iframe + "\n"
        '<section class="card"><h2 data-ko="\uc804\uccb4 \ud134" data-en="All turns">\uc804\uccb4 \ud134</h2>'
        '<div id="turns"></div></section>\n'
        "</div>\n"
        '<div class="col-side">\n'
        '<section class="card"><h2 data-ko="\uacb0\uc815\uc801 \uc7a5\uba74" data-en="Pivotal turns">\uacb0\uc815\uc801 \uc7a5\uba74</h2>'
        '<div id="pivots"></div></section>\n'
        '<section class="card"><h2 data-ko="\uc815\uc0b0" data-en="Score">\uc815\uc0b0</h2>'
        '<div id="score"></div><div id="outcomes"></div>'
        '<p id="validity" class="validity"></p></section>\n'
        "</div>\n</main>\n"
        '<script type="application/json" id="data">'
        + blob
        + "</script>\n<script>\n"
        + js
        + "</script>\n</body>\n</html>\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-viewer")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--casino", default=None)
    ap.add_argument("--outcome", default=None)
    ap.add_argument("--youtube-id", default=None)
    ap.add_argument("--turns", default=None)
    ap.add_argument("--stakes", default=None)
    args = ap.parse_args(argv)

    with open(args.item, encoding="utf-8") as f:
        item = json.load(f)
    with open(args.commentary, encoding="utf-8") as f:
        commentary = json.load(f)
    outcomes = None
    if args.casino:
        from .outcome import dialogue_id_of, load_outcomes

        all_outcomes = load_outcomes(args.casino)
        did = dialogue_id_of(item.get("id", "") if isinstance(item, dict) else "")
        if did is not None and did in all_outcomes:
            outcomes = all_outcomes[did]
    outcome_stats = None
    if args.outcome:
        with open(args.outcome, encoding="utf-8") as f:
            outcome_stats = json.load(f)
    turns_list = None
    if args.turns:
        from .transcribe import load_turns

        turns_list = load_turns(args.turns)
    youtube_id = args.youtube_id if args.youtube_id else None
    stakes = None
    if args.stakes:
        with open(args.stakes, encoding="utf-8") as f:
            stakes = json.load(f)
    view = build_view(
        item,
        commentary,
        outcomes=outcomes,
        outcome_stats=outcome_stats,
        youtube_id=youtube_id,
        turns=turns_list,
        stakes=stakes,
    )
    html = render_html(view)
    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
