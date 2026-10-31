"""Layer-3 "guess the next move" game page.

Standalone page: the tide stays hidden while the player guesses, turn by
turn, whether the speaker's standing went up or down vs their previous
standing; the end card reveals the full simplified tide (task48), the hit
rate, the actual outcome, and a share text. Self-contained HTML, light
theme, mobile-first, Korean UI, stdlib only. All game logic runs in page
JS; no server.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from . import misread
from .board2 import MOVE_KO, decisive_of, outcome_line, tide_svg
from .commentary import mark_for, standing_round
from .viewer2 import build_view2, tide_y

INTRO = ("두 사람이 캠핑 물품을 나누는 협상입니다. 발언을 하나씩 보고, "
         "상대가 이 말을 어떻게 받아들일지 맞혀 보세요.")
INTRO_NOTE = "해설자는 정의상 항상 맞습니다. 해설자와 겨뤄 보세요."

GUESSES = (
    ("오른다", "오른다"),
    ("그대로", "그대로"),
    ("내린다", "내린다"),
)

_DIR_NOUN = {"오른다": "오름", "그대로": "그대로", "내린다": "내림"}

_DELTA_KEY = 6
_SKIP_LO = 3
_SKIP_HI = 6


def reader_verdict(z_abs) -> str:
    """Reader's verdict from absolute reception: >=0.75 잘, <=-0.75 나쁘게.

    z_abs is (value - mu) / sd from the record's scale table (task49b).
    """
    try:
        zf = float(z_abs)
    except (TypeError, ValueError):
        return "보통"
    if not math.isfinite(zf):
        return "보통"
    if zf >= 0.75:
        return "잘"
    if zf <= -0.75:
        return "나쁘게"
    return "보통"


def direction_verdict(delta) -> str:
    """Direction key from a standing change: >=+6 오른다, <=-6 내린다.

    delta is standing_after - standing_before on the 0-100 scale (the same
    integers shown in the score line); the speaker's first turn is vs 50.
    """
    try:
        df = float(delta)
    except (TypeError, ValueError):
        return "그대로"
    if not math.isfinite(df):
        return "그대로"
    if df >= _DELTA_KEY:
        return "오른다"
    if df <= -_DELTA_KEY:
        return "내린다"
    return "그대로"


def _is_ambiguous(delta) -> bool:
    """True when |delta| sits in the open (3, 6) band: too close to call."""
    try:
        df = float(delta)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(df):
        return False
    return _SKIP_LO < abs(df) < _SKIP_HI


def question_for(display) -> str:
    """Per-turn question line naming the speaker."""
    name = display if isinstance(display, str) and display else "화자"
    return f"이 말로 {name}의 관계 점수가 오를까, 그대로일까, 내릴까?"


def _has_jongseong(name) -> bool:
    if not name:
        return False
    code = ord(str(name)[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _p75(vals):
    """Nearest-rank 75th percentile, or None when no values."""
    try:
        nums = sorted(float(v) for v in (vals or []))
    except (TypeError, ValueError):
        return None
    nums = [v for v in nums if math.isfinite(v)]
    if not nums:
        return None
    k = int(math.ceil(0.75 * len(nums))) - 1
    k = max(0, min(k, len(nums) - 1))
    return float(nums[k])


def _disagreements_by_index(item) -> dict:
    out: dict = {}
    effects = item.get("effects", []) if isinstance(item, dict) else []
    for pos, e in enumerate(effects or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) else pos
        if idx in out or "disagreement" not in e:
            continue
        try:
            out[idx] = float(e["disagreement"])
        except (TypeError, ValueError):
            continue
    return out


def _move_ko_first(moves) -> str:
    for m in moves or []:
        if isinstance(m, str) and m:
            return MOVE_KO.get(m, m)
    return "발언"


def _fallback_reason(moves, z_abs) -> str:
    ko = _move_ko_first(moves)
    particle = "이" if _has_jongseong(ko) else "가"
    try:
        zf = float(z_abs)
    except (TypeError, ValueError):
        zf = 0.0
    direction = "잘" if zf >= 0 else "나쁘게"
    return f"{ko}{particle} 평균보다 {direction} 읽힘"


def _special_note(verdict, z_resid, moves):
    """Mismatch note: absolute key vs move-kind-relative surprise."""
    try:
        zr = float(z_resid)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(zr):
        return None
    ko = _move_ko_first(moves)
    if verdict == "나쁘게" and zr >= 1.0:
        return f"{ko}치고는 부드러웠다"
    if verdict == "잘" and zr <= -1.0:
        return f"{ko}치고는 싸늘했다"
    return None


def _reason_of(turn, before_i, after_i, z_abs=None, z_resid=None,
               verdict=None) -> str:
    line = turn.get("line_ko") if isinstance(turn, dict) else None
    if not (isinstance(line, str) and line.strip()):
        line = turn.get("line_en") if isinstance(turn, dict) else None
    moves = turn.get("moves") if isinstance(turn, dict) else None
    moves = list(moves) if isinstance(moves, list) else []
    base = None
    if isinstance(line, str) and line.strip():
        s = line.strip().replace("잔고", "관계 점수")
        if ". " in s:
            s = s.split(". ", 1)[0]
        elif "." in s:
            s = s.split(".", 1)[0]
        s = s.strip()
        for m in moves:
            if isinstance(m, str) and m and m in s:
                s = s.replace(m, MOVE_KO.get(m, m))
        if s.strip():
            base = s.strip()
    if base is None:
        if z_abs is None:
            ko = ", ".join(MOVE_KO.get(m, m) for m in moves
                           if isinstance(m, str))
            if not ko:
                ko = "발언"
            base = f"{ko} · 관계 점수 {before_i} → {after_i}"
        else:
            base = _fallback_reason(moves, z_abs)
    if verdict is not None and z_resid is not None:
        special = _special_note(verdict, z_resid, moves)
        if special:
            return special
    return base


def _outcome_text(view, outcomes) -> str:
    view = view if isinstance(view, dict) else {}
    speakers = list(view.get("speakers") or [])
    disp_names = view.get("display_names") or {}
    if not isinstance(disp_names, dict):
        disp_names = {}
    trail = view.get("standings") or []
    last = trail[-1] if trail and isinstance(trail[-1], dict) else {}
    parts = []
    for sp in speakers:
        disp = disp_names.get(sp, sp)
        try:
            end = standing_round(last.get(sp, 50.0))
        except (TypeError, ValueError):
            end = 50
        parts.append(f"{disp} 50 → {end}")
    base = " · ".join(parts) if parts else "기록 없음"
    extra = []
    if isinstance(outcomes, dict):
        for sp in speakers:
            try:
                line = outcome_line(outcomes.get(sp))
            except (TypeError, ValueError):
                line = None
            if line:
                extra.append(f"{disp_names.get(sp, sp)}: {line}")
    if extra:
        return base + " / " + "; ".join(extra)
    return base


def build_game(item, commentary, names=None, outcomes=None,
               panel=None) -> dict:
    """Assemble the layer-3 game model from record + commentary JSON."""
    from . import board2 as _B2

    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    names = names if isinstance(names, dict) else {}
    if outcomes is None and isinstance(item.get("outcomes"), dict):
        outcomes = item["outcomes"]
    view = build_view2(item, commentary, outcomes=outcomes, names=names)
    turns_v = view.get("turns") or []
    if not isinstance(turns_v, list):
        turns_v = []
    trail = view.get("standings") or []
    if not isinstance(trail, list):
        trail = []
    momentum = list(view.get("momentum") or [])
    decisive = decisive_of(commentary, view, names)
    flags = [{"n": d["n"], "pos": d["pos"], "color": d["color"]}
             for d in decisive
             if isinstance(d, dict) and d.get("pos") is not None]
    dis_by_idx = _disagreements_by_index(item)
    p75 = _p75(list(dis_by_idx.values())) if dis_by_idx else None
    panel_data = misread.load_panel(panel)
    panel_mode = bool(panel_data)
    read_by_idx: dict = {}
    if panel_mode:
        for _a in misread.analyze_panel(panel_data):
            if _a["index"] not in read_by_idx:
                read_by_idx[_a["index"]] = _a
    rows: list[dict] = []
    seen: set = set()
    for pos, t in enumerate(turns_v):
        if not isinstance(t, dict):
            continue
        speaker = t.get("speaker", "")
        if not isinstance(speaker, str):
            speaker = str(speaker)
        display = t.get("display", speaker)
        if not isinstance(display, str):
            display = str(display)
        idx = t.get("index", pos)
        text = t.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        try:
            za_f = float(t.get("z", 0.0))
        except (TypeError, ValueError):
            za_f = 0.0
        if not math.isfinite(za_f):
            za_f = 0.0
        raw_zr = t.get("z_resid", 0.0)
        if raw_zr is None:
            zr_f = 0.0
        else:
            try:
                zr_f = float(raw_zr)
            except (TypeError, ValueError):
                zr_f = 0.0
            if not math.isfinite(zr_f):
                zr_f = 0.0
        dis = dis_by_idx.get(idx)
        if dis is None and isinstance(idx, int) is False:
            dis = dis_by_idx.get(pos)
        try:
            dis_f = float(dis) if dis is not None else None
        except (TypeError, ValueError):
            dis_f = None
        if dis_f is not None and not math.isfinite(dis_f):
            dis_f = None
        try:
            if (pos > 0 and pos - 1 < len(trail)
                    and isinstance(trail[pos - 1], dict)):
                before_v = trail[pos - 1].get(speaker, 50.0)
            else:
                before_v = 50.0
            if pos < len(trail) and isinstance(trail[pos], dict):
                after_v = trail[pos].get(speaker, 50.0)
            else:
                after_v = 50.0
        except (TypeError, ValueError):
            before_v, after_v = 50.0, 50.0
        before_i = standing_round(before_v)
        after_i = standing_round(after_v)
        delta = after_i - before_i
        verdict = direction_verdict(delta)
        noun = _DIR_NOUN.get(verdict, "그대로")
        is_first = speaker not in seen
        seen.add(speaker)
        if is_first:
            asked = True
        else:
            asked = not _is_ambiguous(delta)
            if (asked and dis_f is not None and p75 is not None
                    and dis_f > p75):
                asked = False
        abs_verdict = reader_verdict(za_f)
        options: list = []
        invited: list = []
        key_in_invited = False
        reveal = ""
        row_misread = False
        if panel_mode:
            question = "상대는 이 말을 어떻게 받을까?"
            a = None
            if isinstance(idx, int) and not isinstance(idx, bool):
                a = read_by_idx.get(idx)
            if a is None:
                a = read_by_idx.get(pos)
            if a is not None and a.get("majority"):
                key = a["majority"]
                verdict = key
                invited = list(a.get("invited") or [])
                key_in_invited = key in invited
                options = [{"value": v, "label": lb}
                           for v, lb in misread.options_for(
                               key, invited, a.get("dist"))]
                reveal = misread.reveal_line(a, before_i, after_i)
                row_misread = bool(a.get("misread"))
                try:
                    asked = float(a.get("share", 0.0)) >= 0.5
                except (TypeError, ValueError):
                    asked = False
            else:
                asked = False
        else:
            question = question_for(display)
        rows.append({
            "index": idx,
            "speaker": speaker,
            "display": display,
            "text": text,
            "z_abs": za_f,
            "z_resid": zr_f,
            "verdict": verdict,
            "asked": bool(asked),
            "disagreement": dis_f,
            "score_before": before_i,
            "score_after": after_i,
            "delta": delta,
            "question": question,
            "score_line": f"{display}의 관계 점수 {before_i} → {after_i} ({noun})",
            "reason": _reason_of(t, before_i, after_i, z_abs=za_f,
                                 z_resid=zr_f, verdict=abs_verdict),
            "options": options,
            "invited": invited,
            "key_in_invited": bool(key_in_invited),
            "reveal": reveal,
            "misread": bool(row_misread),
        })
    n = len(momentum)
    marks_xy: list[list[float]] = []
    for i, m in enumerate(momentum):
        try:
            mf = float(m)
        except (TypeError, ValueError):
            mf = 0.0
        try:
            x = float(_B2._tide_x(i, n))
        except (TypeError, ValueError):
            x = 0.0
        try:
            y = float(tide_y(mf, _B2._MID, _B2._AMP))
        except (TypeError, ValueError):
            y = 0.0
        marks_xy.append([round(x, 1), round(y, 1)])
    item_id = item.get("id", "")
    item_id = item_id if isinstance(item_id, str) else str(item_id or "")
    speakers = list(view.get("speakers") or [])
    disp_names = view.get("display_names") or {}
    if not isinstance(disp_names, dict):
        disp_names = {}
    if len(speakers) >= 2:
        d0 = disp_names.get(speakers[0], speakers[0])
        d1 = disp_names.get(speakers[1], speakers[1])
        legend = f"빨강 {d0} · 파랑 {d1} · 선이 내려오면 {d0} 우세"
    elif len(speakers) == 1:
        d0 = disp_names.get(speakers[0], speakers[0])
        legend = f"빨강 {d0} · 선이 내려오면 {d0} 우세"
    else:
        legend = ""
    asked_n = sum(1 for r in rows if r.get("asked"))
    return {
        "id": item_id,
        "total": len(rows),
        "asked": asked_n,
        "panel": panel_mode,
        "disagreement_p75": p75,
        "turns": rows,
        "momentum": [float(m) if isinstance(m, (int, float)) else 0.0
                     for m in momentum],
        "marks_xy": marks_xy,
        "flags": flags,
        "tide_svg": tide_svg(momentum, flags),
        "legend": legend,
        "outcome": _outcome_text(view, outcomes),
    }


_CSS = """
*{box-sizing:border-box}
[hidden]{display:none!important}
body{max-width:480px;margin:0 auto;padding:0 16px 40px;background:#fff;
  color:#222;font-family:system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
  line-height:1.6}
h1{font-size:20px;margin:20px 0 2px}
h2{font-size:16px;margin:0 0 8px}
.badge{display:inline-block;border:1px solid #c9d1d9;border-radius:999px;
  padding:1px 10px;font-size:13px;color:#5b6470;background:#f7f8fa}
small{color:#5b6470}
.card{border:1px solid #e3e6ea;border-radius:12px;padding:14px;margin:12px 0;
  background:#f7f8fa}
.note{color:#5b6470;font-size:13px}
#speaker{font-weight:700;margin:0}
#text{font-size:17px;margin:8px 0}
#question{font-weight:700;margin:8px 0 0}
#guessbtns{display:flex;flex-direction:column;gap:8px;margin-top:10px}
button{border:1px solid #c9d1d9;background:#fff;color:#222;border-radius:10px;
  padding:10px 12px;font-size:16px;cursor:pointer}
button:active{background:#eef1f4}
#start,#next,#copy,#export{background:#2f6fed;border-color:#2f6fed;color:#fff;font-weight:700}
#export{margin-left:8px}
#verdict{font-weight:700;margin:8px 0 0}
#scoreline{margin:4px 0}
#reason{margin:4px 0}
#tally{font-weight:700;margin:10px 0 0}
#hitrate{font-weight:700}
#tide{width:100%;height:auto;display:block}
.tide-a{stroke:none}
.tide-b{stroke:none}
.tide-line{fill:none;stroke:#333;stroke-width:2}
.mid{stroke:#9aa4b2;stroke-width:1;stroke-dasharray:4 3}
.flag line{stroke-width:2}
.flag text{font-size:11px;font-weight:700;fill:#fff;text-anchor:middle}
.gmark{font-size:12px;font-weight:700;text-anchor:middle}
#share{width:100%;min-height:64px;font-size:14px;border:1px solid #c9d1d9;
  border-radius:10px;padding:8px}
.legend{font-size:13px;color:#5b6470;margin:6px 0 0}
"""

_JS = """(function(){
"use strict";
var data=JSON.parse(document.getElementById("data").textContent);
var turns=data.turns||[];
var N=turns.length;
var xy=data.marks_xy||[];
var dialogueId=data.id||"";
var askedTotal=0;
if(typeof data.asked==="number"){askedTotal=data.asked;}
else{for(var ci=0;ci<N;ci++){if(turns[ci]&&turns[ci].asked)askedTotal++;}}
var idx=0,hits=0,askedDone=0;
var guesses=[];
for(var gi=0;gi<N;gi++)guesses.push(null);
var picks=[];
function $(id){return document.getElementById(id);}
function askedBefore(i){
  var c=0;
  for(var k=0;k<i;k++){if(turns[k]&&turns[k].asked)c++;}
  return c;
}
function progressText(i){
  if(!askedTotal)return "이야기";
  var k=askedBefore(i)+1;
  if(k>askedTotal)k=askedTotal;
  if(turns[i]&&turns[i].asked)return "문제 "+k+" / "+askedTotal;
  return "이야기 · 문제 "+k+" / "+askedTotal;
}
function savePicks(){
  try{
    localStorage.setItem("uptake-game:"+dialogueId,JSON.stringify(
      {dialogue:dialogueId,picks:picks,hits:hits,asked:askedTotal}));
  }catch(e){}
}
function isHit(g,t){
  if(!t)return false;
  if(g===t.verdict)return true;
  if(t.key_in_invited&&t.invited){
    for(var j=0;j<t.invited.length;j++){if(g===t.invited[j])return true;}
  }
  return false;
}
var staticBtns=null;
function bindGuessBtns(){
  var btns=document.querySelectorAll("#guessbtns button");
  for(var b=0;b<btns.length;b++){
    btns[b].onclick=function(){onGuess(this.getAttribute("data-guess"));};
  }
}
function showTurn(i){
  var t=turns[i];
  $("progress").textContent=progressText(i);
  $("speaker").textContent=t.display||t.speaker;
  $("text").textContent=t.text||"";
  if(!t.asked){
    $("question").textContent="";
    $("guessbtns").hidden=true;
    $("verdict").textContent="해설 없음";
    $("scoreline").textContent=t.score_line||"";
    $("reason").textContent="";
    $("reveal").hidden=false;
    $("tally").textContent="나 "+hits+" / 해설자 "+askedDone;
    $("next").textContent=(i+1<N)?"다음":"결과 보기";
    return;
  }
  $("question").textContent=t.question||"";
  $("reveal").hidden=true;
  var gb=$("guessbtns");
  if(t.options&&t.options.length){
    gb.innerHTML="";
    for(var o=0;o<t.options.length;o++){
      var btn=document.createElement("button");
      btn.type="button";
      btn.setAttribute("data-guess",t.options[o].value);
      btn.textContent=t.options[o].label;
      gb.appendChild(btn);
    }
    gb.setAttribute("data-dyn","1");
  }else if(gb.getAttribute("data-dyn")==="1"&&staticBtns!==null){
    gb.innerHTML=staticBtns;
    gb.removeAttribute("data-dyn");
  }
  bindGuessBtns();
  $("guessbtns").hidden=false;
}
function onGuess(g){
  var t=turns[idx];
  if(!t||!t.asked)return;
  var ok=isHit(g,t);
  guesses[idx]=g;
  picks.push({turn:t.index,pick:g,key:t.verdict,z_abs:t.z_abs});
  if(ok)hits++;
  askedDone++;
  savePicks();
  if(t.reveal){$("verdict").textContent=t.reveal+(ok?" (맞힘 \\u2713)":" (틀림 \\u2717)");}
  else{$("verdict").textContent="해설자 판정: "+t.verdict+(ok?" (맞힘 \\u2713)":" (틀림 \\u2717)");}
  $("scoreline").textContent=t.score_line||"";
  $("reason").textContent=t.reason||"";
  $("guessbtns").hidden=true;
  $("reveal").hidden=false;
  $("tally").textContent="나 "+hits+" / 해설자 "+askedDone;
  $("next").textContent=(idx+1<N)?"다음":"결과 보기";
}
function markTide(){
  var svg=document.getElementById("tide");
  if(!svg)return;
  var NS=svg.namespaceURI;
  for(var i=0;i<N;i++){
    if(!turns[i]||!turns[i].asked)continue;
    if(guesses[i]===null||guesses[i]===undefined)continue;
    var p=xy[i];
    if(!p)continue;
    var ok=isHit(guesses[i],turns[i]);
    var tx=document.createElementNS(NS,"text");
    tx.setAttribute("x",p[0]);
    tx.setAttribute("y",p[1]-14);
    tx.setAttribute("class","gmark");
    tx.setAttribute("fill",ok?"#1a7f37":"#cf222e");
    tx.textContent=ok?"\\u2713":"\\u2717";
    var ti=document.createElementNS(NS,"title");
    ti.textContent="#"+turns[i].index+" "+(ok?"맞힘":"틀림");
    tx.appendChild(ti);
    svg.appendChild(tx);
  }
}
function showEnd(){
  $("play").hidden=true;
  $("end").hidden=false;
  var pct=askedTotal?Math.round(100*hits/askedTotal):0;
  $("hitrate").textContent="해설자와 "+askedTotal+"수 중 "+hits+"수 일치 ("+pct+"%)";
  var ml=$("misreadline");
  if(ml){
    if(data.panel){
      var misN=0,misK=0;
      for(var m=0;m<N;m++){
        var tm=turns[m];
        if(!tm||!tm.misread)continue;
        misN++;
        if(tm.asked&&guesses[m]!==null&&guesses[m]!==undefined&&isHit(guesses[m],tm))misK++;
      }
      ml.textContent="오해 지점 "+misN+"곳 중 "+misK+"곳을 미리 봤다";
    }else{ml.textContent="";}
  }
  var share="나는 협상의 흐름을 "+askedTotal+"수 중 "+hits+"수 맞혔다 — uptake";
  $("share").value=share;
  savePicks();
  try{localStorage.setItem("uptake-game-last",share);}catch(e){}
  try{
    var last=localStorage.getItem("uptake-game-last");
    if(last)$("lastscore").textContent="지난 기록: "+last;
  }catch(e2){}
  markTide();
}
$("start").addEventListener("click",function(){
  $("intro").hidden=true;
  if(!N){showEnd();return;}
  $("play").hidden=false;
  idx=0;hits=0;askedDone=0;picks=[];
  guesses=[];
  for(var k=0;k<N;k++)guesses.push(null);
  $("tally").textContent="나 0 / 해설자 0";
  showTurn(0);
});
staticBtns=document.getElementById("guessbtns").innerHTML;
bindGuessBtns();
$("next").addEventListener("click",function(){
  idx++;
  if(idx<N)showTurn(idx);
  else showEnd();
});
$("copy").addEventListener("click",function(){
  var s=$("share");
  s.select();
  try{
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(s.value);
    }else{document.execCommand("copy");}
  }catch(e){
    try{document.execCommand("copy");}catch(e2){}
  }
});
var expBtn=$("export");
if(expBtn){expBtn.addEventListener("click",function(){
  var payload={dialogue:dialogueId,picks:picks,hits:hits,asked:askedTotal};
  var text=JSON.stringify(payload);
  try{
    var blob=new Blob([text],{type:"application/json"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(blob);
    a.download="uptake-game-"+(dialogueId||"game")+".json";
    document.body.appendChild(a);
    a.click();
    setTimeout(function(){
      try{URL.revokeObjectURL(a.href);}catch(e){}
      try{a.remove();}catch(e2){}
    },0);
  }catch(e3){
    try{navigator.clipboard.writeText(text);}catch(e4){}
  }
});}
try{
  var prev=localStorage.getItem("uptake-game-last");
  if(prev)$("lastscore").textContent="지난 기록: "+prev;
}catch(e3){}
})();
"""


def render_html(game) -> str:
    """Render the game model as one self-contained HTML page."""
    game = game if isinstance(game, dict) else {}
    turns = game.get("turns") or []
    if not isinstance(turns, list):
        turns = []
    total = game.get("total", len(turns))
    try:
        total = int(total)
    except (TypeError, ValueError):
        total = len(turns)
    asked = game.get("asked", total)
    try:
        asked = int(asked)
    except (TypeError, ValueError):
        asked = sum(1 for t in turns
                    if isinstance(t, dict) and t.get("asked"))
    raw_id = game.get("id", "")
    raw_id = raw_id if isinstance(raw_id, str) else str(raw_id or "")
    item_id = html.escape(raw_id)
    tide = game.get("tide_svg", "")
    if not isinstance(tide, str):
        tide = ""
    legend = str(game.get("legend", "") or "")
    legend_html = ("<p class=\"legend\">" + html.escape(legend) + "</p>"
                   if legend else "")
    outcome = html.escape(str(game.get("outcome", "") or ""))
    data = {
        "id": raw_id,
        "turns": [t for t in turns if isinstance(t, dict)],
        "total": total,
        "asked": asked,
        "panel": bool(game.get("panel")),
        "marks_xy": game.get("marks_xy", []),
    }
    if game.get("panel"):
        badge_html = ""
        footnote = "질문은 '상대가 어떻게 받을까'입니다"
    else:
        badge_html = '<p class="note"><span class="badge">관계 축만</span></p>\n'
        footnote = "질문은 '직전보다 오르나 내리나'입니다"
    data_json = json.dumps(data, ensure_ascii=False)
    data_json = (data_json.replace("<", "\\u003c")
                 .replace(">", "\\u003e").replace("&", "\\u0026"))
    guess_btns = "".join(
        '<button type="button" data-guess="'
        + html.escape(val, quote=True) + '">'
        + html.escape(label) + "</button>"
        for val, label in GUESSES
    )
    share_hint = (
        "<p class=\"note\">끝까지 가면 공유 문장을 복사할 수 있습니다: "
        "나는 협상의 흐름을 " + html.escape(str(asked)) + "수 중 N수 맞혔다 — uptake</p>"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>다음 수 맞히기 — uptake</title>\n"
        "<style>" + _CSS + "</style>\n</head>\n<body>\n"
        "<h1>다음 수 맞히기 <small>guess the next move</small></h1>\n"
        f'<p class="note">{item_id}</p>\n'
        '<section id="intro" class="card">\n'
        f"<p>{html.escape(INTRO)}</p>\n"
        f'<p class="note">{html.escape(INTRO_NOTE)}</p>\n'
        + badge_html +
        '<button id="start" type="button">시작</button>\n'
        "</section>\n"
        '<section id="play" class="card" hidden>\n'
        '<p class="note" id="progress"></p>\n'
        '<p id="speaker"></p>\n'
        '<p id="text"></p>\n'
        '<p id="question"></p>\n'
        '<div id="guessbtns">' + guess_btns + "</div>\n"
        '<div id="reveal" hidden>\n'
        '<p id="verdict"></p>\n'
        '<p id="scoreline"></p>\n'
        '<p id="reason" class="note"></p>\n'
        '<button id="next" type="button">다음</button>\n'
        "</div>\n"
        '<p id="tally">나 0 / 해설자 0</p>\n'
        "</section>\n"
        '<section id="end" class="card" hidden>\n'
        "<h2>결과</h2>\n"
        '<p id="hitrate"></p>\n'
        '<p id="misreadline"></p>\n'
        '<div id="tidewrap">' + tide + "</div>\n"
        + legend_html + "\n"
        + '<p id="outcome"><b>실제 결과:</b> ' + outcome + "</p>\n"
        + share_hint + "\n"
        + '<textarea id="share" readonly rows="3"></textarea>\n'
        + '<button id="copy" type="button">복사</button>\n'
        + '<button id="export" type="button">내 답 내보내기</button>\n'
        + '<p class="note">' + footnote + "</p>\n"
        + '<p id="lastscore" class="note"></p>\n'
        + "</section>\n"
        + '<script id="data" type="application/json">'
        + data_json + "</script>\n"
        + "<script>" + _JS + "</script>\n"
        + "</body>\n</html>\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-game")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None,
                    help="JSON object mapping speaker labels to display names")
    ap.add_argument("--panel", default=None,
                    help="0-100/dist5 panel record of the same dialogue")
    args = ap.parse_args(argv)

    with open(args.item, encoding="utf-8") as f:
        item = json.load(f)
    with open(args.commentary, encoding="utf-8") as f:
        commentary = json.load(f)
    names = None
    if args.names:
        try:
            names = json.loads(args.names)
        except (TypeError, ValueError):
            names = None
    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    game = build_game(item, commentary, names=names, panel=args.panel)
    out_path.write_text(render_html(game), encoding="utf-8")
