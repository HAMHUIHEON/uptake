"""Layer-3 "rewrite the decisive turn" play server + page (task53).

Stdlib HTTP server (127.0.0.1 only) plus a mobile-first Korean play page.
Pairwise judging reuses the vote code from ``forecaster.pairwise`` and
``forecaster.perturb``: shared context up to the turn, candidates
A = original / B = player text, both orders x 3 stances = 6 votes.
"""
from __future__ import annotations

import argparse
import html
import http.server
import json
import threading
import time
import urllib.parse
from pathlib import Path

from . import pairwise as PW
from . import perturb
from . import commentary as CM
from .commentary import standing_round
from .reader import DEFAULT_MODEL
from .usage import BudgetExceeded, Meter

STANCE_KO = {"neutral": "중립", "trusting": "우호", "suspicious": "의심"}
MAX_TEXT_LEN = 400
DEFAULT_COST_HINT = 0.01
BUDGET_MESSAGE = "예산이 소진되었습니다. 잠시 후 다시 시도해 주세요."


def _ro_particle(number) -> str:
    """'로' vs '으로' after a standing, by its Korean reading."""
    try:
        ones = abs(int(number)) % 10
    except (TypeError, ValueError):
        return "로"
    if ones in (1, 2, 4, 5, 7, 8, 9):
        return "로"
    return "으로"


def _decisive_line(display, before, after) -> str:
    """Original-turn marker, e.g. '이 말에서 B의 관계 점수가 50 → 8로 떨어졌다'."""
    try:
        b = int(before)
    except (TypeError, ValueError):
        b = 50
    try:
        a = int(after)
    except (TypeError, ValueError):
        a = 50
    name = display if isinstance(display, str) and display else "화자"
    ro = _ro_particle(a)
    if a > b:
        verb = "올랐다"
    elif a < b:
        verb = "떨어졌다"
    else:
        verb = "그대로였다"
    return f"이 말에서 {name}의 관계 점수가 {b} → {a}{ro} {verb}"


def margin_word(margin) -> str:
    """Softness label from the vote margin: 부드럽다 / 비슷하다 / 거칠다."""
    try:
        m = float(margin)
    except (TypeError, ValueError):
        return "비슷하다"
    if m > 0.15:
        return "부드럽다"
    if m < -0.15:
        return "거칠다"
    return "비슷하다"


def validate_text(text) -> tuple[bool, str]:
    """Check a player rewrite; returns (ok, error_message)."""
    if not isinstance(text, str) or not text.strip():
        return (False, "다시 쓴 말을 입력해 주세요.")
    if len(text) > MAX_TEXT_LEN:
        return (False, f"{MAX_TEXT_LEN}자까지 입력할 수 있습니다.")
    return (True, "")


def _load_scene_pool(scenes):
    if scenes is None:
        return []
    if isinstance(scenes, (str, Path)):
        try:
            from .scenes import load_scenes as _load
        except ImportError:
            return []
        try:
            return _load(scenes)
        except SystemExit:
            return []
    if isinstance(scenes, dict) and isinstance(scenes.get("scenes"), list):
        return [s for s in scenes["scenes"] if isinstance(s, dict)]
    if isinstance(scenes, list):
        return [s for s in scenes if isinstance(s, dict)]
    return []


def _load_suggest_map(suggests) -> dict:
    if suggests is None:
        return {}
    data = None
    if isinstance(suggests, (str, Path)):
        try:
            with open(suggests, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
    else:
        data = suggests
    if isinstance(data, dict) and isinstance(data.get("notices"), list):
        seq = data["notices"]
    elif isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        out: dict = {}
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            out[idx] = val
        return out
    else:
        return {}
    out = {}
    for n in seq:
        if not isinstance(n, dict):
            continue
        idx = n.get("turn", n.get("index"))
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        out[idx] = n
    return out


def _is_blunder_detail(detail, commentary_turn=None) -> bool:
    cand = dict(detail) if isinstance(detail, dict) else {}
    if isinstance(commentary_turn, dict):
        if commentary_turn.get("z_resid") is not None:
            cand["z_resid"] = commentary_turn.get("z_resid")
        if "standing_before" in commentary_turn:
            cand["standing_before"] = commentary_turn.get("standing_before")
        if "standing_after" in commentary_turn:
            cand["standing_after"] = commentary_turn.get("standing_after")
    # Map play detail standings to the blunder-test keys.
    if "standing_before" not in cand and "before" in cand:
        cand["standing_before"] = cand.get("before")
    if "standing_after" not in cand and "after" in cand:
        cand["standing_after"] = cand.get("after")
    try:
        from .scenes import is_blunder as _ib

        return bool(_ib(cand))
    except ImportError:
        pass
    try:
        zr = cand.get("z_resid")
        if zr is not None and float(zr) <= -1.5:
            return True
    except (TypeError, ValueError):
        pass
    try:
        b = float(cand.get("standing_before", 50))
        a = float(cand.get("standing_after", 50))
        return (b - a) >= 15
    except (TypeError, ValueError):
        return False


def attach_play_notices(details, commentary=None, item=None, scenes=None,
                        suggests=None, dialogue_id="", culture="mix",
                        lang="ko") -> int:
    """Attach notice_html to blunder details in place. Returns notice count."""
    pool = _load_scene_pool(scenes)
    sug_map = _load_suggest_map(suggests)
    if not pool and not sug_map:
        return 0
    commentary = commentary if isinstance(commentary, dict) else {}
    raw = commentary.get("turns", [])
    raw = raw if isinstance(raw, list) else []
    com_by_index: dict[int, dict] = {}
    for c in raw:
        if isinstance(c, dict) and isinstance(c.get("index"), int):
            com_by_index.setdefault(c["index"], c)
    moves_by_index: dict[int, list] = {}
    item = item if isinstance(item, dict) else {}
    for m in (item.get("moves", []) or []):
        if isinstance(m, dict):
            idx = m.get("turn_index")
            mv = m.get("moves", []) or []
        else:
            idx = getattr(m, "turn_index", None)
            try:
                mv = list(getattr(m, "moves", []) or [])
            except TypeError:
                mv = []
        if isinstance(idx, int) and idx not in moves_by_index:
            moves_by_index[idx] = [x for x in mv if isinstance(x, str)]
    by_id = {str(s.get("id", "")): s for s in pool}
    try:
        from .scenes import pick_scene as _pick
        from .scenes import render_notice as _render
    except ImportError:
        _pick = None
        _render = None
    used: set[str] = set()
    for n in sug_map.values():
        sid = n.get("scene_id", "") if isinstance(n, dict) else ""
        if sid:
            used.add(str(sid))
    count = 0
    blunder_pos = 0
    for d in details or []:
        if not isinstance(d, dict):
            continue
        idx = d.get("index", 0)
        try:
            idx_i = int(idx) if not isinstance(idx, bool) else 0
        except (TypeError, ValueError):
            idx_i = 0
        com_turn = com_by_index.get(idx_i)
        if not _is_blunder_detail(d, com_turn):
            continue
        sug = sug_map.get(idx_i, {})
        alts = []
        if isinstance(sug, dict):
            alts = [a for a in (sug.get("alternatives", []) or [])
                    if isinstance(a, dict)]
        scene = None
        sid = sug.get("scene_id", "") if isinstance(sug, dict) else ""
        if sid and str(sid) in by_id:
            scene = by_id[str(sid)]
        elif _pick is not None and pool:
            moves = []
            if isinstance(com_turn, dict):
                moves = [m for m in (com_turn.get("moves", []) or [])
                         if isinstance(m, str)]
            if not moves:
                moves = list(moves_by_index.get(idx_i, []))
            try:
                scene = _pick(moves or "other", dialogue_id, idx_i, pool,
                              culture=culture, used=used,
                              position=blunder_pos)
            except (TypeError, ValueError):
                scene = None
            if scene is not None:
                used.add(str(scene.get("id", "")))
        try:
            before = int(d.get("before", 50))
        except (TypeError, ValueError):
            before = 50
        try:
            after = int(d.get("after", 50))
        except (TypeError, ValueError):
            after = 50
        notice = {
            "before": before,
            "after": after,
            "display": d.get("display", d.get("speaker", "")),
            "speaker": d.get("speaker", ""),
            "alternatives": alts,
            "scene": scene,
            "lang": lang,
        }
        d["notice"] = notice
        if _render is not None:
            try:
                d["notice_html"] = _render(
                    before=before, after=after,
                    display=notice["display"], alternatives=alts,
                    scene=scene, lang=lang, speaker=notice["speaker"])
            except (TypeError, ValueError):
                d["notice_html"] = ""
        else:
            d["notice_html"] = ""
        count += 1
        blunder_pos += 1
    return count


def build_dialogue(item, commentary=None, names=None, fallback_norm=None,
                   scenes=None, suggests=None, culture="mix", lang="ko",
                   **kwargs) -> dict:
    """Assemble the play payload: turns + decisive indices + standings.

    Standings come from the commentary norm via
    ``commentary.standing_pairs`` (``fallback_norm`` covers a commentary
    without a norm). Decisive turns are the commentary's picked turns
    (up to 3); when the commentary has none, the 3 largest standing
    swings are used so the page still has tabs.
    """
    if suggests is None:
        suggests = kwargs.get("suggestions", kwargs.get("suggest"))
    if scenes is None:
        scenes = kwargs.get("scene_list", kwargs.get("scenes_path"))
    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    names = names if isinstance(names, dict) else {}
    raw_id = item.get("id", "")
    did = raw_id if isinstance(raw_id, str) else str(raw_id or "")
    norm = commentary.get("norm")
    norm = norm if isinstance(norm, dict) else {}
    if ("mu" not in norm and "sd" not in norm) and isinstance(
            fallback_norm, dict):
        norm = fallback_norm
    effects = [e for e in (item.get("effects", []) or [])
               if isinstance(e, dict)]
    try:
        pairs = CM.standing_pairs(item, norm)
    except (TypeError, ValueError):
        pairs = [(50.0, 50.0)] * len(effects)
    turns: list[dict] = []
    for pos, e in enumerate(effects):
        idx = e.get("index", pos)
        if isinstance(idx, bool) or not isinstance(idx, int):
            idx = pos
        sp = e.get("speaker", "")
        sp = sp if isinstance(sp, str) else str(sp)
        tx = e.get("text", "")
        tx = tx if isinstance(tx, str) else str(tx)
        if pos < len(pairs):
            b_raw, a_raw = pairs[pos]
        else:
            b_raw, a_raw = (50.0, 50.0)
        try:
            bi = standing_round(b_raw)
        except (TypeError, ValueError):
            bi = 50
        try:
            ai = standing_round(a_raw)
        except (TypeError, ValueError):
            ai = 50
        disp = names.get(sp, sp)
        if not isinstance(disp, str):
            disp = str(disp)
        turns.append({"index": idx, "speaker": sp, "display": disp,
                      "text": tx, "score_before": bi, "score_after": ai})
    have = set()
    for t in turns:
        have.add(t["index"])
    raw_turns = commentary.get("turns", [])
    raw_turns = raw_turns if isinstance(raw_turns, list) else []
    dec_idx: list[int] = []
    for c in raw_turns:
        if not isinstance(c, dict):
            continue
        ci = c.get("index")
        if isinstance(ci, bool) or not isinstance(ci, int):
            continue
        if ci in have and ci not in dec_idx:
            dec_idx.append(ci)
        if len(dec_idx) >= 3:
            break
    if not dec_idx and turns:
        ranked = sorted(turns, key=lambda t: (
            -abs(t["score_after"] - t["score_before"]), t["index"]))
        dec_idx = sorted(t["index"] for t in ranked[:3])
    dec_idx = dec_idx[:3]
    standings = [{"index": t["index"], "speaker": t["speaker"],
                  "before": t["score_before"], "after": t["score_after"]}
                 for t in turns]
    details: list[dict] = []
    for ci in dec_idx:
        pos = None
        found = None
        for p, t in enumerate(turns):
            if t["index"] == ci:
                pos, found = p, t
                break
        if found is None:
            continue
        quote_full = found["text"]
        quote = (quote_full[:80] + "…") if len(quote_full) > 80 else quote_full
        details.append({
            "index": ci,
            "pos": pos,
            "speaker": found["speaker"],
            "display": found["display"],
            "before": found["score_before"],
            "after": found["score_after"],
            "line": _decisive_line(found["display"], found["score_before"],
                                   found["score_after"]),
            "quote": quote,
            "quote_full": quote_full,
        })
    if scenes is not None or suggests is not None:
        try:
            attach_play_notices(details, commentary=commentary, item=item,
                                scenes=scenes, suggests=suggests,
                                dialogue_id=did, culture=culture, lang=lang)
        except Exception:
            pass
    return {"id": did, "turns": turns, "decisive": list(dec_idx),
            "decisive_turns": list(dec_idx), "standings": standings,
            "decisive_details": details, "lang": lang}


def _target_pos(record, turn) -> int | None:
    effects = record.get("effects", []) if isinstance(record, dict) else []
    for pos, e in enumerate(effects or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else pos
        if idx == turn:
            return pos
    return None


def compare_texts(client, record, turn, player_text, model=DEFAULT_MODEL,
                  meter=None, dialogue_id="") -> dict:
    """Judge a player rewrite against the original (6 votes).

    Shared context up to ``turn``, candidates A = original / B = player
    text, both presentation orders x 3 stances. ``p_better`` counts a
    tie as half a win for the player; ``margin`` is
    (player wins - original wins) / 6. Raises ``ValueError`` for an
    unknown turn and ``BudgetExceeded`` when the meter is exhausted.
    """
    record = record if isinstance(record, dict) else {}
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise ValueError("unknown_turn")
    pos_in_record = _target_pos(record, turn)
    if pos_in_record is None:
        raise ValueError("unknown_turn")
    effects = [e for e in (record.get("effects", []) or [])
               if isinstance(e, dict)]
    target = effects[pos_in_record]
    orig_text = target.get("text", "")
    orig_text = orig_text if isinstance(orig_text, str) else str(orig_text)
    speaker = target.get("speaker", "")
    speaker = speaker if isinstance(speaker, str) else str(speaker)
    did = dialogue_id
    if not did:
        raw = record.get("id", "")
        did = raw if isinstance(raw, str) else str(raw or "")
    try:
        pool, contexts = PW.collect_pool({did: record} if did else {})
    except (TypeError, ValueError):
        pool, contexts = [], {}
    ctx = contexts.get(did, {}) if isinstance(contexts, dict) else {}
    turns = ctx.get("turns", []) if isinstance(ctx, dict) else []
    pos_of = ctx.get("pos_of", {}) if isinstance(ctx, dict) else {}
    if turns and isinstance(pos_of, dict) and turn in pos_of:
        pos = pos_of[turn]
        if not isinstance(pos, int) or pos < 0 or pos >= len(turns):
            pos = min(pos_in_record, max(0, len(turns) - 1))
    else:
        from .transcribe import Turn as _Turn
        turns = [_Turn(speaker=(e.get("speaker", "")
                               if isinstance(e.get("speaker"), str)
                               else str(e.get("speaker", ""))),
                       text=(e.get("text", "")
                             if isinstance(e.get("text"), str)
                             else str(e.get("text", ""))))
                 for e in effects]
        pos = min(pos_in_record, max(0, len(turns) - 1))
    listener = ""
    for e in pool or []:
        if (isinstance(e, dict) and e.get("dialogue") == did
                and e.get("turn") == turn):
            cand = e.get("listener", "")
            if isinstance(cand, str) and cand:
                listener = cand
                break
    if not listener:
        seen: list[str] = []
        for t in turns:
            if t.speaker not in seen:
                seen.append(t.speaker)
        others = [s for s in seen if s != speaker and s != "unknown"]
        listener = others[0] if others else "listener"
    context_lines = PW.snippet_lines(turns, pos, PW.WINDOW)
    m = meter if meter is not None else getattr(client, "meter", None)
    if m is not None:
        m.check()
        cost_before = m.cost()
    else:
        cost_before = 0.0
    try:
        pid = int(turn)
    except (TypeError, ValueError):
        pid = 0
    orders = (("orig_first", orig_text, player_text),
              ("player_first", player_text, orig_text))
    votes: list[dict] = []
    for order_name, text_first, text_second in orders:
        for stance in PW.STANCES:
            try:
                s_idx = list(PW.STANCES).index(stance)
            except ValueError:
                s_idx = 0
            try:
                receivers = PW.pairwise_panel(pid)
            except (TypeError, ValueError):
                receivers = []
            if s_idx < len(receivers):
                rec = receivers[s_idx]
            else:
                from .types import Receiver as _Receiver
                rec = _Receiver(
                    id=f"pair{pid}-{s_idx}",
                    relationship="participant in this conversation",
                    power="equal", channel="conversation", state="",
                    stance=stance)
            messages = perturb.build_perturb_messages(
                context_lines, speaker, text_first, text_second,
                rec, listener)
            meta = {"layer": "pairwise", "pair": pid, "stance": stance,
                    "member": getattr(rec, "id", ""), "item": did,
                    "dialogue": did, "dialogues": [did], "turn": turn,
                    "task": "play"}
            parsed, _ = perturb._judge_one_perturb(
                client, messages, model, meta)
            better = parsed.get("better") if isinstance(parsed, dict) else None
            if better not in ("first", "second", "tie"):
                better = None
            if better is None or better == "tie":
                winner = "tie"
            elif order_name == "orig_first":
                winner = "orig" if better == "first" else "player"
            else:
                winner = "player" if better == "first" else "orig"
            votes.append({"stance": stance,
                          "stance_ko": STANCE_KO.get(stance, stance),
                          "order": order_name, "better": better,
                          "winner": winner})
    n_player = sum(1 for v in votes if v["winner"] == "player")
    n_orig = sum(1 for v in votes if v["winner"] == "orig")
    n_tie = sum(1 for v in votes if v["winner"] == "tie")
    denom = len(votes) or 6
    p_better = (n_player + 0.5 * n_tie) / denom
    margin = (n_player - n_orig) / denom
    cost_usd = (m.cost() - cost_before) if m is not None else 0.0
    return {"p_better": float(p_better), "votes": votes,
            "margin": float(margin), "cost_usd": float(cost_usd),
            "model": model}


_CSS = """
*{box-sizing:border-box}
[hidden]{display:none!important}
body{max-width:480px;margin:0 auto;padding:0 16px 40px;background:#fff;
  color:#222;font-family:system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
  line-height:1.6}
h1{font-size:20px;margin:20px 0 2px}
h2{font-size:16px;margin:0 0 8px}
small{color:#333}
.card{border:1px solid #e3e6ea;border-radius:12px;padding:14px;margin:12px 0;
  background:#f7f8fa}
.note{color:#333;font-size:13px}
.err{color:#cf222e;font-weight:700}
#tabs{margin:12px 0 0}
.tab{margin:0 6px 6px 0}
.tab.active{background:#2f6fed;border-color:#2f6fed;color:#fff;font-weight:700}
button{border:1px solid #c9d1d9;background:#fff;color:#222;border-radius:10px;
  padding:10px 12px;font-size:16px;cursor:pointer}
button:active{background:#eef1f4}
button:disabled{opacity:.55;cursor:default}
#ask{background:#2f6fed;border-color:#2f6fed;color:#fff;font-weight:700;margin-top:8px}
.ctx{margin:6px 0;font-size:14px}
details{margin:8px 0}
summary{cursor:pointer;color:#2f6fed}
#orig-speaker{font-weight:700;margin:0}
#orig-text{font-size:17px;margin:8px 0}
#orig-line{font-weight:700;margin:8px 0 0}
.notice{border:1px dashed #d0b429;border-radius:10px;padding:10px 12px;margin:8px 0;
  background:#fffbe6}
.warn{font-weight:700;margin:0}
.althead{font-weight:700;margin:8px 0 4px}
.alts{margin:4px 0;padding-left:22px}
.pct{color:#333;font-size:13px}
.scene{margin:8px 0 0}
.src{margin:2px 0 0;color:#333;font-size:13px}
#input{width:100%;min-height:84px;font-size:15px;border:1px solid #c9d1d9;
  border-radius:10px;padding:8px;font-family:inherit}
#waiting{font-weight:700}
#prob{font-weight:700;font-size:17px}
.vrow{margin:4px 0}
.vlab{display:inline-block;width:38px;font-size:13px;color:#222}
.dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin:0 3px;
  vertical-align:middle;border:1px solid #c9d1d9}
.dot.win{background:#1a7f37;border-color:#1a7f37}
.dot.lose{background:#cf222e;border-color:#cf222e}
.dot.tie{background:#c9d1d9}
#attempt-list{margin:6px 0;padding-left:22px;font-size:14px}
#share{width:100%;min-height:56px;font-size:14px;border:1px solid #c9d1d9;
  border-radius:10px;padding:8px;font-family:inherit}
footer{margin-top:20px;font-size:13px;color:#333}
"""

_JS = """(function(){
"use strict";
var data=JSON.parse(document.getElementById("data").textContent);
var turns=data.turns||[];
var details=data.decisive_details||[];
var dialogueId=data.id||"";
var API=(data.api_url||"").replace(/\\/$/,"");
var current=0;
var attemptsByTurn={};
var STANCE_KO={"neutral":"중립","trusting":"우호","suspicious":"의심"};
var STANCE_ORDER=["neutral","trusting","suspicious"];
function $(id){return document.getElementById(id);}
function api(path){return API+path;}
function ro(n){var o=Math.abs(Number(n))%10;
  return (o===1||o===2||o===4||o===5||o===7||o===8||o===9)?"로":"으로";}
function origLine(d){var v=d.after>d.before?"올랐다":
  (d.after<d.before?"떨어졌다":"그대로였다");
  return "이 말에서 "+d.display+"의 관계 점수가 "+
    d.before+" → "+d.after+ro(d.after)+" "+v;}
function turnEl(t){var p=document.createElement("p");p.className="ctx";
  p.textContent=(t.display||t.speaker)+": "+(t.text||"");return p;}
function renderAttempts(){var d=details[current];if(!d)return;
  var arr=attemptsByTurn[String(d.index)]||[];
  var box=$("attempts");box.innerHTML="";
  if(!arr.length)return;
  var h=document.createElement("p");h.className="note";
  h.textContent="내 시도 (잘된 순)";
  box.appendChild(h);
  var ol=document.createElement("ol");ol.id="attempt-list";
  for(var i=0;i<arr.length;i++){var li=document.createElement("li");
    var sn=arr[i].text;
    if(sn.length>30)sn=sn.slice(0,30)+"…";
    li.textContent=arr[i].p+"% — "+sn;ol.appendChild(li);}
  box.appendChild(ol);}
function select(i){current=i;var d=details[i];if(!d)return;
  var tabs=$("tabs").children;
  for(var k=0;k<tabs.length;k++)tabs[k].className=(k===i)?"tab active":"tab";
  var body=$("context-body");body.innerHTML="";
  var ctx=turns.slice(0,d.pos);
  var start=Math.max(0,ctx.length-3);
  if(start>0){var det=document.createElement("details");
    var sum=document.createElement("summary");
    sum.textContent="앞 대화 "+start+"개 보기";
    det.appendChild(sum);
    for(var a=0;a<start;a++)det.appendChild(turnEl(ctx[a]));
    body.appendChild(det);}
  for(var b=start;b<ctx.length;b++)body.appendChild(turnEl(ctx[b]));
  if(!ctx.length){var none=document.createElement("p");none.className="note";
    none.textContent="앞 대화 없음 (첫 수)";
    body.appendChild(none);}
  $("orig-speaker").textContent=d.display||d.speaker;
  $("orig-text").textContent=d.quote_full||d.quote||"";
  $("orig-line").textContent=d.line||origLine(d);
  var nb=$("notice");
  if(nb){if(d.notice_html){nb.innerHTML=d.notice_html;nb.hidden=false;}
    else{nb.innerHTML="";nb.hidden=true;}}
  $("input").value="";$("error").textContent="";
  $("result").hidden=true;$("waiting").hidden=true;$("ask").disabled=false;
  renderAttempts();}
function showResult(js,text){var p=Math.round(Number(js.p_better)*100);
  $("prob").textContent="당신의 말이 원문보다 잘 받아들여질 확률 "+p+"%";
  var wrap=$("votes");wrap.innerHTML="";
  for(var s=0;s<STANCE_ORDER.length;s++){var st=STANCE_ORDER[s];
    var row=document.createElement("div");row.className="vrow";
    var lab=document.createElement("span");lab.className="vlab";
    lab.textContent=STANCE_KO[st];row.appendChild(lab);
    var vs=(js.votes||[]).filter(function(v){return v&&v.stance===st;});
    for(var j=0;j<vs.length;j++){var dot=document.createElement("span");
      var w=vs[j].winner;
      dot.className="dot "+(w==="player"?"win":(w==="orig"?"lose":"tie"));
      dot.title=STANCE_KO[st]+" · "+
        (w==="player"?"당신":(w==="orig"?"원문":"비김"));
      row.appendChild(dot);}
    wrap.appendChild(row);}
  var m=Number(js.margin);
  var soft=m>0.15?"원문보다 부드럽다":
    (m<-0.15?"원문보다 거칠다":
     "원문보다 비슷하다");
  $("soft").textContent=soft;
  var d=details[current];
  var key=String(d.index);
  var arr=attemptsByTurn[key]||[];
  arr.push({text:text,p:p});
  arr.sort(function(a,b2){return b2.p-a.p;});
  attemptsByTurn[key]=arr;
  renderAttempts();
  $("share").value="나는 "+dialogueId+"의 "+d.index+
    "수를 "+p+"%짜리 말로 고쳤다 — uptake";
  if(js.cost_usd!==null&&js.cost_usd!==undefined)
    $("cost").textContent="$"+Number(js.cost_usd).toFixed(4);
  $("result").hidden=false;}
$("ask").addEventListener("click",function(){
  var d=details[current];if(!d)return;
  var text=$("input").value||"";
  if(!text.trim()){$("error").textContent="다시 쓴 말을 입력해 주세요.";return;}
  if(text.length>400){$("error").textContent="400자까지 입력할 수 있습니다.";return;}
  $("error").textContent="";$("ask").disabled=true;$("waiting").hidden=false;
  fetch(api("/api/compare"),{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({id:dialogueId,turn:d.index,text:text})}
  ).then(function(res){return res.json().then(function(js){
    return {status:res.status,body:js};});}).then(function(out){
    $("waiting").hidden=true;$("ask").disabled=false;
    if(out.status===402){$("error").textContent=(out.body&&out.body.message)||
      "예산이 소진되었습니다.";return;}
    if(out.status!==200){$("error").textContent=(out.body&&out.body.message)||
      "다시 시도해 주세요.";return;}
    showResult(out.body,text);
  }).catch(function(){$("waiting").hidden=true;$("ask").disabled=false;
    $("error").textContent="연결에 실패했습니다. "+
      "서버가 켜져 있는지 확인해 주세요.";});
});
$("retry").addEventListener("click",function(){
  $("input").value="";$("input").focus();});
$("copy").addEventListener("click",function(){var s=$("share");s.select();
  try{if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(s.value);}else{document.execCommand("copy");}
  }catch(e){try{document.execCommand("copy");}catch(e2){}}});
(function init(){var box=$("tabs");box.innerHTML="";
  if(!details.length){var p=document.createElement("p");p.className="note";
    p.textContent="결정적 장면을 찾지 못했습니다.";
    box.appendChild(p);return;}
  for(var i=0;i<details.length&&i<3;i++){
    (function(k){var b=document.createElement("button");
      b.type="button";b.className="tab";
      b.textContent=details[k].index+"수";
      b.addEventListener("click",function(){select(k);});
      box.appendChild(b);})(i);}
  select(0);})();
})();
"""


def render_page(dialogue, api_url="", cost_hint=DEFAULT_COST_HINT,
                lang="ko") -> str:
    """Render the play model as one self-contained HTML page."""
    dialogue = dialogue if isinstance(dialogue, dict) else {}
    lang = dialogue.get("lang", lang)
    raw_id = dialogue.get("id", "")
    raw_id = raw_id if isinstance(raw_id, str) else str(raw_id or "")
    api_url = api_url.strip() if isinstance(api_url, str) else ""
    if api_url.endswith("/"):
        api_url = api_url[:-1]
    try:
        ch = float(cost_hint)
    except (TypeError, ValueError):
        ch = DEFAULT_COST_HINT
    if ch >= 0.01:
        cost_str = f"${ch:.2f}"
    else:
        cost_str = f"${ch:.4f}"
    data = {
        "id": raw_id,
        "turns": dialogue.get("turns", []),
        "decisive": dialogue.get("decisive", []),
        "decisive_details": dialogue.get("decisive_details", []),
        "api_url": api_url,
        "cost_hint": ch,
    }
    data_json = json.dumps(data, ensure_ascii=False)
    data_json = (data_json.replace("<", "\\u003c")
                 .replace(">", "\\u003e").replace("&", "\\u0026"))
    lang_attr = "en" if str(lang).lower().startswith("en") else "ko"
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{lang_attr}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        "<title>다시 말해 보기 — uptake</title>\n"
        "<style>" + _CSS + "</style>\n</head>\n<body>\n"
        "<h1>다시 말해 보기</h1>\n"
        f'<p class="note">{html.escape(raw_id)}</p>\n'
        '<div id="tabs"></div>\n'
        '<section id="context" class="card">\n<h2>앞 대화</h2>\n'
        '<div id="context-body"></div>\n</section>\n'
        '<section id="original" class="card">\n<h2>원문</h2>\n'
        '<p id="orig-speaker"></p>\n'
        '<p id="orig-text"></p>\n'
        '<p id="orig-line"></p>\n'
        '<div id="notice" hidden></div>\n'
        "</section>\n"
        '<section id="rewrite" class="card">\n<h2>당신의 말</h2>\n'
        '<label for="input">같은 뜻을 당신이라면 어떻게 말할까요?</label>\n'
        '<textarea id="input" rows="3" maxlength="400"></textarea>\n'
        '<div><button id="ask" type="button">독자에게 물어보기</button></div>\n'
        '<p id="waiting" hidden>읽는 중… (약 5초)</p>\n'
        '<p id="error" class="err"></p>\n'
        "</section>\n"
        '<section id="result" class="card" hidden>\n<h2>결과</h2>\n'
        '<p id="prob"></p>\n'
        '<div id="votes"></div>\n'
        '<p id="soft"></p>\n'
        '<div><button id="retry" type="button">다시 써 보기</button></div>\n'
        '<div id="attempts"></div>\n'
        '<textarea id="share" readonly rows="2"></textarea>\n'
        '<div><button id="copy" type="button">복사</button></div>\n'
        "</section>\n"
        "<footer>모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다 · "
        '한 번 묻는 비용 약 <span id="cost">'
        + html.escape(cost_str) + "</span></footer>\n"
        + '<script id="data" type="application/json">'
        + data_json + "</script>\n"
        + "<script>" + _JS + "</script>\n"
        + "</body>\n</html>\n"
    )


class PlayState:
    """Server-side dialogue store, judge client, budget meter and log."""

    def __init__(self, records_dir, commentary_dir, fit_dir=None,
                 model=DEFAULT_MODEL, budget=5.0, meter=None,
                 log_path=None, client=None, scenes=None, suggests=None,
                 suggests_dir=None, culture="mix", lang="ko"):
        self.records_dir = Path(records_dir)
        self.commentary_dir = Path(commentary_dir)
        self.fit_dir = Path(fit_dir) if fit_dir else None
        self.model = model
        self.scenes = scenes
        self.suggests = suggests
        self.suggests_dir = Path(suggests_dir) if suggests_dir else None
        self.culture = culture
        self.lang = lang
        if meter is not None:
            self.meter = meter
        else:
            try:
                b = float(budget) if budget is not None else None
            except (TypeError, ValueError):
                b = 5.0
            self.meter = Meter(budget_usd=b)
        self.log_path = Path(log_path) if log_path else None
        self.client = client
        self._log_lock = threading.Lock()
        self._fit_norm = None
        self._fit_loaded = False

    def dialogues(self) -> list[str]:
        try:
            files = sorted(self.records_dir.glob("*.json"))
        except OSError:
            return []
        out = []
        for p in files:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            out.append(p.stem)
        return out

    def load_item(self, dialogue_id):
        if not isinstance(dialogue_id, str) or not dialogue_id:
            return None
        if "/" in dialogue_id or "\\" in dialogue_id or ".." in dialogue_id:
            return None
        p = self.records_dir / f"{dialogue_id}.json"
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def load_commentary(self, dialogue_id):
        if not isinstance(dialogue_id, str) or not dialogue_id:
            return None
        if "/" in dialogue_id or "\\" in dialogue_id or ".." in dialogue_id:
            return None
        for name in (f"{dialogue_id}.json",
                     f"commentary_{dialogue_id}.json",
                     f"{dialogue_id}.commentary.json"):
            p = self.commentary_dir / name
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return None

    def fit_norm(self):
        if self._fit_loaded:
            return self._fit_norm
        self._fit_loaded = True
        if self.fit_dir is None:
            return None
        try:
            self._fit_norm = CM._load_fit_table(self.fit_dir)
        except (OSError, SystemExit, ValueError):
            self._fit_norm = None
        return self._fit_norm

    def load_suggests(self, dialogue_id):
        if isinstance(self.suggests, dict) and "notices" not in self.suggests:
            # Per-dialogue mapping {id: {...}}.
            cand = self.suggests.get(dialogue_id)
            if cand is not None:
                return cand
        if self.suggests is not None and not isinstance(self.suggests, dict):
            return self.suggests
        if isinstance(self.suggests, dict):
            return self.suggests
        if self.suggests_dir is not None:
            if (not isinstance(dialogue_id, str) or not dialogue_id
                    or "/" in dialogue_id or "\\" in dialogue_id
                    or ".." in dialogue_id):
                return None
            for name in (f"{dialogue_id}.json",
                         f"suggest_{dialogue_id}.json",
                         f"{dialogue_id}.suggest.json"):
                p = self.suggests_dir / name
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict):
                    return data
        return None

    def get_dialogue(self, dialogue_id, names=None):
        item = self.load_item(dialogue_id)
        if item is None:
            return None
        commentary = self.load_commentary(dialogue_id) or {}
        return build_dialogue(item, commentary, names=names,
                              fallback_norm=self.fit_norm(),
                              scenes=self.scenes,
                              suggests=self.load_suggests(dialogue_id),
                              culture=self.culture, lang=self.lang)

    def log_compare(self, record: dict) -> None:
        if self.log_path is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._log_lock:
            try:
                parent = self.log_path.parent
                if str(parent):
                    parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "uptake-play/1.0"

    def log_message(self, fmt, *args):  # keep pytest output clean
        pass

    @property
    def state(self) -> PlayState:
        return self.server.state

    def _send_json(self, obj, status=200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text, status=200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight for static exports
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            did = (qs.get("id", [""])[0] or "").strip()
            if not did:
                ids = self.state.dialogues()
                if not ids:
                    self._send_html("<h1>대화가 없습니다</h1>", status=404)
                    return
                did = ids[0]
            dlg = self.state.get_dialogue(did)
            if dlg is None:
                self._send_html("<h1>대화를 찾지 못했습니다</h1>", status=404)
                return
            self._send_html(render_page(dlg, api_url="",
                                        cost_hint=DEFAULT_COST_HINT))
            return
        if parsed.path == "/api/dialogue":
            did = (qs.get("id", [""])[0] or "").strip()
            if not did:
                self._send_json({"error": "missing_id",
                                 "message": "id가 필요합니다."}, status=400)
                return
            dlg = self.state.get_dialogue(did)
            if dlg is None:
                self._send_json({"error": "unknown_id",
                                 "message": "대화를 찾지 못했습니다."},
                                status=404)
                return
            self._send_json(dlg)
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/compare":
            self._send_json({"error": "not_found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(max(0, length)) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "")
        except (ValueError, UnicodeDecodeError):
            self._send_json({"error": "bad_json",
                             "message": "요청을 읽지 못했습니다."}, status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "bad_json",
                             "message": "요청을 읽지 못했습니다."}, status=400)
            return
        did = payload.get("id", "")
        did = did.strip() if isinstance(did, str) else ""
        turn = payload.get("turn")
        text = payload.get("text", "")
        if not did:
            self._send_json({"error": "missing_id",
                             "message": "id가 필요합니다."}, status=400)
            return
        if isinstance(turn, bool) or not isinstance(turn, int):
            self._send_json({"error": "bad_turn",
                             "message": "turn은 정수여야 합니다."}, status=400)
            return
        ok, msg = validate_text(text)
        if not ok:
            err = ("empty_text" if not isinstance(text, str) or not text.strip()
                   else "too_long")
            self._send_json({"error": err, "message": msg}, status=400)
            return
        item = self.state.load_item(did)
        if item is None:
            self._send_json({"error": "unknown_id",
                             "message": "대화를 찾지 못했습니다."}, status=404)
            return
        if _target_pos(item, turn) is None:
            self._send_json({"error": "unknown_turn",
                             "message": "해당 수를 찾지 못했습니다."},
                            status=400)
            return
        try:
            self.state.meter.check()
        except BudgetExceeded:
            self._send_json({"error": "budget_exceeded",
                             "message": BUDGET_MESSAGE}, status=402)
            return
        try:
            result = compare_texts(self.state.client, item, turn, text,
                                   model=self.state.model,
                                   meter=self.state.meter, dialogue_id=did)
        except BudgetExceeded:
            self._send_json({"error": "budget_exceeded",
                             "message": BUDGET_MESSAGE}, status=402)
            return
        except (TypeError, ValueError):
            self._send_json({"error": "judge_failed",
                             "message": "판정에 실패했습니다. 다시 시도해 주세요."},
                            status=500)
            return
        self.state.log_compare({"ts": time.time(), "id": did, "turn": turn,
                                "text": text, "votes": result["votes"],
                                "cost": result["cost_usd"],
                                "model": result["model"],
                                "p_better": result["p_better"],
                                "margin": result["margin"]})
        self._send_json(result)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-play-server")
    ap.add_argument("--records", required=True)
    ap.add_argument("--commentary-dir", "--commentary_dir",
                    dest="commentary_dir", required=True)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--suggests", "--suggestions", dest="suggests",
                    default=None)
    ap.add_argument("--suggests-dir", "--suggests_dir", dest="suggests_dir",
                    default=None)
    ap.add_argument("--culture", default="mix",
                    choices=["east", "west", "mix"])
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
    args = ap.parse_args(argv)
    if not Path(args.records).is_dir():
        raise SystemExit(f"no records dir: {args.records}")
    if not Path(args.commentary_dir).is_dir():
        raise SystemExit(f"no commentary dir: {args.commentary_dir}")

    from .reader import MuseClient

    meter = Meter(budget_usd=args.budget)
    client = MuseClient(meter=meter)
    state = PlayState(records_dir=args.records,
                      commentary_dir=args.commentary_dir, fit_dir=args.fit,
                      model=args.model, meter=meter, log_path=args.log,
                      client=client, scenes=args.scenes,
                      suggests=args.suggests,
                      suggests_dir=args.suggests_dir,
                      culture=args.culture, lang=args.lang)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                             _Handler)
    server.daemon_threads = True
    server.state = state
    print(f"serving play on http://127.0.0.1:{args.port} "
          f"({len(state.dialogues())} dialogues)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def play_main(argv=None) -> None:
    """Static export: one dialogue page with the API URL embedded."""
    ap = argparse.ArgumentParser(prog="uptake-play")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-url", "--api_url", dest="api_url",
                    default="http://127.0.0.1:8765")
    ap.add_argument("--names", default=None,
                    help="JSON object mapping speaker labels to display names")
    ap.add_argument("--cost-hint", "--cost_hint", dest="cost_hint",
                    type=float, default=DEFAULT_COST_HINT)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--suggests", "--suggestions", "--suggest",
                    dest="suggests", default=None)
    ap.add_argument("--culture", default="mix",
                    choices=["east", "west", "mix"])
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
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
    page = render_page(build_dialogue(item, commentary, names=names,
                                      scenes=args.scenes,
                                      suggests=args.suggests,
                                      culture=args.culture, lang=args.lang),
                       api_url=args.api_url, cost_hint=args.cost_hint,
                       lang=args.lang)
    out_path.write_text(page, encoding="utf-8")


export_main = play_main
