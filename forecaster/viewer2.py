"""Viewer v2: the tide (momentum) board with replay and guess mode.

One self-contained HTML page (inline CSS/JS, no external assets except the
optional ``--audio`` file). Input: a board record (any scale, ``fine100``
expected) plus its commentary JSON (task33 format with ``norm.table`` /
``sd_resid``). Two speakers get the red/blue tide; 3+ speakers get stacked
momentum bands.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path

from .board import _value_of
from .commentary import (
    LABEL_EN,
    LABEL_KO,
    pairwise_standing,
    rel_mean_of,
    standing_of,
    turn_values,
)

MARKS = ("!!", "!", "!?", "?", "??")
SPEAKER_COLORS = (
    "#f0606a",  # first speaker (A): red, top in tide mode
    "#7fb3ff",  # second speaker (B): blue, bottom in tide mode
    "#3fb950",
    "#d29922",
    "#bc8cff",
    "#39c5cf",
)
_MOMENTUM_DECAY = 0.55
_M_CLAMP = 3.0
_SQUARE_LO = 20.0
_SQUARE_HI = 80.0


def _num(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def z_received(entry) -> float:
    """Panel surprise behind one turn: z_resid, falling back to z."""
    if not isinstance(entry, dict):
        return 0.0
    if entry.get("z_resid") is not None:
        try:
            return float(entry["z_resid"])
        except (TypeError, ValueError):
            pass
    try:
        return float(entry.get("z", 0.0))
    except (TypeError, ValueError):
        return 0.0


def speakers_of(tv) -> list[str]:
    seen: list[str] = []
    for e in tv or []:
        sp = e.get("speaker", "") if isinstance(e, dict) else ""
        sp = sp if isinstance(sp, str) else str(sp)
        if sp not in seen:
            seen.append(sp)
    return seen


def momentum_series(tv, speakers=None) -> list[float]:
    """Tide momentum: M_t = 0.55·M_{t-1} + s_t.

    s_t is +z_resid for the first (top, red) speaker and −z_resid for the
    other side, so A's well-received move raises M and pushes the boundary
    down into B's base.
    """
    tv = list(tv or [])
    if speakers is None:
        speakers = speakers_of(tv)
    top = speakers[0] if speakers else "A"
    out: list[float] = []
    m = 0.0
    for e in tv:
        if not isinstance(e, dict):
            out.append(m)
            continue
        z = z_received(e)
        s = z if e.get("speaker") == top else -z
        m = _MOMENTUM_DECAY * m + s
        out.append(float(m))
    return out


def clamp_m(m) -> float:
    return max(-_M_CLAMP, min(_M_CLAMP, _num(m)))


def hero_swings(momentum, k=3) -> list[dict]:
    """Turns with the largest |ΔM|, ranked (ΔM_t = M_t − M_{t−1}, M_{−1}=0)."""
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = 3
    kk = max(0, kk)
    moms = [_num(m) for m in (momentum or [])]
    ranked: list[dict] = []
    prev = 0.0
    for pos, m in enumerate(moms):
        ranked.append({"pos": pos, "delta": float(m - prev)})
        prev = m
    ranked.sort(key=lambda r: (-abs(r["delta"]), r["pos"]))
    return ranked[:kk]


def spread_of_effect(effect) -> tuple:
    """Per-turn judge spread: mean reading ``rel_spread``, else disagreement.

    Returns ``(value, source)`` with source one of ``"rel_spread"``,
    ``"disagreement"``, or ``None`` when the effect carries neither field.
    """
    if not isinstance(effect, dict):
        return (None, None)
    vals: list[float] = []
    for r in effect.get("readings", []) or []:
        if isinstance(r, dict) and "rel_spread" in r:
            try:
                vals.append(float(r["rel_spread"]))
            except (TypeError, ValueError):
                pass
    if vals:
        return (sum(vals) / len(vals), "rel_spread")
    if "disagreement" in effect:
        try:
            return (float(effect["disagreement"]), "disagreement")
        except (TypeError, ValueError):
            return (None, None)
    return (None, None)


def resolve_audio_src(audio, out, copy_audio=True) -> str | None:
    """HTML-side audio reference for ``--audio`` (never the raw CLI string).

    With ``copy_audio`` (default) the file is copied next to the HTML as
    ``<out stem>.audio.<ext>`` and referenced relatively; when the file does
    not exist its basename is used so the page still points at a sibling.
    With ``--no-copy-audio`` the reference is ``relpath(audio, out_dir)``.
    """
    if not audio:
        return None
    out_path = Path(str(out))
    out_dir = out_path.parent
    if copy_audio:
        src = Path(str(audio))
        if src.is_file():
            dest = out_dir / (out_path.stem + ".audio" + src.suffix)
            dest.write_bytes(src.read_bytes())
            return os.path.relpath(str(dest), str(out_dir))
        return src.name
    return os.path.relpath(str(audio), str(out_dir))


def _r100(x) -> int:
    """Round a 0–100 standing for display (half up, like JS Math.round)."""
    return int(math.floor(float(x) + 0.5))


def tide_y(m, mid=150.0, amp=40.0) -> float:
    """Boundary y for momentum m (SVG y grows downward)."""
    return float(mid) + clamp_m(m) * float(amp)


def band_series(tv, speakers, alpha=0.5) -> list[dict]:
    """Per-turn stacked-band fractions for 3+ speakers.

    Each speaker's band width is proportional to
    max(0.2, 1 + 0.5·EMA) of that speaker's received z_resid; the EMA only
    moves on that speaker's own turns.
    """
    tv = list(tv or [])
    speakers = list(speakers or [])
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        a = 0.5
    a = max(0.0, min(1.0, a))
    ema = {sp: 0.0 for sp in speakers}
    out: list[dict] = []
    for e in tv:
        sp = e.get("speaker") if isinstance(e, dict) else None
        if sp in ema:
            ema[sp] = (1.0 - a) * ema[sp] + a * z_received(e)
        widths = {s: max(0.2, 1.0 + 0.5 * ema[s]) for s in speakers}
        total = sum(widths.values()) or 1.0
        out.append({s: widths[s] / total for s in speakers})
    return out


def guess_class(z_resid) -> str:
    """Guess-mode answer key: better (!), as expected (=), worse (?)."""
    zr = _num(z_resid)
    if zr >= 1.0:
        return "!"
    if zr <= -1.0:
        return "?"
    return "="


def score_guess(guess, z_resid) -> int:
    """+1 when the player's guess matches the panel's reading."""
    return 1 if guess == guess_class(z_resid) else 0


def raw100_of(effect, scale, standing) -> float:
    """Raw panel reading on 0–100 for hover text.

    For 0–100 scales this is the mean rel_raw; otherwise the standing
    (50 + 25·rel_norm) is the same number on the same scale.
    """
    if scale in ("fine100", "dist5") and isinstance(effect, dict):
        vals: list[float] = []
        for r in effect.get("readings", []) or []:
            if isinstance(r, dict) and "rel_raw" in r:
                try:
                    vals.append(float(r["rel_raw"]))
                except (TypeError, ValueError):
                    pass
        if vals:
            return sum(vals) / len(vals)
    return float(standing)


def _span_of(effect):
    if not isinstance(effect, dict):
        return None
    try:
        start = float(effect.get("start"))
        end = float(effect.get("end"))
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return (start, end)


def _timings_by_index(item, tv, audio_record) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for src in (item, audio_record):
        effects = src.get("effects", []) if isinstance(src, dict) else []
        for pos, e in enumerate(effects or []):
            if not isinstance(e, dict):
                continue
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            if idx in out:
                continue
            span = _span_of(e)
            if span is not None:
                out[idx] = span
    if isinstance(item, dict) and isinstance(item.get("turns"), list):
        for pos, t in zip(range(len(tv)), item["turns"]):
            idx = tv[pos].get("index", pos) if isinstance(tv[pos], dict) else pos
            if idx in out:
                continue
            span = _span_of(t)
            if span is not None:
                out[idx] = span
    return out


def build_view2(item, commentary, outcomes=None, audio_record=None,
                audio_src=None, names=None) -> dict:
    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    norm = commentary.get("norm", {})
    tv = turn_values(item, norm)
    speakers = speakers_of(tv)
    mode = "bands" if len(speakers) >= 3 else "tide"
    scale = item.get("scale", "v2")
    if scale not in ("v2", "fine100", "seven", "dist5", "pairwise"):
        scale = "v2"
    pairwise_view = (scale == "pairwise")
    pw_mu = pw_sd = 0.0
    if pairwise_view:
        nd = norm if isinstance(norm, dict) else {}
        try:
            pw_mu = float(nd.get("mu", 0.0))
        except (TypeError, ValueError):
            pw_mu = 0.0
        try:
            pw_sd = float(nd.get("sd", 0.0))
        except (TypeError, ValueError):
            pw_sd = 0.0
        if nd.get("scale") != "pairwise" or not pw_sd:
            raw_vals: list[float] = []
            for e in (item.get("effects", []) or []):
                if not isinstance(e, dict):
                    continue
                try:
                    raw_vals.append(float(_value_of(e)))
                except (TypeError, ValueError):
                    continue
            if raw_vals:
                pw_mu = sum(raw_vals) / len(raw_vals)
                var = sum((v - pw_mu) ** 2 for v in raw_vals) / len(raw_vals)
                pw_sd = var ** 0.5 if var > 0 else 0.0
    eff_by_index: dict[int, dict] = {}
    for pos, e in enumerate(item.get("effects", []) or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) else pos
        if idx not in eff_by_index:
            eff_by_index[idx] = e
    pivots = commentary.get("turns", [])
    if not isinstance(pivots, list):
        pivots = []
    pivot_by_index: dict[int, dict] = {}
    for p in pivots:
        if not isinstance(p, dict):
            continue
        idx = p.get("index")
        if isinstance(idx, int) and idx not in pivot_by_index:
            pivot_by_index[idx] = p
    timings = _timings_by_index(item, tv, audio_record)
    momentum = momentum_series(tv, speakers)
    bands = band_series(tv, speakers) if mode == "bands" else None
    names = names if isinstance(names, dict) else {}
    colors = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
              for i, sp in enumerate(speakers)}
    last = {sp: 50.0 for sp in speakers}
    trail: list[dict] = []
    turns_out: list[dict] = []
    spreads: list = []
    spread_srcs: list = []
    for pos, e in enumerate(tv):
        idx = e.get("index", pos)
        eff = eff_by_index.get(idx, {})
        if pairwise_view:
            try:
                pw_raw = float(_value_of(eff))
            except (TypeError, ValueError):
                pw_raw = 0.0
            standing = pairwise_standing(pw_raw, pw_mu, pw_sd)
        else:
            pw_raw = None
            standing = standing_of(rel_mean_of(eff))
        if e.get("speaker") in last:
            last[e["speaker"]] = standing
        trail.append({sp: float(last[sp]) for sp in speakers})
        piv = pivot_by_index.get(idx, {})
        mark = piv.get("mark") if piv.get("mark") in MARKS else None
        span = timings.get(idx)
        spread_val, spread_src = spread_of_effect(eff)
        spreads.append(float(spread_val) if spread_val is not None else None)
        spread_srcs.append(spread_src)
        rec = {
            "index": idx,
            "speaker": e.get("speaker", ""),
            "display": names.get(e.get("speaker"), e.get("speaker", "")),
            "spread": float(spread_val) if spread_val is not None else None,
            "spread_src": spread_src,
            "text": e.get("text", ""),
            "moves": list(e.get("moves", []) or []),
            "value": float(e.get("value", 0.0)),
            "z": float(e.get("z", 0.0)),
            "raw100": float(raw100_of(eff, scale, standing)),
            "standing": float(standing),
            "guess": guess_class(e.get("z_resid", e.get("z", 0.0))),
            "mark": mark,
            "label_ko": piv.get("label_ko"),
            "label_en": piv.get("label_en"),
            "line_ko": piv.get("line_ko"),
            "line_en": piv.get("line_en"),
            "start": span[0] if span else None,
            "end": span[1] if span else None,
        }
        if e.get("resid") is not None:
            rec["resid"] = float(e["resid"])
        if e.get("z_resid") is not None:
            rec["z_resid"] = float(e["z_resid"])
        if pairwise_view and pw_raw is not None:
            rec["rating"] = float(pw_raw)
            rec["rating_label"] = "쌍대 평점"
        turns_out.append(rec)
    if len(speakers) >= 2:
        x_sp, y_sp = speakers[0], speakers[1]
        xs = [t[x_sp] for t in trail]
        ys = [t[y_sp] for t in trail]
    elif len(speakers) == 1:
        x_sp, y_sp = speakers[0], ""
        xs = [t[x_sp] for t in trail]
        ys = [50.0 for t in trail]
    else:
        x_sp, y_sp, xs, ys = "", "", [], []
    if len(speakers) >= 2:
        settlement = (f"{x_sp} 잔고 50 → {_r100(xs[-1])} · "
                      f"{y_sp} 잔고 50 → {_r100(ys[-1])}" if xs else "")
    elif len(speakers) == 1:
        settlement = f"{x_sp} 잔고 50 → {_r100(xs[-1])}" if xs else ""
    else:
        settlement = ""
    item_id = item.get("id", "")
    if isinstance(item_id, str) and item_id:
        title = "Dialogue " + item_id + " — tide board"
    elif item_id:
        title = "Dialogue " + str(item_id) + " — tide board"
    else:
        title = "Dialogue — tide board"
    audio = None
    if audio_src:
        audio = {"src": str(audio_src), "linked": bool(timings)}
    if "rel_spread" in spread_srcs:
        band_src = "rel_spread"
    elif "disagreement" in spread_srcs:
        band_src = "disagreement"
    else:
        band_src = None
    swings = hero_swings(momentum, 3)
    hero_top = None
    runners: list[dict] = []
    for rank, sw in enumerate(swings):
        pos = sw["pos"]
        if pos < 0 or pos >= len(turns_out):
            continue
        t = turns_out[pos]
        entry = {"pos": pos, "index": t["index"], "speaker": t["speaker"],
                 "display": t["display"], "delta": sw["delta"]}
        if rank == 0:
            entry["text80"] = (t["text"] or "")[:80]
            entry["line"] = t.get("line_ko") or t.get("line_en") or None
            hero_top = entry
        else:
            runners.append(entry)
    return {
        "id": item_id,
        "title": title,
        "scale": scale,
        "mode": mode,
        "speakers": speakers,
        "colors": colors,
        "display_names": {sp: names.get(sp, sp) for sp in speakers},
        "momentum": momentum,
        "hero": {"top": hero_top, "runners": runners},
        "spread": spreads,
        "spread_src": band_src,
        "standings": trail,
        "bands": bands,
        "square": {"x_sp": x_sp, "y_sp": y_sp, "xs": xs, "ys": ys,
                   "lo": _SQUARE_LO, "hi": _SQUARE_HI},
        "settlement": settlement,
        "turns": turns_out,
        "outcomes": outcomes if outcomes is not None else None,
        "audio": audio,
    }


_CSS = """
:root{
  --bg:#0d1117; --fg:#e6edf3; --card:#161b22; --muted:#9aa4b2; --border:#30363d;
  --a:#f0606a; --b:#7fb3ff; --good:#3fb950; --bad:#f0606a; --mid:#d29922;
  --accent:#7fb3ff;
}
*{box-sizing:border-box}
body{max-width:1100px;margin:0 auto;padding:0 16px;background:var(--bg);color:var(--fg);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;line-height:1.55}
header{padding:20px 0 8px}
h1{font-size:22px;margin:0 0 4px;overflow-wrap:break-word}
h2{font-size:16px;margin:0 0 10px}
.note{color:var(--muted);font-size:13px;margin:0 0 10px}
.layout{display:flex;flex-direction:column;gap:14px;padding-bottom:28px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px;min-width:0;overflow:hidden}
#tide{width:100%;height:auto;display:block;max-width:100%}
.sq-wrap{overflow-x:auto;max-width:100%}
#square{display:block}
.sq-axis{fill:var(--muted);font-size:13px}
.tide-a{fill:var(--a);opacity:.55}
.tide-b{fill:var(--b);opacity:.55}
.tide-line{fill:none;stroke:var(--fg);stroke-width:1.5;opacity:.8}
.band{stroke:none;opacity:.8}
.mid{stroke:#4a525c;stroke-width:1;stroke-dasharray:4 3}
.axis-tx{fill:var(--muted);font-size:12px}
.dot{stroke:var(--bg);stroke-width:1.5;cursor:default}
.mark-g{font-size:14px;font-weight:700;fill:var(--fg);paint-order:stroke;stroke:var(--bg);stroke-width:3px}
.sq-trail{fill:none;stroke:var(--muted);stroke-width:1.5;stroke-dasharray:2 3}
.sq-dot{fill:var(--muted)}
.sq-last{fill:var(--accent);stroke:var(--fg);stroke-width:1.5}
.sq-now{fill:none;stroke:#fff;stroke-width:2}
.sq-grid{stroke:#21262d;stroke-width:1}
.sq-corner{fill:var(--muted);font-size:13px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
.controls button{border:1px solid var(--border);background:var(--bg);color:var(--fg);
  border-radius:8px;padding:4px 12px;font-size:13px;cursor:pointer}
.controls button.active{background:var(--accent);border-color:var(--accent);color:#0d1117}
.controls button:disabled{opacity:.4;cursor:default}
#scrub{flex:1 1 160px;min-width:120px}
#tally{font-size:13px;color:var(--muted)}
#stands{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;font-size:14px;font-weight:700}
.stand{border:1px solid var(--border);border-radius:8px;padding:2px 10px}
#turns{display:flex;flex-direction:column;gap:8px;max-height:420px;overflow-y:auto}
.t2turn{border:1px solid var(--border);border-radius:10px;padding:8px 10px;min-width:0}
.t2turn.current{outline:2px solid var(--accent)}
.t2head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.sp{font-weight:700}
.raw{color:var(--muted);font-size:12px;margin-left:auto}
.t2text{margin:4px 0;overflow-wrap:break-word;word-break:break-word;font-size:14px}
#comment{margin:0}
#cmt-list{display:flex;flex-direction:column;gap:8px}
.cmt{border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.cmt-line{margin:4px 0;overflow-wrap:break-word}
.cmt-en{color:var(--muted);font-size:13px}
.badge{display:inline-block;min-width:1.7em;text-align:center;border-radius:6px;
  padding:0 6px;font-weight:700;font-size:13px;border:1px solid var(--border)}
.badge[data-mark="!"]{background:#122619;color:var(--good);border-color:var(--good)}
.badge[data-mark="?"]{background:#2c1518;color:var(--bad);border-color:var(--bad)}
.badge[data-mark="!?"]{background:#2b230a;color:var(--mid);border-color:var(--mid)}
.badge[data-mark="!!"]{background:var(--good);color:#fff;border-color:var(--good)}
.badge[data-mark="??"]{background:var(--bad);color:#fff;border-color:var(--bad)}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
.lg{display:inline-flex;align-items:center;gap:5px}
#settle{font-size:14px;font-weight:700;margin:8px 0 2px}
#likeness{color:var(--muted);font-size:13px;margin:0 0 4px}
#guessCard{border-color:var(--mid)}
#guessText{overflow-wrap:break-word}
.gbtn{font-size:18px;font-weight:700;min-width:3em}
#guessResult{color:var(--muted);font-size:13px}
audio{width:100%;margin-top:10px}
.muted{color:var(--muted)}
.conf-band{fill:#fff;opacity:.14;stroke:none}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px}
#hero{border:1px solid var(--border);border-radius:10px;padding:8px 10px;
  margin-bottom:10px;transition:border-color .3s ease,background-color .3s ease}
#hero.reached{border-color:var(--accent);background:#16202e}
.hero-main{font-size:14px;overflow-wrap:break-word}
.hero-line{font-size:13px;color:var(--muted);margin-top:2px;overflow-wrap:break-word}
.hero-sub{font-size:12px;color:var(--muted);margin-top:2px}
.dot.auto-hide{opacity:0;transition:opacity .2s ease}
#tide:hover .dot.auto-hide{opacity:1}
.pulse{fill:none;stroke:var(--accent);stroke-width:2}
@media (prefers-reduced-motion:reduce){
  #hero{transition:none}
  .dot.auto-hide{transition:none}
}
"""

_JS = """(function(){
"use strict";
var data=JSON.parse(document.getElementById("data").textContent);
var turns=data.turns||[];
var N=turns.length;
var NS=document.getElementById("tide").namespaceURI;
var shown=N;
var playing=false;
var speed=1;
var timer=null;
var lastTick=0;
var guessOn=false;
var gIdx=0,gHits=0;
var RATE=1/0.3;
var REDUCED=(typeof matchMedia==="function")&&
  matchMedia("(prefers-reduced-motion: reduce)").matches;
var markBorn={};
var markAnimating=false;
var markAnimQueued=false;
var pulseT=-1e9;
function $(id){return document.getElementById(id);}
function el(tag,cls,text){
  var e=document.createElement(tag);
  if(cls)e.className=cls;
  if(text!==null&&text!==undefined)e.textContent=text;
  return e;
}
function clampM(m){return Math.max(-3,Math.min(3,m));}
function smoothstep(f){f=Math.max(0,Math.min(1,f));return f*f*(3-2*f);}
function svgEl(tag){
  return document.createElementNS(NS,tag);
}
function smoothD(pts){
  if(!pts.length)return "";
  if(pts.length===1)return "M"+pts[0][0].toFixed(1)+","+pts[0][1].toFixed(1);
  if(pts.length===2){
    return "M"+pts[0][0].toFixed(1)+","+pts[0][1].toFixed(1)+
      "L"+pts[1][0].toFixed(1)+","+pts[1][1].toFixed(1);
  }
  var d="M"+pts[0][0].toFixed(1)+","+pts[0][1].toFixed(1);
  for(var i=0;i<pts.length-1;i++){
    var p0=pts[Math.max(0,i-1)],p1=pts[i],p2=pts[i+1];
    var p3=pts[Math.min(pts.length-1,i+2)];
    var c1x=p1[0]+(p2[0]-p0[0])/6,c1y=p1[1]+(p2[1]-p0[1])/6;
    var c2x=p2[0]-(p3[0]-p1[0])/6,c2y=p2[1]-(p3[1]-p1[1])/6;
    d+="C"+c1x.toFixed(1)+","+c1y.toFixed(1)+" "+
      c2x.toFixed(1)+","+c2y.toFixed(1)+" "+
      p2[0].toFixed(1)+","+p2[1].toFixed(1);
  }
  return d;
}
function tipText(t){
  var s="#"+t.index+" "+(t.display||t.speaker);
  if(t.rating_label&&t.rating!==null&&t.rating!==undefined){
    s+=" · "+t.rating_label+" "+Number(t.rating).toFixed(2);
  }else{
    s+=" · panel "+Math.round(t.raw100);
  }
  if(t.spread!==null&&t.spread!==undefined){
    var tsrc=t.spread_src||data.spread_src;
    if(tsrc==="disagreement")s+=" · 확신도: 심사위원 의견 불일치 "+
      Number(t.spread).toFixed(2);
    else s+=" · 확신도: 심사위원 분포 퍼짐 "+Math.round(t.spread);
  }
  s+=" · "+(t.text||"");
  var line=t.line_ko||t.line_en||"";
  if(line)s+=" · "+line;
  if(s.length>220)s=s.slice(0,217)+"…";
  return s;
}
function heroPos(){
  if(data.hero&&data.hero.top&&typeof data.hero.top.pos==="number")
    return data.hero.top.pos;
  return -1;
}
function easeOutBack(t){
  t=Math.max(0,Math.min(1,t));
  var c1=1.70158,c3=c1+1;
  return 1+c3*Math.pow(t-1,3)+c1*Math.pow(t-1,2);
}
function markScale(age){
  if(!(age>=0)||age>=350)return 1;
  return easeOutBack(age/350);
}
function queueMarkFrame(){
  if(REDUCED||markAnimQueued)return;
  markAnimQueued=true;
  var fire=function(){markAnimQueued=false;renderTide();};
  if(typeof requestAnimationFrame==="function")requestAnimationFrame(fire);
  else setTimeout(fire,50);
}
function finishMarks(){
  if(markAnimating)queueMarkFrame();
}
function spreadPx(sp,src){
  if(sp===null||sp===undefined)return 0;
  var v=Number(sp);
  if(isNaN(v))return 0;
  if(src==="disagreement"){
    var d=Math.max(0,Math.min(1,v));
    return 2+12*d;
  }
  if(v<=10)return 2;
  if(v>=25)return 14;
  return 2+(v-10)*12/15;
}
function bandD(pts){
  if(!pts.length||!data.spread_src)return "";
  var k=Math.max(0,Math.min(N,shown));
  var idx=Math.floor(k),fr=smoothstep(k-idx);
  var up=Math.min(idx,N);
  var hw=[];
  for(var i=0;i<up;i++)
    hw.push(spreadPx(turns[i].spread,turns[i].spread_src||data.spread_src));
  if(idx<N){
    var hN=spreadPx(turns[idx].spread,turns[idx].spread_src||data.spread_src);
    var hP=(idx>0)?
      spreadPx(turns[idx-1].spread,turns[idx-1].spread_src||data.spread_src):hN;
    hw.push(hP+(hN-hP)*fr);
  }
  var any=false;
  for(var j=0;j<hw.length;j++)if(hw[j]>0)any=true;
  if(!any||hw.length<2||pts.length<2)return "";
  var top=[],bot=[];
  for(var p=0;p<pts.length;p++){
    top.push([pts[p][0],pts[p][1]-hw[p]]);
    bot.push([pts[p][0],pts[p][1]+hw[p]]);
  }
  var rev=bot.slice().reverse();
  return smoothD(top)+"L"+rev[0][0].toFixed(1)+","+rev[0][1].toFixed(1)+
    smoothD(rev).slice(1)+"Z";
}
var TW=640,TH=300,padL=46,padR=14,padT=26,padB=22;
var tMid=(padT+TH-padB)/2,tAmp=(TH-padT-padB)/2/3.2;
function tX(i){
  if(N<=1)return padL+(TW-padL-padR)/2;
  return padL+(i/(N-1))*(TW-padL-padR);
}
function tY(m){return tMid+clampM(m)*tAmp;}
function tidePts(){
  var M=data.momentum||[];
  var pts=[];
  var k=Math.max(0,Math.min(N,shown));
  var idx=Math.floor(k),fr=smoothstep(k-idx);
  var up=Math.min(idx,N);
  for(var i=0;i<up;i++)pts.push([tX(i),tY(M[i]||0)]);
  if(idx<N){
    var x0=tX(0),y0=tMid;
    if(idx>0){x0=tX(idx-1);y0=tY(M[idx-1]||0);}
    var x1=tX(idx),y1=tY(M[idx]||0);
    pts.push([x0+(x1-x0)*fr,y0+(y1-y0)*fr]);
  }
  return pts;
}
function renderTide(){
  var svg=$("tide");
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  svg.setAttribute("viewBox","0 0 "+TW+" "+TH);
  svg.setAttribute("role","img");
  svg.setAttribute("aria-label","tide chart");
  var zl=svgEl("line");
  zl.setAttribute("x1",padL);zl.setAttribute("x2",TW-padR);
  zl.setAttribute("y1",tMid);zl.setAttribute("y2",tMid);
  zl.setAttribute("class","mid");
  svg.appendChild(zl);
  if(!N)return;
  for(var mk in markBorn){if(Number(mk)>=shown)delete markBorn[mk];}
  markAnimating=false;
  if(data.mode==="bands"){renderBands(svg);finishMarks();return;}
  var pts=tidePts();
  if(pts.length>=2){
    var d=smoothD(pts);
    var x0=pts[0][0].toFixed(1),x1=pts[pts.length-1][0].toFixed(1);
    var top=svgEl("path");
    top.setAttribute("d",d+"L"+x1+","+padT+"L"+x0+","+padT+"Z");
    top.setAttribute("class","tide-a");
    svg.appendChild(top);
    var bot=svgEl("path");
    bot.setAttribute("d",d+"L"+x1+","+(TH-padB)+"L"+x0+","+(TH-padB)+"Z");
    bot.setAttribute("class","tide-b");
    svg.appendChild(bot);
    var bd=bandD(pts);
    if(bd){
      var band=svgEl("path");
      band.setAttribute("d",bd);
      band.setAttribute("class","conf-band");
      svg.appendChild(band);
    }
    var line=svgEl("path");
    line.setAttribute("d",d);
    line.setAttribute("class","tide-line");
    svg.appendChild(line);
  }
  var M=data.momentum||[];
  for(var i=0;i<N&&i<shown;i++){
    addDot(svg,turns[i],tX(i),tY(M[i]||0),i);
  }
  addXLabels(svg);
  for(var m2=0;m2<N&&m2<shown;m2++){
    if(turns[m2].mark)addMark(svg,turns[m2],tX(m2),tY(M[m2]||0),m2);
  }
  finishMarks();
}
function addDot(svg,t,cx,cy,pos){
  var now=(typeof performance!=="undefined"&&performance.now)?
    performance.now():0;
  var hp=heroPos();
  var c=svgEl("circle");
  c.setAttribute("cx",cx.toFixed(1));c.setAttribute("cy",cy.toFixed(1));
  c.setAttribute("r",t.mark?"6":"4");
  c.setAttribute("class",(!t.mark&&N>40)?"dot auto-hide":"dot");
  c.style.fill=data.colors[t.speaker]||"#999";
  var tip=svgEl("title");
  tip.textContent=tipText(t);
  c.appendChild(tip);
  svg.appendChild(c);
  if(pos===hp&&hp>=0&&!REDUCED){
    var age=now-pulseT;
    if(age>=0&&age<700){
      var pr=age/700;
      var ring=svgEl("circle");
      ring.setAttribute("cx",cx.toFixed(1));ring.setAttribute("cy",cy.toFixed(1));
      ring.setAttribute("r",(6+12*pr).toFixed(1));
      ring.setAttribute("class","pulse");
      ring.style.opacity=((1-pr)*0.9).toFixed(2);
      svg.appendChild(ring);
    }
  }
}
function addMark(svg,t,cx,cy,pos){
  if(!t.mark)return;
  var g=svgEl("text");
  g.setAttribute("x",cx.toFixed(1));g.setAttribute("y",(cy-10).toFixed(1));
  g.setAttribute("text-anchor","middle");
  g.setAttribute("class","mark-g");
  g.textContent=t.mark;
  if(!REDUCED&&(typeof performance!=="undefined"&&performance.now)){
    var now=performance.now();
    if(markBorn[pos]===undefined)markBorn[pos]=now;
    var ma=now-markBorn[pos];
    var s=markScale(ma);
    if(ma>=0&&ma<350&&s!==1){
      var gx=cx.toFixed(1),gy=(cy-10).toFixed(1);
      g.setAttribute("transform","translate("+gx+" "+gy+") scale("+
        s.toFixed(3)+") translate("+(-cx).toFixed(1)+" "+
        (-(cy-10)).toFixed(1)+")");
      markAnimating=true;
    }
  }
  svg.appendChild(g);
}
function addXLabels(svg){
  var step=Math.max(1,Math.floor(N/8));
  for(var i=0;i<N;i+=step){
    var lab=svgEl("text");
    lab.setAttribute("x",tX(i));lab.setAttribute("y",TH-6);
    lab.setAttribute("text-anchor","middle");
    lab.setAttribute("class","axis-tx");
    lab.textContent=""+turns[i].index;
    svg.appendChild(lab);
  }
}
function cumAt(fr,S,t){
  var f=fr[t]||{};
  var ys=[padT],acc=0;
  for(var s=0;s<S.length;s++){acc+=(f[S[s]]||0);ys.push(padT+acc*(TH-padT-padB));}
  return ys;
}
function renderBands(svg){
  var fr=data.bands||[];
  var S=data.speakers||[];
  if(!S.length)return;
  var k=Math.max(0,Math.min(N,shown));
  var idx=Math.floor(k),fr2=smoothstep(k-idx);
  var up=Math.min(idx,N);
  var nb=S.length+1,b,bi,xs=[];
  for(bi=0;bi<nb;bi++)xs.push([]);
  var even=[];
  for(b=0;b<nb;b++)even.push(padT+b*(TH-padT-padB)/S.length);
  for(var i=0;i<up;i++){
    var ys=cumAt(fr,S,i);
    for(b=0;b<nb;b++)xs[b].push([tX(i),ys[b]]);
  }
  if(idx<N){
    var prev=(idx>0)?cumAt(fr,S,idx-1):even;
    var next=cumAt(fr,S,idx);
    var x0=(idx>0)?tX(idx-1):tX(0),x1=tX(idx);
    for(b=0;b<nb;b++){
      xs[b].push([x0+(x1-x0)*fr2,prev[b]+(next[b]-prev[b])*fr2]);
    }
  }
  for(var s=0;s<S.length;s++){
    var top=xs[s],bot=xs[s+1].slice().reverse();
    if(top.length<1)continue;
    var p=svgEl("path");
    var dF=smoothD(top);
    var dB=smoothD(bot);
    var d=dF+"L"+bot[0][0].toFixed(1)+","+bot[0][1].toFixed(1)+
      dB.slice(1)+"Z";
    if(top.length===1){
      d="M"+top[0][0].toFixed(1)+","+top[0][1].toFixed(1)+
        "L"+bot[0][0].toFixed(1)+","+bot[0][1].toFixed(1)+"Z";
    }
    p.setAttribute("d",d);
    p.setAttribute("class","band");
    p.style.fill=data.colors[S[s]]||"#999";
    var tip=svgEl("title");
    tip.textContent=S[s];
    p.appendChild(tip);
    svg.appendChild(p);
  }
  for(var j=0;j<N&&j<shown;j++){
    var ys2=cumAt(fr,S,j);
    var si=S.indexOf(turns[j].speaker);
    if(si<0)si=0;
    addDot(svg,turns[j],tX(j),(ys2[si]+ys2[si+1])/2,j);
  }
  addXLabels(svg);
  for(var k2=0;k2<N&&k2<shown;k2++){
    if(!turns[k2].mark)continue;
    var ys3=cumAt(fr,S,k2);
    var s3=S.indexOf(turns[k2].speaker);
    if(s3<0)s3=0;
    addMark(svg,turns[k2],tX(k2),(ys3[s3]+ys3[s3+1])/2,k2);
  }
}
function curStands(){
  var S=data.speakers||[];
  var base={};
  for(var s=0;s<S.length;s++)base[S[s]]=50;
  var trail=data.standings||[];
  var i=Math.min(trail.length,Math.ceil(shown))-1;
  if(i>=0&&trail[i])return trail[i];
  return base;
}
function renderStands(){
  var box=$("stands");
  while(box.firstChild)box.removeChild(box.firstChild);
  var cur=curStands();
  var S=data.speakers||[];
  for(var i=0;i<S.length;i++){
    var v=(cur[S[i]]!==undefined)?cur[S[i]]:50;
    var d=el("span","stand",S[i]+" · 잔고 "+Math.round(v));
    d.style.borderColor=data.colors[S[i]]||"var(--border)";
    box.appendChild(d);
  }
}
var SW=360,SH=320,sqL=40,sqR=16,sqT=34,sqB=54;
function sqX(v,lo,hi){return sqL+(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*(SW-sqL-sqR);}
function sqY(v,lo,hi){return sqT+(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo))*(SH-sqT-sqB);}
function renderSquare(){
  var svg=$("square");
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  svg.setAttribute("viewBox","0 0 "+SW+" "+SH);
  svg.setAttribute("width",SW);svg.setAttribute("height",SH);
  svg.setAttribute("role","img");
  svg.setAttribute("aria-label","standing square");
  var sq=data.square||{};
  var lo=(sq.lo!==undefined)?sq.lo:20,hi=(sq.hi!==undefined)?sq.hi:80;
  var grid=[20,50,80];
  for(var g=0;g<grid.length;g++){
    var gl=svgEl("line");
    gl.setAttribute("x1",sqL);gl.setAttribute("x2",SW-sqR);
    gl.setAttribute("y1",sqY(grid[g],lo,hi));gl.setAttribute("y2",sqY(grid[g],lo,hi));
    gl.setAttribute("class","sq-grid");
    svg.appendChild(gl);
    var gl2=svgEl("line");
    gl2.setAttribute("x1",sqX(grid[g],lo,hi));gl2.setAttribute("x2",sqX(grid[g],lo,hi));
    gl2.setAttribute("y1",sqT);gl2.setAttribute("y2",SH-sqB);
    gl2.setAttribute("class","sq-grid");
    svg.appendChild(gl2);
    var lx=svgEl("text");
    lx.setAttribute("x",sqX(grid[g],lo,hi));lx.setAttribute("y",SH-sqB+16);
    lx.setAttribute("text-anchor","middle");
    lx.setAttribute("class","axis-tx");
    lx.textContent=""+grid[g];
    svg.appendChild(lx);
    var ly=svgEl("text");
    ly.setAttribute("x",sqL-5);ly.setAttribute("y",sqY(grid[g],lo,hi)+3);
    ly.setAttribute("text-anchor","end");
    ly.setAttribute("class","axis-tx");
    ly.textContent=""+grid[g];
    svg.appendChild(ly);
  }
  var corners=[
    [SW-sqR-4,sqT+12,"end","둘 다 잘됨"],
    [sqL+4,SH-sqB-6,"start","둘 다 잃음"],
    [sqL+4,sqT+12,"start","한쪽만"],
    [SW-sqR-4,SH-sqB-6,"end","한쪽만"]
  ];
  for(var c=0;c<corners.length;c++){
    var ct=svgEl("text");
    ct.setAttribute("x",corners[c][0]);ct.setAttribute("y",corners[c][1]);
    ct.setAttribute("text-anchor",corners[c][2]);
    ct.setAttribute("class","sq-corner");
    ct.textContent=corners[c][3];
    svg.appendChild(ct);
  }
  var xs=sq.xs||[],ys=sq.ys||[];
  var n=Math.min(xs.length,ys.length);
  if(!n)return;
  var xl=svgEl("text");
  xl.setAttribute("x",(sqL+SW-sqR)/2);xl.setAttribute("y",SH-12);
  xl.setAttribute("text-anchor","middle");
  xl.setAttribute("class","sq-axis");
  xl.textContent=(sq.x_sp||"A")+" 잔고 →";
  svg.appendChild(xl);
  var yl=svgEl("text");
  yl.setAttribute("x",sqL);yl.setAttribute("y",16);
  yl.setAttribute("text-anchor","start");
  yl.setAttribute("class","sq-axis");
  yl.textContent=(sq.y_sp||"B")+" 잔고 ↑";
  svg.appendChild(yl);
  var pts=[];
  for(var i=0;i<n;i++)pts.push([sqX(xs[i],lo,hi),sqY(ys[i],lo,hi)]);
  var trail=svgEl("path");
  trail.setAttribute("d",smoothD(pts));
  trail.setAttribute("class","sq-trail");
  svg.appendChild(trail);
  for(var j=0;j<n-1;j++){
    var q=svgEl("circle");
    q.setAttribute("cx",sqX(xs[j],lo,hi));q.setAttribute("cy",sqY(ys[j],lo,hi));
    q.setAttribute("r","2.5");
    q.setAttribute("class","sq-dot");
    svg.appendChild(q);
  }
  var last=svgEl("circle");
  last.setAttribute("cx",sqX(xs[n-1],lo,hi));
  last.setAttribute("cy",sqY(ys[n-1],lo,hi));
  last.setAttribute("r","7");
  last.setAttribute("class","sq-last");
  var lt=svgEl("title");
  lt.textContent=(sq.x_sp||"A")+" "+Math.round(xs[n-1])+" · "+
    (sq.y_sp||"B")+" "+Math.round(ys[n-1]);
  last.appendChild(lt);
  svg.appendChild(last);
  var ci=Math.min(n,Math.ceil(shown))-1;
  if(ci>=0&&ci<n-1){
    var now=svgEl("circle");
    now.setAttribute("cx",sqX(xs[ci],lo,hi));
    now.setAttribute("cy",sqY(ys[ci],lo,hi));
    now.setAttribute("r","5");
    now.setAttribute("class","sq-now");
    svg.appendChild(now);
  }
}
function renderTurns(){
  var box=$("turns");
  while(box.firstChild)box.removeChild(box.firstChild);
  for(var i=0;i<N;i++){
    var t=turns[i];
    var d=el("div","t2turn");
    d.setAttribute("id","t2-"+i);
    var head=el("div","t2head");
    head.appendChild(el("span","turn-idx","#"+t.index));
    var sp=el("span","sp",t.display||t.speaker);
    sp.style.color=data.colors[t.speaker]||"var(--fg)";
    head.appendChild(sp);
    if(t.mark){
      var b=el("span","badge",t.mark);
      b.setAttribute("data-mark",t.mark);
      head.appendChild(b);
    }
    head.appendChild(el("span","raw","panel "+Math.round(t.raw100)));
    d.appendChild(head);
    d.appendChild(el("p","t2text",t.text||""));
    box.appendChild(d);
  }
}
function updateTurns(scroll){
  var cur=Math.ceil(shown)-1;
  for(var i=0;i<N;i++){
    var d=$("t2-"+i);
    if(!d)continue;
    if(i===cur)d.classList.add("current");
    else d.classList.remove("current");
  }
  if(scroll&&cur>=0){
    var tgt=$("t2-"+cur);
    if(tgt)tgt.scrollIntoView({block:"nearest"});
  }
}
function renderComments(){
  var box=$("cmt-list");
  while(box.firstChild)box.removeChild(box.firstChild);
  for(var i=0;i<N;i++){
    var t=turns[i];
    if(!t.line_ko&&!t.line_en)continue;
    var c=el("article","cmt");
    c.setAttribute("id","c2-"+i);
    var top=el("div","t2head");
    if(t.mark){
      var b=el("span","badge",t.mark);
      b.setAttribute("data-mark",t.mark);
      top.appendChild(b);
    }
    top.appendChild(el("span","sp","#"+t.index+" "+(t.display||t.speaker)));
    c.appendChild(top);
    if(t.line_ko)c.appendChild(el("p","cmt-line",t.line_ko));
    if(t.line_en)c.appendChild(el("p","cmt-line cmt-en",t.line_en));
    c.style.display="none";
    box.appendChild(c);
  }
  if(!box.firstChild)box.appendChild(el("p","muted","결정적 장면 없음"));
}
function updateComments(){
  for(var i=0;i<N;i++){
    var c=$("c2-"+i);
    if(c)c.style.display=(i<shown)?"":"none";
  }
}
function fmtS(v){
  if(v===null||v===undefined)return "–";
  var n=Math.round(v);
  return (n>=0?"+":"")+n;
}
function renderSettle(){
  $("settle").textContent=data.settlement||"";
  var oc=data.outcomes,tx="";
  if(oc&&oc.A&&oc.B){
    tx="actual likeness →"+(data.square.x_sp||"A")+" "+fmtS(oc.B.likeness)+
      " (rated by "+(data.square.y_sp||"B")+") · →"+
      (data.square.y_sp||"B")+" "+fmtS(oc.A.likeness)+
      " (rated by "+(data.square.x_sp||"A")+")";
  }
  $("likeness").textContent=tx;
}
function audioLinked(){
  return !!(data.audio&&data.audio.src&&data.audio.linked);
}
function turnAtTime(t){
  var k=0;
  for(var i=0;i<N;i++){
    var s=turns[i].start;
    if(s!==null&&s!==undefined&&s<=t)k=i+1;
    else break;
  }
  return k;
}
function render(scroll){
  renderTide();
  renderSquare();
  renderStands();
  updateTurns(scroll);
  updateComments();
  var hb=$("hero");
  if(hb){var hp=heroPos();hb.classList.toggle("reached",hp>=0&&shown>hp);}
  var sc=$("scrub");
  if(sc&&document.activeElement!==sc)sc.value=shown;
}
function tick(){
  var now=performance.now();
  if(REDUCED){
    if(now-lastTick<300)return;
    lastTick=now;
    shown=Math.min(N,Math.floor(shown)+1);
    if(shown>=N){shown=N;pause();}
    render(true);
    return;
  }
  var dt=(now-lastTick)/1000;
  lastTick=now;
  shown+=dt*speed*RATE;
  if(shown>=N){shown=N;pause();}
  render(true);
}
function play(){
  if(guessOn||playing)return;
  if(shown>=N)shown=0;
  playing=true;
  $("play").textContent="❚❚";
  $("play").classList.add("active");
  if(audioLinked()){
    var au=$("aud");
    if(au){try{au.play();}catch(e){}}
  }else{
    lastTick=performance.now();
    timer=setInterval(tick,50);
  }
}
function pause(){
  playing=false;
  $("play").textContent="▶";
  $("play").classList.remove("active");
  if(timer){clearInterval(timer);timer=null;}
  if(audioLinked()){
    var au=$("aud");
    if(au){try{au.pause();}catch(e){}}
  }
}
function togglePlay(){
  if(playing)pause();
  else play();
}
function guessClassJS(z){
  if(z>=1)return "!";
  if(z<=-1)return "?";
  return "=";
}
function setGuess(on){
  guessOn=on;
  var btn=$("guessToggle");
  btn.classList.toggle("active",on);
  btn.setAttribute("aria-pressed",on?"true":"false");
  $("guessCard").style.display=on?"":"none";
  $("play").disabled=on;
  $("scrub").disabled=on;
  var sp=document.querySelectorAll(".spdbtn");
  for(var i=0;i<sp.length;i++)sp[i].disabled=on;
  if(on){
    pause();
    gIdx=0;gHits=0;shown=0;
    $("tally").textContent="나 0 / 해설자 0";
    render(false);
    renderGuess();
  }else{
    shown=N;
    render(false);
  }
}
function renderGuess(){
  if(!guessOn)return;
  var txt=$("guessText"),res=$("guessResult");
  var btns=document.querySelectorAll(".gbtn");
  if(gIdx>=N){
    txt.textContent="끝! 전 턴을 맞혀봤습니다.";
    res.textContent="최종: 나 "+gHits+" / 해설자 "+N;
    for(var i=0;i<btns.length;i++)btns[i].disabled=true;
    return;
  }
  for(var j=0;j<btns.length;j++)btns[j].disabled=false;
  var t=turns[gIdx];
  txt.textContent="#"+t.index+" "+(t.display||t.speaker)+": "+(t.text||"");
  res.textContent="이 수가 평소보다 나을지, 비슷할지, 못할지 맞혀보세요.";
}
function onGuess(btn){
  if(!guessOn||gIdx>=N)return;
  var t=turns[gIdx];
  var z=(t.z_resid!==null&&t.z_resid!==undefined)?t.z_resid:t.z;
  var want=t.guess||guessClassJS(z);
  var ok=(btn===want);
  if(ok)gHits++;
  gIdx++;
  shown=gIdx;
  $("tally").textContent="나 "+gHits+" / 해설자 "+gIdx;
  $("guessResult").textContent="panel "+Math.round(t.raw100)+
    " · 정답 "+want+" → "+(ok?"맞음 +1":"빗나감");
  render(false);
  renderGuess();
  var txt=$("guessText");
  if(gIdx<N){
    var nx=turns[gIdx];
    txt.textContent="#"+nx.index+" "+(nx.display||nx.speaker)+": "+(nx.text||"");
  }else{
    renderGuess();
  }
}
function init(){
  $("title").textContent=data.title||"";
  try{document.title=data.title||"Uptake viewer2";}catch(e){}
  renderTurns();
  renderComments();
  renderSettle();
  render(false);
  $("play").addEventListener("click",togglePlay);
  var sp=document.querySelectorAll(".spdbtn");
  for(var i=0;i<sp.length;i++){
    (function(b){
      b.addEventListener("click",function(){
        speed=parseFloat(b.getAttribute("data-speed"))||1;
        for(var j=0;j<sp.length;j++)sp[j].classList.remove("active");
        b.classList.add("active");
      });
    })(sp[i]);
  }
  $("scrub").setAttribute("max",N);
  $("scrub").value=shown;
  $("scrub").addEventListener("input",function(e){
    var v=parseFloat(e.target.value)||0;
    shown=Math.max(0,Math.min(N,v));
    if(audioLinked()){
      var au=$("aud");
      var ti=Math.min(N-1,Math.floor(shown));
      var s=turns[ti]?turns[ti].start:null;
      if(au&&s!==null&&s!==undefined){try{au.currentTime=s+0.01;}catch(err){}}
    }
    render(false);
  });
  var hb2=$("heroBtn");
  if(hb2){
    hb2.addEventListener("click",function(){
      var hp=heroPos();
      if(hp<0||hp>=N)return;
      pause();
      shown=Math.min(N,hp+1);
      pulseT=(typeof performance!=="undefined"&&performance.now)?
        performance.now():0;
      render(true);
      if(!REDUCED){
        var iv=setInterval(function(){
          render(false);
          var t2=(typeof performance!=="undefined"&&performance.now)?
            performance.now():0;
          if(t2-pulseT>700){clearInterval(iv);render(false);}
        },50);
      }
    });
  }
  $("guessToggle").addEventListener("click",function(){setGuess(!guessOn);});
  var gb=document.querySelectorAll(".gbtn");
  for(var k=0;k<gb.length;k++){
    (function(b){
      b.addEventListener("click",function(){onGuess(b.getAttribute("data-g"));});
    })(gb[k]);
  }
  if(audioLinked()){
    var au2=$("aud");
    if(au2){
      au2.addEventListener("timeupdate",function(){
        if(!playing)return;
        shown=turnAtTime(au2.currentTime);
        render(true);
      });
      au2.addEventListener("ended",function(){pause();shown=N;render(false);});
      au2.addEventListener("play",function(){
        playing=true;
        $("play").textContent="❚❚";
        $("play").classList.add("active");
      });
      au2.addEventListener("pause",function(){
        if(playing)pause();
      });
    }
  }
}
init();
})();
"""

_SCALE_LABELS = {
    "fine100": "panel 0–100",
    "seven": "panel −3..+3",
    "v2": "panel −2..+2",
    "dist5": "panel distribution 0–100",
    "pairwise": "읽기 척도: 쌍대 비교(BT 평점)",
}


def _legend_html() -> str:
    parts = []
    for mark in MARKS:
        ko = html.escape(LABEL_KO.get(mark, ""))
        en = html.escape(LABEL_EN.get(mark, ""))
        parts.append(
            f'<span class="lg"><span class="badge" data-mark="{mark}">'
            f"{mark}</span> {ko}/{en}</span>"
        )
    return "".join(parts)


def _speaker_legend_html(view) -> str:
    speakers = view.get("speakers") or []
    if len(speakers) < 3:
        return ""
    colors = view.get("colors") or {}
    names = view.get("display_names") or {}
    parts = []
    for sp in speakers:
        disp = names.get(sp, sp)
        col = colors.get(sp, "#999")
        parts.append(
            '<span class="lg"><span class="sw" style="background:'
            f'{html.escape(str(col), quote=True)}"></span> '
            f"{html.escape(str(disp))}</span>"
        )
    return "".join(parts)


def _hero_html(view) -> str:
    hero = view.get("hero") if isinstance(view.get("hero"), dict) else None
    top = hero.get("top") if hero else None
    if not isinstance(top, dict):
        return ""
    disp = top.get("display") or top.get("speaker", "")
    label = f"판이 갈린 수 · #{top.get('index')} {disp}"
    parts = ['<div id="hero">',
             "<div class=\"hero-main\"><strong>"
             + html.escape(label)
             + "</strong> — "
             + html.escape(str(top.get("text80") or ""))
             + "</div>"]
    if top.get("line"):
        parts.append("<div class=\"hero-line\">"
                     + html.escape(str(top["line"])) + "</div>")
    bits = []
    for r in hero.get("runners") or []:
        if not isinstance(r, dict):
            continue
        rd = r.get("display") or r.get("speaker", "")
        try:
            delta = float(r.get("delta", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        bits.append(f"#{r.get('index')} {html.escape(str(rd))} "
                    f"(ΔM {delta:+.1f})")
    if bits:
        parts.append("<div class=\"hero-sub\">그다음 변동: "
                     + " · ".join(bits) + "</div>")
    parts.append("<div class=\"controls\"><button id=\"heroBtn\" "
                 "type=\"button\">그 장면 보기</button></div>")
    parts.append("</div>")
    return "".join(parts)


def render_html(view) -> str:
    view = view if isinstance(view, dict) else {}
    blob = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("<", "\\u003c")
    audio = view.get("audio") if isinstance(view.get("audio"), dict) else None
    if audio and audio.get("src"):
        src = html.escape(str(audio["src"]), quote=True)
        audio_tag = f'<audio id="aud" controls src="{src}"></audio>'
    else:
        audio_tag = ""
    scale_label = _SCALE_LABELS.get(view.get("scale", "v2"),
                                    _SCALE_LABELS["v2"])
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Uptake viewer2 — tide board</title>\n"
        "<style>" + _CSS + "</style>\n</head>\n<body>\n"
        "<header>\n"
        '<h1 id="title"></h1>\n'
        '<p class="note">model reading of moves, not a judgment of people</p>\n'
        '<p id="scale" class="note">' + scale_label + "</p>\n"
        "</header>\n"
        '<main class="layout">\n'
        '<section class="card"><h2>밀물/썰물 · Tide</h2>'
        + _hero_html(view)
        + '<div id="stands"></div>'
        '<svg id="tide"></svg>'
        '<div class="controls">'
        '<button id="play" type="button">▶</button>'
        '<button type="button" class="spdbtn active" data-speed="1">×1</button>'
        '<button type="button" class="spdbtn" data-speed="2">×2</button>'
        '<button type="button" class="spdbtn" data-speed="4">×4</button>'
        '<input id="scrub" type="range" min="0" max="1" step="0.1" value="0">'
        '<button id="guessToggle" type="button" aria-pressed="false">'
        "다음 수 맞히기</button>"
        '<span id="tally">나 0 / 해설자 0</span>'
        "</div>\n"
        + audio_tag + "\n"
        + '<div class="legend">' + _speaker_legend_html(view)
        + _legend_html() + "</div></section>\n"
        '<section class="card" id="guessCard" style="display:none">'
        "<h2>다음 수 맞히기 · Guess mode</h2>"
        '<p id="guessText"></p>'
        '<div class="controls">'
        '<button type="button" class="gbtn" data-g="!">!</button>'
        '<button type="button" class="gbtn" data-g="=">=</button>'
        '<button type="button" class="gbtn" data-g="?">?</button>'
        "</div>\n"
        '<p id="guessResult"></p></section>\n'
        '<section class="card"><h2>정산 사각형 · Square</h2>'
        '<div class="sq-wrap">'
        '<svg id="square" width="360" height="320"></svg></div>'
        '<p id="settle"></p><p id="likeness"></p>'
        '<p class="note">axes 20–80 (zoomed); dotted trail of all turns, '
        "last point large</p></section>\n"
        '<section class="card"><h2>전체 턴 · All turns</h2>'
        '<div id="turns"></div></section>\n'
        '<section class="card"><h2>결정적 장면 · Pivotal turns</h2>'
        '<div id="cmt-list"></div></section>\n'
        "</main>\n"
        '<script type="application/json" id="data">'
        + blob
        + "</script>\n<script>\n"
        + _JS
        + "</script>\n</body>\n</html>\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-viewer2")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--casino", default=None)
    ap.add_argument("--audio-record", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--no-copy-audio", dest="copy_audio", action="store_false",
                    default=True,
                    help="reference --audio via relpath instead of copying "
                         "it next to the HTML")
    ap.add_argument("--names", default=None,
                    help="JSON object mapping speaker labels to display names")
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
    audio_record = None
    if args.audio_record:
        with open(args.audio_record, encoding="utf-8") as f:
            audio_record = json.load(f)
    names = None
    if args.names:
        try:
            names = json.loads(args.names)
        except (TypeError, ValueError):
            names = None
    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    view = build_view2(
        item,
        commentary,
        outcomes=outcomes,
        audio_record=audio_record,
        audio_src=resolve_audio_src(args.audio, out_path,
                                    copy_audio=args.copy_audio),
        names=names,
    )
    out_path.write_text(render_html(view), encoding="utf-8")



