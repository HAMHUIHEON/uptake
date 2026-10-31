"""Layer-2 relationship board: a one-page story for first-time viewers.

Layer 1 (viewer2) is the tide board. Layer 2 is what a person who has
never seen the project reads in 30 seconds. Everything is drawn from the
board record plus its commentary JSON (standings on 0-100, decisive
turns, caption data). Single self-contained HTML, light theme, stdlib
only.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from . import alerts
from . import misread
from .commentary import load_references, standing_round
from .viewer2 import (
    MOVE_KO,
    SPEAKER_COLORS,
    build_view2,
    hero_swings,
    momentum_series,
    resolve_audio_src,
    tide_caption,
    tide_y,
)

LIKE_KO = {
    "Extremely like": "매우 좋아함",
    "Slightly like": "약간 좋아함",
    "Undecided": "미정",
    "Slightly dislike": "약간 싫어함",
    "Extremely dislike": "매우 싫어함",
}
SAT_KO = {
    "Extremely satisfied": "매우 만족함",
    "Slightly satisfied": "약간 만족함",
    "Undecided": "미정",
    "Slightly dissatisfied": "약간 불만족",
    "Extremely dissatisfied": "매우 불만족",
}
_LIKE_NUM = {2: "매우 좋아함", 1: "약간 좋아함", 0: "미정",
            -1: "약간 싫어함", -2: "매우 싫어함"}
_SAT_NUM = {2: "매우 만족함", 1: "약간 만족함", 0: "미정",
           -1: "약간 불만족", -2: "매우 불만족"}

_TIDE_W = 640.0
_TIDE_H = 220.0
_PAD_L = 30.0
_PAD_R = 10.0
_PAD_T = 18.0
_PAD_B = 24.0
_MID = (_PAD_T + _TIDE_H - _PAD_B) / 2.0
_AMP = (_TIDE_H - _PAD_T - _PAD_B) / 2.0 / 3.2


def likeness_ko(value) -> str:
    """Korean label for a CaSiNo opponent-likeness value.

    Accepts the raw label ("Slightly like") or the -2..2 numeric score
    that ``outcome.load_outcomes`` produces.
    """
    if isinstance(value, str) and value.strip() in LIKE_KO:
        return LIKE_KO[value.strip()]
    if isinstance(value, bool):
        return "미정"
    try:
        return _LIKE_NUM.get(int(value), "미정")
    except (TypeError, ValueError):
        return "미정"


def satisfaction_ko(value) -> str:
    """Korean label for a CaSiNo satisfaction value (label or -2..2)."""
    if isinstance(value, str) and value.strip() in SAT_KO:
        return SAT_KO[value.strip()]
    if isinstance(value, bool):
        return "미정"
    try:
        return _SAT_NUM.get(int(value), "미정")
    except (TypeError, ValueError):
        return "미정"


def outcome_line(rec) -> str | None:
    """One-line actual outcome for a score card, or None when unknown."""
    if not isinstance(rec, dict):
        return None
    pts = rec.get("points")
    if isinstance(pts, bool):
        return None
    try:
        points = int(pts)
    except (TypeError, ValueError):
        return None
    like = likeness_ko(rec.get("likeness"))
    if like == "미정":
        return f"점수 {points}, 상대 평가 미정"
    return f"점수 {points}, 상대를 {like}"


def delta_word(end) -> tuple[str, str]:
    """(arrow, word) for a final standing against the 50 average."""
    try:
        e = int(end)
    except (TypeError, ValueError):
        e = 50
    if e > 50:
        return ("↑", "얻음")
    if e < 50:
        return ("↓", "잃음")
    return ("→", "지킴")


def _tide_x(i, n) -> float:
    span = _TIDE_W - _PAD_L - _PAD_R
    if n <= 1:
        return _PAD_L + span / 2.0
    return _PAD_L + ((float(i) + 0.5) / (float(n) - 0.5)) * span


def tide_svg(momentum, flags) -> str:
    """Simplified tide: flat A/B fills, boundary curve, midline, flags."""
    moms: list[float] = []
    for m in momentum or []:
        try:
            moms.append(float(m))
        except (TypeError, ValueError):
            moms.append(0.0)
    n = len(moms)
    parts = [
        '<svg id="tide" width="100%" height="220" viewBox="0 0 640 220" '
        'role="img" aria-label="simplified tide">'
    ]
    parts.append(
        f'<line x1="{_PAD_L:.1f}" x2="{_TIDE_W - _PAD_R:.1f}" '
        f'y1="{_MID:.1f}" y2="{_MID:.1f}" class="mid"/>'
    )
    if n:
        pts = [(_PAD_L, _MID)]
        for i, m in enumerate(moms):
            pts.append((_tide_x(i, n), tide_y(m, _MID, _AMP)))
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        x0 = pts[0][0]
        x1 = pts[-1][0]
        y_top = _PAD_T
        y_bot = _TIDE_H - _PAD_B
        col_a = SPEAKER_COLORS[0] if len(SPEAKER_COLORS) > 0 else "#f0606a"
        col_b = SPEAKER_COLORS[1] if len(SPEAKER_COLORS) > 1 else "#7fb3ff"
        top_d = f"{d}L{x1:.1f},{y_top:.1f}L{x0:.1f},{y_top:.1f}Z"
        bot_d = f"{d}L{x1:.1f},{y_bot:.1f}L{x0:.1f},{y_bot:.1f}Z"
        parts.append(
            f'<path d="{top_d}" fill="{col_a}" fill-opacity="0.25" '
            'class="tide-a"/>'
        )
        parts.append(
            f'<path d="{bot_d}" fill="{col_b}" fill-opacity="0.25" '
            'class="tide-b"/>'
        )
        parts.append(f'<path d="{d}" class="tide-line"/>')
        for fl in flags or []:
            if not isinstance(fl, dict):
                continue
            pos = fl.get("pos")
            if not isinstance(pos, int) or pos < 0 or pos >= n:
                continue
            cx = _tide_x(pos, n)
            cy = tide_y(moms[pos], _MID, _AMP)
            num = fl.get("n", "")
            col = str(fl.get("color") or "#333")
            ty = cy - 26.0
            if ty < 12.0:
                ty = 12.0
            sy = ty + 10.0
            if sy > cy:
                sy = cy
            parts.append(
                f'<g class="flag"><line x1="{cx:.1f}" y1="{cy:.1f}" '
                f'x2="{cx:.1f}" y2="{sy:.1f}" stroke="{col}"/>'
                f'<circle cx="{cx:.1f}" cy="{ty:.1f}" r="10" '
                f'fill="{col}" stroke="#fff" stroke-width="2"/>'
                f'<text x="{cx:.1f}" y="{ty + 4.0:.1f}">{num}</text></g>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _quote80(text) -> str:
    t = str(text or "")
    if len(t) > 80:
        return t[:80] + "…"
    return t


def scene_card_html(turn) -> str:
    """Seasoning card under a decisive card (task50 import point)."""
    return ""


def _standing_pair(trail, pos, speaker) -> tuple[int, int]:
    """(before, after) standings around one turn, rounded for display."""
    try:
        before_v = 50.0
        if (isinstance(pos, int) and pos > 0
                and isinstance(trail, list) and pos - 1 < len(trail)
                and isinstance(trail[pos - 1], dict)):
            before_v = trail[pos - 1].get(speaker, 50.0)
        after_v = 50.0
        if (isinstance(pos, int) and isinstance(trail, list)
                and 0 <= pos < len(trail)
                and isinstance(trail[pos], dict)):
            after_v = trail[pos].get(speaker, 50.0)
        before = standing_round(before_v)
        after = standing_round(after_v)
    except (TypeError, ValueError):
        before, after = 50, 50
    return before, after


def _ro_particle(number) -> str:
    """'로' vs '으로' after a standing, by its Korean reading."""
    try:
        ones = abs(int(number)) % 10
    except (TypeError, ValueError):
        return "로"
    if ones in (1, 2, 4, 5, 7, 8, 9):
        return "로"
    return "으로"


def _card_line(moves, display, before, after) -> str:
    """Plain-language card body: action + standing move + direction verb."""
    phrases = [MOVE_KO.get(m, m) for m in moves or []
               if isinstance(m, str) and m]
    if not phrases:
        phrases = ["발언"]
    action = "하며 ".join(phrases)
    ro = _ro_particle(after)
    if after > before:
        return (f"{action}해서 {display}의 관계 점수가 "
                f"{before} → {after}{ro} 올랐다")
    if after < before:
        return (f"{action}했지만, {display}의 관계 점수가 "
                f"{before} → {after}{ro} 떨어졌다")
    return (f"{action}했지만, {display}의 관계 점수가 "
            f"{before} → {after}{ro} 그대로였다")


def _decisive_entry(n, pos, vt, com_turn, colors, names, trail) -> dict:
    vt = vt if isinstance(vt, dict) else {}
    com_turn = com_turn if isinstance(com_turn, dict) else None
    if com_turn is not None and com_turn.get("speaker"):
        speaker = com_turn.get("speaker", "")
    else:
        speaker = vt.get("speaker", "")
    if not isinstance(speaker, str):
        speaker = str(speaker)
    if com_turn is not None and "moves" in com_turn:
        moves = com_turn.get("moves", []) or []
    else:
        moves = vt.get("moves", []) or []
    moves = [m for m in moves if isinstance(m, str)]
    ko = ", ".join(MOVE_KO.get(m, m) for m in moves) if moves else "발언"
    display = names.get(speaker, speaker)
    before, after = _standing_pair(trail, pos, speaker)
    line = _card_line(moves, display, before, after)
    if com_turn is not None and "text" in com_turn:
        text = com_turn.get("text", "")
    else:
        text = vt.get("text", "")
    if com_turn is not None and "index" in com_turn:
        idx = com_turn.get("index", vt.get("index", 0))
    else:
        idx = vt.get("index", 0)
    ref = com_turn.get("reference") if com_turn is not None else None
    z_resid = None
    if com_turn is not None and com_turn.get("z_resid") is not None:
        z_resid = com_turn.get("z_resid")
    elif vt.get("z_resid") is not None:
        z_resid = vt.get("z_resid")
    try:
        zr = float(z_resid) if z_resid is not None else None
    except (TypeError, ValueError):
        zr = None
    return {
        "n": n,
        "pos": pos if isinstance(pos, int) else None,
        "index": idx,
        "speaker": speaker,
        "display": display,
        "quote": _quote80(text),
        "line": line,
        "moves": list(moves),
        "moves_ko": ko,
        "moves_raw": ", ".join(moves),
        "standing_before": before,
        "standing_after": after,
        "z_resid": zr,
        "color": colors.get(speaker, "#333"),
        "reference": ref if isinstance(ref, dict) else {},
    }


def _diverse_positions(deltas, speakers) -> list[int]:
    """Up to 3 positions by |ΔM| with reversal + speaker diversity."""
    n = len(deltas)
    order = sorted(range(n), key=lambda p: (-abs(deltas[p]), p))
    if len(order) <= 3:
        return list(order)
    ranked = list(order)
    sel = ranked[:3]
    first = ranked[0]
    d0 = deltas[first]
    big = abs(d0) >= 1.0 - 1e-9 and d0 != 0.0
    if big:
        has_rev = any(deltas[p] * d0 < 0 for p in sel)
        if not has_rev:
            for p in ranked[3:]:
                if abs(deltas[p]) < 1.0 - 1e-9:
                    continue
                if deltas[p] * d0 < 0:
                    sel = [sel[0], sel[1], p]
                    sel.sort(key=lambda q: (-abs(deltas[q]), q))
                    break
    if len(sel) == 3:
        sps = [speakers[p] for p in sel]
        if sps[0] == sps[1] == sps[2]:
            maj = sps[0]
            cand = None
            for p in ranked:
                if p in sel:
                    continue
                if speakers[p] == maj:
                    continue
                if abs(deltas[p]) < 1.0 - 1e-9:
                    continue
                cand = p
                break
            if cand is not None:
                if not big or any(deltas[p] * d0 < 0 for p in sel):
                    revs = [p for p in sel[1:]
                            if big and deltas[p] * d0 < 0]
                    if len(revs) >= 2 or not big:
                        victim = sel[-1]
                    elif len(revs) == 1:
                        if deltas[cand] * d0 < 0:
                            victim = sel[-1]
                        else:
                            same = [p for p in sel[1:]
                                    if not (deltas[p] * d0 < 0)]
                            victim = same[0] if same else sel[-1]
                    else:
                        victim = sel[-1]
                else:
                    victim = sel[-1]
                sel = [p for p in sel if p != victim] + [cand]
                sel.sort(key=lambda q: (-abs(deltas[q]), q))
    return sel


def decisive_of(commentary, view, names=None) -> list[dict]:
    """Up to 3 decisive turns by |ΔM| with reversal/speaker diversity."""
    names = names if isinstance(names, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    view = view if isinstance(view, dict) else {}
    turns = view.get("turns") or []
    if not isinstance(turns, list):
        turns = []
    colors = view.get("colors") or {}
    if not isinstance(colors, dict):
        colors = {}
    trail = view.get("standings") or []
    raw = commentary.get("turns") or []
    if not isinstance(raw, list):
        raw = []
    com_by_index: dict = {}
    for c in raw:
        if not isinstance(c, dict):
            continue
        idx = c.get("index")
        if idx not in com_by_index:
            com_by_index[idx] = c
    by_index: dict = {}
    for pos, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        idx = t.get("index")
        if idx not in by_index:
            by_index[idx] = pos
    moms_raw = view.get("momentum") or []
    use_dm = bool(turns) and bool(moms_raw)
    if use_dm:
        try:
            n_all = len(turns)
            moms: list[float] = []
            for i in range(n_all):
                try:
                    moms.append(float(moms_raw[i]))
                except (TypeError, ValueError, IndexError):
                    moms.append(0.0)
        except TypeError:
            use_dm = False
    if not use_dm:
        out: list[dict] = []
        for c in raw[:3]:
            if not isinstance(c, dict):
                continue
            pos = by_index.get(c.get("index"))
            vt = (turns[pos] if isinstance(pos, int)
                  and 0 <= pos < len(turns) else {})
            out.append(_decisive_entry(len(out) + 1, pos, vt, c,
                                       colors, names, trail))
        return out
    idx_pos: list[int] = []
    deltas_all: list[float] = []
    speakers_all: list[str] = []
    for pos, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        prev = moms[pos - 1] if pos > 0 and pos - 1 < len(moms) else 0.0
        cur = moms[pos] if pos < len(moms) else 0.0
        deltas_all.append(float(cur - prev))
        sp = t.get("speaker", "")
        speakers_all.append(sp if isinstance(sp, str) else str(sp))
        idx_pos.append(pos)
    if not idx_pos:
        return []
    local = _diverse_positions(deltas_all, speakers_all)
    chosen = [idx_pos[i] for i in local]
    chosen.sort()
    out = []
    for pos in chosen:
        vt = turns[pos] if 0 <= pos < len(turns) else {}
        idx = vt.get("index", pos) if isinstance(vt, dict) else pos
        com_turn = com_by_index.get(idx)
        out.append(_decisive_entry(len(out) + 1, pos, vt, com_turn,
                                   colors, names, trail))
    return out


def _lesson_pick(decisive) -> tuple[str, dict]:
    """(idea, reference record) behind the one-line lesson."""
    try:
        refs = {r.get("id"): r for r in load_references()
                if isinstance(r, dict)}
    except (OSError, ValueError):
        return "", {}
    for d in decisive or []:
        if not isinstance(d, dict):
            continue
        ref = d.get("reference") or {}
        if not isinstance(ref, dict):
            continue
        rec = refs.get(ref.get("id"), {})
        idea = rec.get("idea_ko", "") if isinstance(rec, dict) else ""
        if isinstance(idea, str) and idea.strip():
            return idea.strip(), rec if isinstance(rec, dict) else {}
    return "", {}


def lesson_of(decisive) -> str:
    """One-line lesson: the first decisive turn's reference idea (Korean)."""
    return _lesson_pick(decisive)[0]


def _short_author(author) -> str:
    """'Roger Fisher; William Ury' -> 'Fisher & Ury'."""
    surnames = []
    for bit in str(author or "").split(";"):
        depth = 0
        chars = []
        for ch in bit:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0:
                chars.append(ch)
        toks = "".join(chars).strip().strip(",").split()
        if toks:
            surnames.append(toks[-1].strip(","))
    return " & ".join(surnames)


def _short_title(title) -> str:
    """'Getting to Yes: Negotiating ...' -> 'Getting to Yes'."""
    return str(title or "").split(":")[0].strip()


def lesson_source_of(decisive) -> str:
    """Short source for the lesson: 'Fisher & Ury, Getting to Yes'."""
    idea, rec = _lesson_pick(decisive)
    if not idea:
        return ""
    author = _short_author(rec.get("author", ""))
    title = _short_title(rec.get("title", ""))
    if author and title:
        return f"{author}, {title}"
    return author or title


def _flag_label(d) -> str:
    """Tide flag text: the turn number as on the cards ('#3')."""
    idx = d.get("index", d.get("n", "")) if isinstance(d, dict) else ""
    try:
        if isinstance(idx, bool):
            raise ValueError("bool")
        return f"#{int(idx)}"
    except (TypeError, ValueError):
        s = str(idx)
        return f"#{s}" if s else ""


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
    """Map turn index -> suggest notice ({alternatives, scene_id})."""
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
    notices = []
    if isinstance(data, dict) and isinstance(data.get("notices"), list):
        notices = data["notices"]
    elif isinstance(data, list):
        notices = data
    elif isinstance(data, dict):
        # Direct {turn: {...}} mapping.
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
    out = {}
    for n in notices:
        if not isinstance(n, dict):
            continue
        idx = n.get("turn", n.get("index"))
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        out[idx] = n
    return out


def _is_blunder_entry(d) -> bool:
    try:
        from .scenes import is_blunder as _ib

        return bool(_ib(d))
    except ImportError:
        pass
    if not isinstance(d, dict):
        return False
    try:
        zr = d.get("z_resid")
        if zr is not None and float(zr) <= -1.5:
            return True
    except (TypeError, ValueError):
        pass
    try:
        b = float(d.get("standing_before", 50))
        a = float(d.get("standing_after", 50))
        return (b - a) >= 15
    except (TypeError, ValueError):
        return False


def attach_notices(decisive, scenes=None, suggests=None, dialogue_id="",
                   culture="mix", lang="ko") -> int:
    """Attach notice data to blunder entries in place. Returns notice count."""
    pool = _load_scene_pool(scenes)
    sug_map = _load_suggest_map(suggests)
    if not pool and not sug_map:
        return 0
    by_id = {str(s.get("id", "")): s for s in pool}
    try:
        from .scenes import pick_scene as _pick
    except ImportError:
        _pick = None
    used: set[str] = set()
    # Reserve suggested scene ids first so picks never repeat them.
    for n in sug_map.values():
        sid = n.get("scene_id", "") if isinstance(n, dict) else ""
        if sid:
            used.add(str(sid))
    count = 0
    blunder_pos = 0
    for d in decisive or []:
        if not isinstance(d, dict):
            continue
        if not _is_blunder_entry(d):
            continue
        idx = d.get("index", 0)
        try:
            idx_i = int(idx) if not isinstance(idx, bool) else 0
        except (TypeError, ValueError):
            idx_i = 0
        sug = sug_map.get(idx_i, {})
        alts = []
        if isinstance(sug, dict):
            raw_alts = sug.get("alternatives", [])
            alts = [a for a in (raw_alts or []) if isinstance(a, dict)]
        scene = None
        sid = sug.get("scene_id", "") if isinstance(sug, dict) else ""
        if sid and str(sid) in by_id:
            scene = by_id[str(sid)]
        elif _pick is not None and pool:
            moves = d.get("moves", [])
            if not moves:
                raw = str(d.get("moves_raw", "") or "")
                moves = [m.strip() for m in raw.split(",") if m.strip()]
            try:
                scene = _pick(moves or "other", dialogue_id, idx_i, pool,
                              culture=culture, used=used,
                              position=blunder_pos)
            except (TypeError, ValueError):
                scene = None
            if scene is not None:
                used.add(str(scene.get("id", "")))
        elif pool:
            # No picker: first unused scene (should not happen).
            for s in sorted(pool, key=lambda x: str(x.get("id", ""))):
                if str(s.get("id", "")) not in used:
                    scene = s
                    used.add(str(s.get("id", "")))
                    break
        try:
            before = int(d.get("standing_before", 50))
        except (TypeError, ValueError):
            before = 50
        try:
            after = int(d.get("standing_after", 50))
        except (TypeError, ValueError):
            after = 50
        d["notice"] = {
            "before": before,
            "after": after,
            "display": d.get("display", d.get("speaker", "")),
            "speaker": d.get("speaker", ""),
            "alternatives": alts,
            "scene": scene,
            "lang": lang,
        }
        count += 1
        blunder_pos += 1
    return count


def notice_html_for(entry, lang="ko") -> str:
    """Render one decisive entry's notice block ("" when none)."""
    if not isinstance(entry, dict):
        return ""
    notice = entry.get("notice")
    if not isinstance(notice, dict):
        return ""
    lang = notice.get("lang", lang)
    try:
        from .scenes import render_notice as _render

        return _render(
            before=notice.get("before", 50),
            after=notice.get("after", 50),
            display=notice.get("display", ""),
            alternatives=notice.get("alternatives", []),
            scene=notice.get("scene"),
            lang=lang,
            speaker=notice.get("speaker", ""),
        )
    except ImportError:
        pass
    # Fallback without the scenes module (same content, plain).
    before = notice.get("before", 50)
    after = notice.get("after", 50)
    display = html.escape(str(notice.get("display", "")))
    parts = ['<div class="notice">',
             f"<p class=\"warn\">⚠ 이 말은 {display}에게 "
             f"{before} → {after}로 읽혔습니다</p>"]
    alts = [a for a in (notice.get("alternatives", []) or [])
            if isinstance(a, dict)]
    if alts:
        parts.append("<p class=\"althead\">이렇게 말해 볼 수도</p><ul>")
        for a in alts:
            try:
                pct = int(round(float(a.get("p_better", 0.0)) * 100))
            except (TypeError, ValueError):
                pct = 0
            parts.append("<li>“" + html.escape(str(a.get("text", "")))
                         + f"” ({pct}%)</li>")
        parts.append("</ul>")
    scene = notice.get("scene")
    if isinstance(scene, dict):
        told = html.escape(str(scene.get("상황_한줄", "")))
        tip = html.escape(str(scene.get("교훈", "")))
        if told or tip:
            parts.append(f"<p class=\"scene\">{told} {tip}</p>")
    parts.append("</div>")
    return "".join(parts)


def _activity_rank(view) -> list:
    """Speakers ranked by turn count, ties by first appearance."""
    view = view if isinstance(view, dict) else {}
    speakers = list(view.get("speakers") or [])
    counts = {sp: 0 for sp in speakers}
    first = {sp: i for i, sp in enumerate(speakers)}
    for t in view.get("turns") or []:
        sp = t.get("speaker") if isinstance(t, dict) else None
        if sp in counts:
            counts[sp] += 1
        elif isinstance(sp, str) and sp:
            counts[sp] = 1
            first.setdefault(sp, len(first))
    return sorted(counts, key=lambda sp: (-counts[sp], first.get(sp, 0)))


def _first_sentence(text) -> str:
    """First sentence of a case outcome (whole text when one sentence)."""
    t = str(text or "").strip()
    if not t:
        return ""
    for i, ch in enumerate(t):
        if ch not in ".!?。！？":
            continue
        if (ch == "." and 0 < i < len(t) - 1
                and t[i - 1].isdigit() and t[i + 1].isdigit()):
            continue
        return t[:i + 1].strip() or t
    return t


def _first_alert_card(commentary, view, names, alert_first):
    """A decisive entry for the first-alert turn, or None if unbuildable."""
    if not isinstance(alert_first, dict):
        return None
    target = alert_first.get("index")
    if target is None or isinstance(target, bool):
        return None
    turns = view.get("turns") or []
    if not isinstance(turns, list):
        return None
    pos = None
    vt: dict = {}
    for p, t in enumerate(turns):
        if isinstance(t, dict) and t.get("index") == target:
            pos, vt = p, t
            break
    if pos is None:
        return None
    com_turn = None
    raw = commentary.get("turns") or []
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict) and c.get("index") == target:
                com_turn = c
                break
    colors = view.get("colors") or {}
    if not isinstance(colors, dict):
        colors = {}
    trail = view.get("standings") or []
    return _decisive_entry(1, pos, vt, com_turn, colors,
                           names if isinstance(names, dict) else {},
                           trail)


def _insert_first_alert(decisive, commentary, view, names, alert_first):
    """Render the first 경계+ turn as the first decisive card in place."""
    if not isinstance(decisive, list) or not isinstance(alert_first, dict):
        return
    target = alert_first.get("index")
    if target is None or isinstance(target, bool):
        return
    hit = next((d for d in decisive
                if isinstance(d, dict) and d.get("index") == target), None)
    if hit is not None:
        decisive.remove(hit)
        decisive.insert(0, hit)
        hit["first_alert"] = True
    else:
        try:
            entry = _first_alert_card(commentary, view, names, alert_first)
        except (TypeError, ValueError):
            entry = None
        if entry is None:
            return
        entry["first_alert"] = True
        decisive.insert(0, entry)
    for i, d in enumerate(decisive):
        if isinstance(d, dict):
            d["n"] = i + 1


def build_board2(item, commentary, outcomes=None, audio_src=None,
                names=None, scenes=None, suggests=None, culture="mix",
                lang="ko", panel=None, intent=None, case=None,
                outcome_turn=None, genre=None,
                **kwargs) -> dict:
    """Assemble the layer-2 board model from record + commentary JSON."""
    if suggests is None:
        suggests = kwargs.get("suggestions", kwargs.get("suggest"))
    if scenes is None:
        scenes = kwargs.get("scene_list", kwargs.get("scenes_path"))
    if culture == "mix":
        culture = kwargs.get("culture_option", culture)
    if lang == "ko":
        lang = kwargs.get("language", lang)
    if panel is None:
        panel = kwargs.get("panel")
    if intent is None:
        intent = kwargs.get("intent_map", kwargs.get("intents"))
    if case is None:
        case = kwargs.get("case_info")
    if outcome_turn is None:
        outcome_turn = kwargs.get("outcome_turn_value")
    if genre is None:
        genre = kwargs.get("genre_option")
    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    names = names if isinstance(names, dict) else {}
    genre_name = None
    if isinstance(genre, str) and genre.strip():
        genre_name = genre.strip()
    elif (isinstance(item.get("genre"), str)
            and item.get("genre", "").strip()):
        genre_name = item["genre"].strip()
    view = build_view2(item, commentary, outcomes=outcomes, names=names)
    view_speakers = list(view.get("speakers") or [])
    if len(view_speakers) >= 3:
        ranked = _activity_rank(view)
        speakers = ranked[:6]
        tide_pair = ranked[:2]
        try:
            momentum = [float(m) for m in
                        momentum_series(view.get("turns"), tide_pair)]
        except (TypeError, ValueError):
            momentum = list(view.get("momentum") or [])
        rest = ranked[2:]
    else:
        speakers = view_speakers[:2]
        momentum = list(view.get("momentum") or [])
        rest = []
    disp_names = view.get("display_names") or {}
    trail = view.get("standings") or []
    last = trail[-1] if trail else {}
    if not isinstance(last, dict):
        last = {}
    colors_all = view.get("colors") or {}
    if not isinstance(colors_all, dict):
        colors_all = {}
    other_speakers = []
    for sp in rest:
        try:
            standing = standing_round(last.get(sp, 50.0))
        except (TypeError, ValueError):
            standing = 50
        other_speakers.append({
            "speaker": sp,
            "display": disp_names.get(sp, sp),
            "standing": standing,
            "color": colors_all.get(sp, "#333"),
        })
    cards = []
    for sp in speakers:
        try:
            end = standing_round(last.get(sp, 50.0))
        except (TypeError, ValueError):
            end = 50
        arrow, word = delta_word(end)
        rec = outcomes.get(sp) if isinstance(outcomes, dict) else None
        cards.append({
            "speaker": sp,
            "display": disp_names.get(sp, sp),
            "start": 50,
            "end": end,
            "arrow": arrow,
            "word": word,
            "outcome": outcome_line(rec),
            "color": (view.get("colors") or {}).get(sp, "#333"),
        })
    decisive = decisive_of(commentary, view, names)
    item_id = item.get("id", "")
    item_id = item_id if isinstance(item_id, str) else str(item_id or "")
    panel_data = misread.load_panel(panel)
    analyses = misread.analyze_panel(panel_data) if panel_data else []
    by_index = {a["index"]: a for a in analyses}
    panel_analyses = analyses if panel_data else None
    times: dict = {}
    for src in (item, panel_data):
        effs = src.get("effects", []) if isinstance(src, dict) else []
        for pos, e in enumerate(effs or []):
            if not isinstance(e, dict) or "time" not in e:
                continue
            idx = e.get("index", pos)
            if isinstance(idx, bool) or not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = pos
            if idx not in times:
                times[idx] = e.get("time")
    alert_list = alerts.compute_alerts(
        view.get("turns"), genre=genre_name,
        analyses=panel_analyses, times=times)
    alert_first = alerts.first_alert(alert_list)
    alert_lead = alerts.lead_time(alert_list, outcome_turn=outcome_turn)
    _insert_first_alert(decisive, commentary, view, names, alert_first)
    if scenes is not None or suggests is not None:
        try:
            attach_notices(decisive, scenes=scenes, suggests=suggests,
                           dialogue_id=item_id, culture=culture, lang=lang)
        except Exception:
            pass
    flags = [{"n": _flag_label(d), "pos": d["pos"], "color": d["color"]}
             for d in decisive if d["pos"] is not None]
    for d in decisive:
        if not isinstance(d, dict):
            continue
        a = by_index.get(d.get("index"))
        if a is not None and a.get("majority"):
            d["misread_line"] = misread.card_line(
                a, before=d.get("standing_before"),
                after=d.get("standing_after"))
        else:
            d["misread_line"] = ""
    strip: list[str] = []
    if panel_data:
        all_speakers = list(view.get("speakers") or [])
        for a in misread.top_misread(analyses, k=5):
            sp = a.get("speaker", "")
            disp = disp_names.get(sp, sp)
            listener = misread.listener_of(a.get("listeners", []), sp,
                                           all_speakers)
            li_disp = disp_names.get(listener, listener) if listener else "상대"
            intent_first = (a.get("intent") or ["발언"])[0]
            rep = a.get("rep_ko") or a.get("majority_ko") or "?"
            strip.append(misread.strip_flag(a.get("index", 0), disp,
                                            intent_first, li_disp, rep))
    mis_n = sum(1 for a in analyses if a.get("misread"))
    intent_list = alerts.intent_rows(
        item=item, view=view, analyses=panel_analyses, intent=intent,
        trail=trail, panel=panel_data)
    case_info = alerts.load_case(case)
    caption = view.get("caption") or ""
    if not caption:
        caption = "발언 기록이 비어 있습니다"
    case_verdict = ""
    if isinstance(case_info, dict):
        case_verdict = _first_sentence(case_info.get("outcome", ""))
    verdict = case_verdict if case_verdict else caption
    verdict_sub = (caption if (case_verdict and case_verdict != caption)
                   else "")
    dash = f"dashboard_{item_id}.html" if item_id else "dashboard.html"
    return {
        "id": item_id,
        "verdict": verdict,
        "verdict_sub": verdict_sub,
        "caption": caption,
        "cards": cards,
        "other_speakers": other_speakers,
        "momentum": momentum,
        "flags": flags,
        "decisive": decisive,
        "panel": bool(panel_data),
        "misread_strip": strip,
        "misread_count": mis_n,
        "misread_total": len(analyses),
        "alerts": alert_list,
        "alert_first": alert_first,
        "alert_lead": alert_lead,
        "alert_genre": genre_name,
        "alert_weight": alerts.stakes_weight(genre_name),
        "outcome_turn": outcome_turn,
        "intent_rows": intent_list,
        "case": case_info,
        "lesson": lesson_of(decisive),
        "lesson_source": lesson_source_of(decisive),
        "dashboard": dash,
        "audio_src": str(audio_src) if audio_src else None,
        "lang": lang,
    }


_CSS = """
*{box-sizing:border-box}
body{max-width:720px;margin:0 auto;padding:0 16px 40px;background:#fff;
  color:#222;font-family:system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
  line-height:1.6}
h1{font-size:20px;margin:20px 0 2px}
h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid #e3e6ea;padding-bottom:4px}
h2 small{font-weight:400}
small{color:#5b6470}
.verdict{font-size:24px;font-weight:700;margin:8px 0 4px}
.badge{display:inline-block;border:1px solid #c9d1d9;border-radius:999px;
  padding:1px 10px;font-size:13px;color:#5b6470;background:#f7f8fa}
.sub{color:#5b6470;margin:0}
.strip{margin:4px 0;padding-left:22px}
.strip li{margin:2px 0}
.misread{margin:6px 0}
.cards{display:flex;gap:12px;flex-wrap:wrap}
.score{flex:1 1 150px;border:1px solid #e3e6ea;border-radius:12px;padding:12px;
  background:#f7f8fa;min-width:0}
.scoreline{font-size:18px;font-weight:700;margin:0}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px}
.delta{margin:4px 0;font-weight:700}
.outcome{margin:0}
.note{color:#5b6470;font-size:13px}
.legend{font-size:13px;color:#5b6470;margin:6px 0 0}
.moment{border:1px solid #e3e6ea;border-radius:12px;padding:10px 12px;margin:8px 0}
.quote{font-weight:700;margin:0}
.line{margin:6px 0}
.movekind{margin:0;color:#5b6470;font-size:13px}
.notice{border:1px dashed #d0b429;border-radius:10px;padding:10px 12px;margin:8px 0;
  background:#fffbe6}
.warn{font-weight:700;margin:0}
.althead{font-weight:700;margin:8px 0 4px}
.alts{margin:4px 0;padding-left:22px}
.pct{color:#5b6470;font-size:13px}
.scene{margin:8px 0 0}
.src{margin:2px 0 0;color:#5b6470;font-size:13px}
blockquote{border-left:3px solid #2f6fed;margin:8px 0;padding:4px 12px;font-size:17px}
.lsrc{font-size:13px;color:#5b6470}
footer{margin-top:28px;border-top:1px solid #e3e6ea;padding-top:10px;
  color:#5b6470;font-size:13px}
#tide{width:100%;height:auto;display:block}
.tide-a{stroke:none}
.tide-b{stroke:none}
.tide-line{fill:none;stroke:#333;stroke-width:2}
.mid{stroke:#9aa4b2;stroke-width:1;stroke-dasharray:4 3}
.flag line{stroke-width:2}
.flag text{font-size:11px;font-weight:700;fill:#fff;text-anchor:middle}
.alert-strip{display:flex;gap:2px;margin:6px 0}
.tick{flex:1 1 0;height:14px;border-radius:3px;min-width:4px}
.alert-flag{font-weight:700;margin:6px 0}
.intent{width:100%;border-collapse:collapse;font-size:14px}
.intent th,.intent td{border:1px solid #e3e6ea;padding:6px 8px;text-align:left;
  vertical-align:top}
.intent thead th{background:#f7f8fa}
.case-title{font-weight:700;margin:0}
audio{width:100%;margin-top:10px}
"""


def _legend_html(cards) -> str:
    if len(cards) >= 2:
        d0 = cards[0]["display"]
        d1 = cards[1]["display"]
        text = f"빨강 {d0} · 파랑 {d1} · 선이 내려오면 {d0} 우세"
    elif len(cards) == 1:
        d0 = cards[0]["display"]
        text = f"빨강 {d0} · 선이 내려오면 {d0} 우세"
    else:
        return ""
    return '<p class="legend">' + html.escape(text) + "</p>"


def _others_html(board) -> str:
    """Small "다른 화자" line: speakers past the tide pair + standings."""
    others = board.get("other_speakers") or []
    bits = []
    for o in others:
        if not isinstance(o, dict):
            continue
        disp = o.get("display", o.get("speaker", ""))
        try:
            st = int(o.get("standing", 50))
        except (TypeError, ValueError):
            st = 50
        bits.append(f"{disp} {st}")
    if not bits:
        return ""
    return ('<p class="note">다른 화자: '
            + html.escape(" · ".join(bits)) + "</p>")


def render_html(board) -> str:
    """Render the board model as one self-contained HTML page."""
    board = board if isinstance(board, dict) else {}
    item_id = html.escape(str(board.get("id", "")))
    verdict = html.escape(str(board.get("verdict", "")))
    cards = board.get("cards") or []
    cards_html = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        disp = html.escape(str(c.get("display", "")))
        col = html.escape(str(c.get("color", "#333")), quote=True)
        try:
            end = int(c.get("end", 50))
        except (TypeError, ValueError):
            end = 50
        arrow = html.escape(str(c.get("arrow", "")))
        word = html.escape(str(c.get("word", "")))
        oc = c.get("outcome")
        if oc:
            oc_tag = '<p class="outcome">' + html.escape(str(oc)) + "</p>"
        else:
            oc_tag = ""
        cards_html.append(
            '<div class="score"><p class="scoreline">'
            f'<span class="sw" style="background:{col}"></span> '
            f"{disp} · 50 → {end}</p>"
            f'<p class="delta">{arrow} {word}</p>'
            f"{oc_tag}</div>"
        )
    moments = []
    for d in board.get("decisive") or []:
        if not isinstance(d, dict):
            continue
        disp = html.escape(str(d.get("display", "")))
        quote = html.escape(str(d.get("quote", "")))
        line = html.escape(str(d.get("line", "")))
        mko_raw = str(d.get("moves_ko", ""))
        mko = html.escape(mko_raw)
        mraw = str(d.get("moves_raw", ""))
        if mraw and mraw != mko_raw:
            mko_tag = ('<p class="movekind" title="'
                       + html.escape(mraw, quote=True) + '">' + mko + "</p>")
        else:
            mko_tag = f'<p class="movekind">{mko}</p>'
        idx = d.get("index", 0)
        try:
            if isinstance(idx, bool):
                raise ValueError("bool")
            idx_s = str(int(idx))
        except (TypeError, ValueError):
            idx_s = str(idx)
        idx_s = html.escape(idx_s, quote=False)
        try:
            scene_html = scene_card_html(d)
        except Exception:
            scene_html = ""
        if not isinstance(scene_html, str):
            scene_html = ""
        try:
            notice_block = notice_html_for(d)
        except Exception:
            notice_block = ""
        if not isinstance(notice_block, str):
            notice_block = ""
        mline = d.get("misread_line", "")
        if isinstance(mline, str) and mline:
            mis_html = ('<p class="misread">' + html.escape(mline) + "</p>")
        else:
            mis_html = ""
        if d.get("first_alert"):
            fa_badge = ' <span class="badge">첫 경보</span>'
        else:
            fa_badge = ""
        moments.append(
            '<article class="moment">'
            f'<p class="quote">#{idx_s} {disp} — "{quote}"{fa_badge}</p>'
            f'<p class="line">{line}</p>'
            f"{mko_tag}"
            f"{mis_html}"
            f"{scene_html}</article>"
            f"{notice_block}"
        )
    if moments:
        moments_html = "".join(moments)
    else:
        moments_html = '<p class="note">결정적 장면 없음</p>'
    lesson = str(board.get("lesson", "") or "")
    if lesson:
        src = str(board.get("lesson_source", "") or "")
        if src:
            lesson_html = (
                "<section><h2>한 줄 교훈 <small>Lesson</small></h2>"
                "<blockquote>" + html.escape(lesson)
                + ' <span class="lsrc">— ' + html.escape(src)
                + "</span></blockquote></section>"
            )
        else:
            lesson_html = (
                "<section><h2>한 줄 교훈 <small>Lesson</small></h2>"
                "<blockquote>" + html.escape(lesson)
                + "</blockquote></section>"
            )
    else:
        lesson_html = ""
    dash = html.escape(str(board.get("dashboard", "dashboard.html")),
                       quote=True)
    audio_src = board.get("audio_src")
    if audio_src:
        audio_html = '<audio controls src="' + html.escape(
            str(audio_src), quote=True) + '"></audio>'
    else:
        audio_html = ""
    lang_attr = "en" if str(board.get("lang", "ko")).lower().startswith("en") else "ko"
    has_panel = bool(board.get("panel"))
    if has_panel:
        badge_html = ""
        strip = [s for s in (board.get("misread_strip") or [])
                 if isinstance(s, str) and s]
        if strip:
            lis = "".join("<li>" + html.escape(s, quote=False) + "</li>"
                          for s in strip)
            strip_html = (
                "<section><h2>오해 지점 <small>Misread turns</small></h2>"
                f'<ul class="strip">{lis}</ul></section>\n'
            )
        else:
            strip_html = (
                "<section><h2>오해 지점 <small>Misread turns</small></h2>"
                '<p class="note">오해 지점 없음</p></section>\n'
            )
        try:
            mis_n = int(board.get("misread_count", 0))
        except (TypeError, ValueError):
            mis_n = 0
        try:
            mis_total = int(board.get("misread_total", 0))
        except (TypeError, ValueError):
            mis_total = 0
        count_html = (f'<p class="note">어긋난 턴 {mis_n} / {mis_total}</p>')
    else:
        badge_html = '<p><span class="badge">관계 축만</span></p>'
        strip_html = ""
        count_html = ""
    case_html = alerts.case_block_html(board.get("case"))
    alert_html = alerts.alert_strip_html(
        board.get("alerts"), outcome_turn=board.get("outcome_turn"),
        genre=board.get("alert_genre"))
    intent_html = alerts.intent_table_html(board.get("intent_rows"))
    source_html = alerts.case_source_html(board.get("case"))
    vsub = board.get("verdict_sub", "")
    if isinstance(vsub, str) and vsub:
        verdict_sub_html = '<p class="sub">' + html.escape(vsub) + "</p>"
    else:
        verdict_sub_html = ""
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{lang_attr}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>관계 보드 — {item_id}</title>\n"
        "<style>" + _CSS + "</style>\n</head>\n<body>\n"
        "<h1>관계 보드 <small>Relationship board</small></h1>\n"
        f'<p class="note">{item_id}</p>\n'
        + case_html +
        "<section><h2>한 문장 <small>Verdict</small></h2>"
        f'<p class="verdict">{verdict}</p>'
        + verdict_sub_html + badge_html +
        "<p class=\"sub\">관계 점수는 상대 눈에 비친 점수입니다 "
        "(50 = 보통)</p></section>\n"
        "<section><h2>관계 점수 <small>Score</small></h2>"
        '<div class="cards">' + "".join(cards_html) + "</div>"
        + count_html + "</section>\n"
        + strip_html
        + alert_html +
        "<section><h2>흐름 <small>Tide</small></h2>"
        + tide_svg(board.get("momentum"), board.get("flags"))
        + _legend_html(cards) + _others_html(board) + audio_html
        + "</section>\n"
        "<section><h2>결정적 순간 <small>Decisive moments</small></h2>"
        + moments_html + "</section>\n"
        + intent_html
        + lesson_html +
        "<footer><p>모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다</p>"
        f'<p>자세한 대시보드: <a href="{dash}">{dash}</a></p>'
        + source_html + "</footer>\n"
        "</body>\n</html>\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-board2")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--casino", default=None)
    ap.add_argument("--names", default=None,
                    help="JSON object mapping speaker labels to display names")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--suggests", "--suggestions", "--suggest",
                    dest="suggests", default=None)
    ap.add_argument("--culture", default="mix",
                    choices=["east", "west", "mix"])
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
    ap.add_argument("--panel", default=None,
                    help="0-100/dist5 panel record of the same dialogue")
    ap.add_argument("--intent", default=None,
                    help="JSON object {turn: intent note} or a path to one")
    ap.add_argument("--case", default=None,
                    help="JSON {title, outcome, source_url, note} or a path")
    ap.add_argument("--outcome-turn", type=int, default=None,
                    help="outcome turn index (default: the last turn)")
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
    names = None
    if args.names:
        try:
            names = json.loads(args.names)
        except (TypeError, ValueError):
            names = None
    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    board = build_board2(
        item,
        commentary,
        outcomes=outcomes,
        audio_src=resolve_audio_src(args.audio, out_path),
        names=names,
        scenes=args.scenes,
        suggests=args.suggests,
        culture=args.culture,
        lang=args.lang,
        panel=args.panel,
        intent=args.intent,
        case=args.case,
        outcome_turn=args.outcome_turn,
    )
    out_path.write_text(render_html(board), encoding="utf-8")
