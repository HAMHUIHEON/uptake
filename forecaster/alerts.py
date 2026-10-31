"""Per-turn alert timeline for board2 (task58): when could this have been seen coming?

Each turn gets an alert score and a level:

* base: with ``--panel``, the task57 misread severity (share of readings
  outside the invited set); without it, the pairwise z-drop
  (z <= -2 -> 1.0, z <= -1 -> 0.5, else 0.0, preferring ``z_resid``).
* score = base * :data:`STAKES_WEIGHT` for the record genre (default 0.5).
* level: score < 0.25 관심, < 0.5 주의, < 0.75 경계, else 심각.

The **first alert** is the first turn at 경계 or above; the **lead time**
counts turns (positions) from it to the outcome turn (``--outcome-turn N``
or the last turn), plus clock time when effects carry ``time`` (seconds or
``M:SS`` / ``H:MM:SS``). Stdlib only, no record I/O beyond JSON helpers.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

#: Stakes weight by dialogue genre. High-stakes face-to-face genres
#: (aviation ATC, medical) count in full; everyday genres are downweighted.
#: Unknown or missing genres fall back to ``DEFAULT_STAKES_WEIGHT``.
STAKES_WEIGHT = {
    "aviation_atc": 1.0,
    "medical": 1.0,
    "meeting": 0.6,
    "negotiation": 0.4,
}

#: Weight for genres outside :data:`STAKES_WEIGHT` (and missing genres).
DEFAULT_STAKES_WEIGHT = 0.5

#: Alert levels from calm to severe.
LEVELS = ("관심", "주의", "경계", "심각")

#: Tick color per level for the strip above the tide (light theme).
ALERT_COLORS = {
    "관심": "#9aa4b2",
    "주의": "#d29922",
    "경계": "#f0883e",
    "심각": "#d73a49",
}

_LEVEL_RANK = {lv: i for i, lv in enumerate(LEVELS)}
_FIRST_RANK = _LEVEL_RANK["경계"]


def _is_num(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _round_half_up(value, default=50) -> int:
    try:
        return int(math.floor(float(value) + 0.5))
    except (TypeError, ValueError):
        return default


def stakes_weight(genre) -> float:
    """Stakes weight for a genre (0.5 for unknown or missing genres)."""
    if isinstance(genre, str) and genre.strip() in STAKES_WEIGHT:
        return float(STAKES_WEIGHT[genre.strip()])
    return float(DEFAULT_STAKES_WEIGHT)


def base_from_z(z) -> float:
    """Pairwise z-drop base: z <= -2 -> 1.0, z <= -1 -> 0.5, else 0.0."""
    try:
        zf = float(z)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(zf):
        return 0.0
    if zf <= -2.0:
        return 1.0
    if zf <= -1.0:
        return 0.5
    return 0.0


def level_of(score) -> str:
    """Alert level for a score: 관심 / 주의 / 경계 / 심각."""
    try:
        sf = float(score)
    except (TypeError, ValueError):
        return "관심"
    if isinstance(sf, float) and math.isnan(sf):
        return "관심"
    if sf < 0.25:
        return "관심"
    if sf < 0.5:
        return "주의"
    if sf < 0.75:
        return "경계"
    return "심각"


def _z_of(turn) -> float:
    """Panel surprise behind one turn: z_resid, falling back to z."""
    if not isinstance(turn, dict):
        return 0.0
    for key in ("z_resid", "z"):
        if turn.get(key) is not None:
            try:
                return float(turn[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _index_map(entries, index_keys=("index",)) -> dict:
    """Map int index -> entry for a list or dict of turn-like records."""
    out: dict = {}
    if isinstance(entries, dict):
        items = []
        for key, val in entries.items():
            if isinstance(key, bool):
                continue
            try:
                items.append((int(key), val))
            except (TypeError, ValueError):
                continue
    else:
        items = []
        for pos, val in enumerate(entries or []):
            idx = val.get("index", pos) if isinstance(val, dict) else pos
            if isinstance(idx, bool) or not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = pos
            items.append((idx, val))
    for idx, val in items:
        if idx not in out:
            out[idx] = val
    return out


def parse_time(value):
    """Clock time in seconds, or None when absent or unparseable.

    Accepts seconds (int/float, or a numeric string) and ``M:SS`` /
    ``H:MM:SS`` strings (seconds may be fractional).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            secs = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(secs) or secs < 0:
            return None
        return float(secs)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        secs = float(text)
    except ValueError:
        secs = None
    if secs is not None:
        if not math.isfinite(secs) or secs < 0:
            return None
        return float(secs)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(p.strip()) for p in parts]
    except ValueError:
        return None
    if any(not math.isfinite(n) or n < 0 for n in nums):
        return None
    if len(nums) == 2:
        mins, secs = nums
        if secs >= 60.0:
            return None
        return mins * 60.0 + secs
    hrs, mins, secs = nums
    if mins >= 60.0 or secs >= 60.0:
        return None
    return hrs * 3600.0 + mins * 60.0 + secs


def format_duration(seconds) -> str:
    """Korean clock span: 511 -> "8분 31초", 45 -> "45초"."""
    try:
        total = int(math.floor(float(seconds) + 0.5))
    except (TypeError, ValueError):
        return ""
    if total < 0:
        return ""
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if hrs:
        parts.append(f"{hrs}시간")
    if mins or hrs:
        parts.append(f"{mins}분")
    parts.append(f"{secs}초")
    return " ".join(parts)


def compute_alerts(turns, genre=None, analyses=None, times=None,
                   weight=None) -> list[dict]:
    """Per-turn alert entries in input order.

    ``turns`` are dicts with ``index`` and (without a panel) ``z`` /
    ``z_resid``. ``analyses`` is None for z-drop mode, else the task57
    per-turn analyses (list or {index: analysis}) whose ``severity`` is
    the base. ``times`` maps turn index -> raw ``time`` value.
    """
    if _is_num(weight):
        w = float(weight)
    else:
        w = stakes_weight(genre)
    amap = None if analyses is None else _index_map(analyses)
    tmap = _index_map(times) if isinstance(times, (dict, list)) else {}
    out: list[dict] = []
    for pos, turn in enumerate(turns or []):
        if not isinstance(turn, dict):
            continue
        idx = turn.get("index", pos)
        if isinstance(idx, bool) or not isinstance(idx, int):
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = pos
        if amap is not None:
            ana = amap.get(idx)
            sev = ana.get("severity", 0.0) if isinstance(ana, dict) else 0.0
            try:
                base = max(0.0, min(1.0, float(sev)))
            except (TypeError, ValueError):
                base = 0.0
        else:
            base = base_from_z(_z_of(turn))
        score = base * w
        secs = None
        if idx in tmap:
            raw = tmap[idx]
            raw = raw.get("time", raw) if isinstance(raw, dict) else raw
            secs = parse_time(raw)
        if secs is None and turn.get("time") is not None:
            secs = parse_time(turn.get("time"))
        out.append({"index": idx, "base": float(base), "weight": float(w),
                    "score": float(score), "level": level_of(score),
                    "seconds": secs})
    return out


def first_alert(alerts):
    """First alert at 경계 or above, or None when levels stay lower."""
    for a in alerts or []:
        if (isinstance(a, dict)
                and _LEVEL_RANK.get(a.get("level"), 0) >= _FIRST_RANK):
            return a
    return None


def lead_time(alerts, outcome_turn=None):
    """Lead from the first alert to the outcome turn, or None.

    The outcome turn is ``outcome_turn`` when it names a listed turn,
    else the last turn. ``turns`` counts positions between the two;
    ``seconds`` is the clock span when both ends carry ``time``.
    """
    alerts = [a for a in (alerts or []) if isinstance(a, dict)]
    if not alerts:
        return None
    first = first_alert(alerts)
    if first is None:
        return None
    by_index = {}
    for pos, a in enumerate(alerts):
        if a.get("index") not in by_index:
            by_index[a.get("index")] = pos
    out_pos = len(alerts) - 1
    if outcome_turn is not None and not isinstance(outcome_turn, bool):
        try:
            key = int(outcome_turn)
        except (TypeError, ValueError):
            key = None
        if key is not None and key in by_index:
            out_pos = by_index[key]
    first_pos = alerts.index(first)
    turns = max(0, out_pos - first_pos)
    t0 = first.get("seconds")
    t1 = alerts[out_pos].get("seconds")
    if (_is_num(t0) and _is_num(t1) and float(t1) >= float(t0)
            and out_pos >= first_pos):
        secs = float(t1) - float(t0)
    else:
        secs = None
    text = f"결과까지 {turns}턴"
    if secs is not None:
        text += f" / {format_duration(secs)}"
    return {"first_index": first.get("index"),
            "first_level": first.get("level"),
            "outcome_index": alerts[out_pos].get("index"),
            "turns": turns, "seconds": secs, "text": text}


def flag_text(first, lead) -> str:
    """First-alert flag line: "#19 경계 — 결과까지 15턴 / 8분 31초"."""
    if not isinstance(first, dict) or not isinstance(lead, dict):
        return ""
    try:
        idx = first.get("index", lead.get("first_index", ""))
        if isinstance(idx, bool):
            raise ValueError("bool")
        idx_s = str(int(idx))
    except (TypeError, ValueError):
        idx_s = str(first.get("index", ""))
    level = first.get("level", lead.get("first_level", ""))
    text = lead.get("text", "")
    if not idx_s or not level or not text:
        return ""
    return f"#{idx_s} {level} — {text}"


def pairwise_verdict(z) -> str:
    """Pairwise reception word: 잘 / 보통 / 나쁘게 (mirrors game.reader_verdict)."""
    try:
        zf = float(z)
    except (TypeError, ValueError):
        return "보통"
    if not math.isfinite(zf):
        return "보통"
    if zf >= 0.75:
        return "잘"
    if zf <= -0.75:
        return "나쁘게"
    return "보통"


def _normalize_intent_map(data) -> dict:
    out: dict = {}
    if not isinstance(data, dict):
        return out
    for key, val in data.items():
        if isinstance(key, bool):
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(val, str):
            continue
        if val.strip() and idx not in out:
            out[idx] = val
    return out


def load_intent(source) -> dict:
    """Turn-intent notes {turn index: text} from a dict, JSON, or path.

    ``source`` may be the mapping itself, a JSON object string, or a path
    to a JSON file holding the object. Anything unreadable yields {}.
    """
    if source is None:
        return {}
    if isinstance(source, dict):
        return _normalize_intent_map(source)
    if isinstance(source, (str, Path)):
        text = None
        try:
            if Path(str(source)).is_file():
                text = Path(str(source)).read_text(encoding="utf-8")
        except OSError:
            text = None
        if text is None and isinstance(source, str):
            text = source
        if text is None:
            return {}
        try:
            data = json.loads(text)
        except ValueError:
            return {}
        return _normalize_intent_map(data)
    return {}


def load_case(source):
    """Case header {title, outcome, source_url, note} from dict/JSON/path.

    Returns None when no case was given or the source is unreadable.
    """
    if source is None:
        return None
    data = None
    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, Path)):
        text = None
        try:
            if Path(str(source)).is_file():
                text = Path(str(source)).read_text(encoding="utf-8")
        except OSError:
            text = None
        if text is None and isinstance(source, str):
            text = source
        if text is None:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    out = {}
    for key in ("title", "outcome", "source_url", "note"):
        val = data.get(key, "")
        out[key] = val if isinstance(val, str) else str(val or "")
    return out


def cockpit_notes(item, panel=None) -> dict:
    """Intent notes turns carry as ``cockpit_before`` ({index: text}).

    Scans the item record's effects and turns, then the panel record's
    effects; the first non-empty note per turn index wins.
    """
    out: dict = {}
    pools: list = []
    if isinstance(item, dict):
        for key in ("effects", "turns"):
            val = item.get(key)
            if isinstance(val, list):
                pools.append(val)
    if isinstance(panel, dict):
        val = panel.get("effects")
        if isinstance(val, list):
            pools.append(val)
    for pool in pools:
        for pos, entry in enumerate(pool):
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index", entry.get("turn_index", pos))
            if isinstance(idx, bool) or not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = pos
            note = entry.get("cockpit_before")
            if (isinstance(note, str) and note.strip()
                    and idx not in out):
                out[idx] = note
    return out


def _standing_pair(trail, pos, speaker) -> tuple:
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
        return (_round_half_up(before_v), _round_half_up(after_v))
    except (TypeError, ValueError):
        return (50, 50)


def intent_rows(item=None, view=None, analyses=None, intent=None,
                trail=None, panel=None) -> list[dict]:
    """One row per turn carrying an intent note, in turn order.

    Notes come from ``cockpit_before`` fields (see :func:`cockpit_notes`),
    overridden per turn by the explicit ``intent`` map ({turn: text}).
    The 읽힌 것 cell is the task57 majority next-action (``맞선다(67%)``)
    when a panel analysis with a majority exists, else the pairwise
    verdict word with the standing change (``나쁘게 50 → 8``).
    """
    notes = cockpit_notes(item, panel)
    notes.update(load_intent(intent))
    if not notes:
        return []
    view = view if isinstance(view, dict) else {}
    vturns = [t for t in (view.get("turns") or []) if isinstance(t, dict)]
    v_by_index: dict = {}
    pos_by_index: dict = {}
    for pos, t in enumerate(vturns):
        idx = t.get("index", pos)
        if isinstance(idx, bool) or not isinstance(idx, int):
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = pos
        if idx not in v_by_index:
            v_by_index[idx] = t
            pos_by_index[idx] = pos
    eff_by_index: dict = {}
    if isinstance(item, dict) and isinstance(item.get("effects"), list):
        for pos, e in enumerate(item["effects"]):
            if not isinstance(e, dict):
                continue
            idx = e.get("index", pos)
            if isinstance(idx, bool) or not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = pos
            if idx not in eff_by_index:
                eff_by_index[idx] = e
    amap = _index_map(analyses) if analyses is not None else {}
    rows: list[dict] = []
    for idx in sorted(notes):
        vt = v_by_index.get(idx, {})
        eff = eff_by_index.get(idx, {})
        ana = amap.get(idx, {})
        if not isinstance(ana, dict):
            ana = {}
        speaker = vt.get("speaker") or eff.get("speaker") or ""
        if not isinstance(speaker, str):
            speaker = str(speaker)
        said = vt.get("text") or eff.get("text") or ana.get("text") or ""
        if not isinstance(said, str):
            said = str(said)
        majority = ana.get("majority") if ana else None
        ko = ana.get("majority_ko") if ana else ""
        if isinstance(majority, str) and majority and ko:
            try:
                pct = int(ana.get("pct",
                                  round(float(ana.get("share", 0.0)) * 100)))
            except (TypeError, ValueError):
                pct = 0
            read = f"{ko}({pct}%)"
        else:
            verdict = pairwise_verdict(_z_of(vt) if vt else _z_of(eff))
            if isinstance(trail, list) and trail and idx in pos_by_index:
                before, after = _standing_pair(trail, pos_by_index[idx],
                                               speaker)
                read = f"{verdict} {before} → {after}"
            else:
                read = verdict
        rows.append({"index": idx, "speaker": speaker,
                     "intent": notes[idx], "said": said, "read": read})
    return rows


def alert_strip_html(alerts, outcome_turn=None, genre=None) -> str:
    """Alert-timeline section: one colored tick per turn + first-alert flag."""
    alerts = [a for a in (alerts or []) if isinstance(a, dict)]
    if not alerts:
        return ""
    ticks = []
    for a in alerts:
        level = a.get("level", "관심")
        color = ALERT_COLORS.get(level, "#9aa4b2")
        try:
            idx = a.get("index", "")
            idx_s = ("?" if isinstance(idx, bool)
                     else str(int(idx)))
        except (TypeError, ValueError):
            idx_s = str(a.get("index", "?"))
        ticks.append(
            '<span class="tick" data-level="'
            + html.escape(str(level), quote=True) + '" title="#'
            + html.escape(idx_s, quote=True) + " "
            + html.escape(str(level), quote=True) + '" style="background:'
            + html.escape(str(color), quote=True) + '"></span>'
        )
    lead = lead_time(alerts, outcome_turn=outcome_turn)
    if lead is not None:
        first = first_alert(alerts)
        flag = html.escape(flag_text(first, lead))
        flag_html = f'<p class="alert-flag">{flag}</p>'
    else:
        flag_html = '<p class="note">경계 이상 없음</p>'
    try:
        w = float(alerts[0].get("weight", DEFAULT_STAKES_WEIGHT))
    except (TypeError, ValueError):
        w = float(DEFAULT_STAKES_WEIGHT)
    note = f'가중치 {w:g}'
    if isinstance(genre, str) and genre.strip():
        note += f" · 장르 {html.escape(genre.strip())}"
    return (
        "<section><h2>경보 타임라인 <small>Alert timeline</small></h2>"
        '<div class="alert-strip" role="img" '
        'aria-label="per-turn alert levels">' + "".join(ticks) + "</div>"
        + flag_html + f'<p class="note">{note}</p></section>\n'
    )


def intent_table_html(rows) -> str:
    """의도 vs 전달 vs 해석 section: 뜻한 것 / 말한 것 / 읽힌 것."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return ""
    trs = []
    for r in rows:
        try:
            idx = r.get("index", "")
            idx_s = "?" if isinstance(idx, bool) else str(int(idx))
        except (TypeError, ValueError):
            idx_s = str(r.get("index", "?"))
        trs.append(
            "<tr><th scope=\"row\">#" + html.escape(idx_s) + "</th>"
            "<td>" + html.escape(str(r.get("intent", ""))) + "</td>"
            "<td>" + html.escape(str(r.get("said", ""))) + "</td>"
            "<td>" + html.escape(str(r.get("read", ""))) + "</td></tr>"
        )
    return (
        "<section><h2>의도 vs 전달 vs 해석 "
        "<small>Intent vs said vs read</small></h2>"
        '<table class="intent"><thead><tr><th scope="col">턴</th>'
        '<th scope="col">뜻한 것</th>'
        '<th scope="col">말한 것</th>'
        "<th scope=\"col\">읽힌 것</th></tr></thead>"
        "<tbody>" + "".join(trs) + "</tbody></table></section>\n"
    )


def case_block_html(case) -> str:
    """사건과 결과 header block from --case {title, outcome, source_url, note}."""
    if not isinstance(case, dict):
        return ""
    title = case.get("title", "")
    outcome = case.get("outcome", "")
    note = case.get("note", "")
    url = case.get("source_url", "")
    if not any(isinstance(v, str) and v.strip()
               for v in (title, outcome, note, url)):
        return ""
    parts = ["<section><h2>사건과 결과 <small>Case</small></h2>"]
    if isinstance(title, str) and title.strip():
        parts.append('<p class="case-title">'
                     + html.escape(title.strip()) + "</p>")
    if isinstance(outcome, str) and outcome.strip():
        parts.append("<p>" + html.escape(outcome.strip()) + "</p>")
    if isinstance(note, str) and note.strip():
        parts.append('<p class="note">' + html.escape(note.strip()) + "</p>")
    if isinstance(url, str) and url.strip():
        href = html.escape(url.strip(), quote=True)
        parts.append('<p class="note">출처: <a href="' + href + '">'
                     + html.escape(url.strip()) + "</a></p>")
    parts.append("</section>\n")
    return "".join(parts)


def case_source_html(case) -> str:
    """Footer source-link paragraph for a case, or "" without a URL."""
    if not isinstance(case, dict):
        return ""
    url = case.get("source_url", "")
    if not (isinstance(url, str) and url.strip()):
        return ""
    href = html.escape(url.strip(), quote=True)
    return ('<p>출처: <a href="' + href + '">'
            + html.escape(url.strip()) + "</a></p>")
