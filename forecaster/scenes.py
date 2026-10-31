"""Blunder notice scenes + one line of counsel from the classics (task55).

Builds the demo scene file from the owner's JSONL (task50 §build plus
``문화``/``상황_한줄``), asks the model once per scene for a one-line
``교훈``/``교훈_en`` (cached, idempotent), picks one scene per blunder
turn (deterministic, culture-aware, never repeated on a page) and
renders the notice block used by board2 and the play page.

Stdlib only plus sibling modules (no third-party imports).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

# --- auditable constants -----------------------------------------------------

# Blunder move kind -> candidate scene 주유형 (SPEC_task50 "received badly").
# Full type names as in the corpus; matching also accepts the code prefix
# (e.g. "C2" matches "C2 체면 위협").
TYPE_FOR_BLUNDER: dict[str, list[str]] = {
    "reject": ["C2 체면 위협", "C1 권력거리"],
    "pressure": ["C2 체면 위협", "C1 권력거리"],
    "challenge": ["C2 체면 위협", "C1 권력거리"],
    "propose": ["B1 양의 격률", "B4 간접화행"],
    "counter": ["B1 양의 격률", "B4 간접화행"],
    "ask": ["B1 양의 격률", "B4 간접화행"],
    "deflect": ["B3 반어·풍자", "E3 침묵·무응답"],
    "joke": ["B3 반어·풍자", "E3 침묵·무응답"],
    "smalltalk": ["B3 반어·풍자", "E3 침묵·무응답"],
    "appeal_need": ["E1 확증 편향", "D2 가치 충돌"],
    "appeal_fairness": ["E1 확증 편향", "D2 가치 충돌"],
    "probe": ["F1 전략적 모호함", "F2 설계된 오독"],
    "reveal_preference": ["F1 전략적 모호함", "F2 설계된 오독"],
    "empathize": ["E2 감정 투사"],
    "repair": ["E2 감정 투사"],
    "concession": ["E2 감정 투사"],
    "concede": ["E2 감정 투사"],
    "accept": ["E2 감정 투사"],
    "inform": ["B4 간접화행"],
    "cooperate": ["B4 간접화행"],
    "praise": ["B4 간접화행"],
    "other": ["B4 간접화행"],
}

# Alias for the task50 name (badly-received side only).
MOVE_TO_TYPES: dict[str, list[str]] = dict(TYPE_FOR_BLUNDER)

# Counsel prompt (module constant, auditable exact wording). One model call
# per scene; the model returns JSON with 교훈 + 교훈_en.
COUNSEL_PROMPT = (
    "You write one line of counsel from a classic scene.\n"
    "Read the told scene (one sentence, outcome included) and its source.\n"
    "Offer perspective in one line. Never claim the situations are the same; "
    "counsel, not analogy.\n"
    "Korean (교훈): one sentence, at most 60 Korean characters, plain, "
    "no imperative \"하라\" tone "
    "(prefer \"…은 …에 달려 있다\" / \"…는 …에서 갈린다\"), no modern jargon.\n"
    "English (교훈_en): the same counsel in English, at most 110 characters.\n"
    "Reply with JSON only (no other text): "
    '{"교훈": "<korean one-liner>", "교훈_en": "<english one-liner>"}.'
)
COUNSEL_SYSTEM = COUNSEL_PROMPT

EAST_KEYWORDS = ("동양고전", "실록", "한국", "중국", "일본")
EAST_DOMAINS = ("zh.wikisource", "sillok", "ko.wikisource")
WEST_DOMAINS = ("gutenberg", "en.wikipedia")

BLUNDER_Z = -1.5
BLUNDER_DROP = 15

SCENE_LINE_LIMIT = 90
COUNSEL_KO_LIMIT = 60
COUNSEL_EN_LIMIT = 110

_SENT_SPLIT = re.compile(r"(.+?[.!?。！？]+)(?:\s+|$)")


# --- culture + one-line scene -------------------------------------------------

def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def infer_culture(row) -> str:
    """'동' for East-Asian classics/annals, else '서'.

    East when 분야/계열/작품 mentions 동양고전·실록·한국·중국·일본 or the
    source domain is zh/ko.wikisource or sillok; everything else is '서'
    (gutenberg / en.wikipedia are the explicit western domains).
    """
    if not isinstance(row, dict):
        return "서"
    texts = " ".join([
        _as_text(row.get("분야", "")),
        _as_text(row.get("계열", "")),
        _as_text(row.get("작품", "")),
    ])
    src = row.get("원문출처", "")
    if isinstance(src, (list, tuple)):
        src_text = " ".join(_as_text(v) for v in src)
    else:
        src_text = _as_text(src)
    if any(k in texts for k in EAST_KEYWORDS):
        return "동"
    low = src_text.lower()
    if any(d in low for d in EAST_DOMAINS):
        return "동"
    return "서"


def culture_of(row) -> str:
    """Alias for infer_culture; honours a stored 문화 value when present."""
    if isinstance(row, dict):
        got = row.get("문화", "")
        if got in ("동", "서"):
            return got
        # English stored forms from --culture flags.
        if got in ("east", "동양"):
            return "동"
        if got in ("west", "서양"):
            return "서"
    return infer_culture(row)


def one_sentence(text, limit=SCENE_LINE_LIMIT) -> str:
    """First sentence of text, whitespace-collapsed, capped at limit chars."""
    try:
        lim = int(limit)
    except (TypeError, ValueError):
        lim = SCENE_LINE_LIMIT
    if lim <= 0:
        lim = SCENE_LINE_LIMIT
    s = " ".join(_as_text(text).split())
    if not s:
        return ""
    m = _SENT_SPLIT.match(s)
    first = m.group(1).strip() if m else s
    first = first.strip()
    if len(first) > lim:
        if lim > 1:
            return first[: lim - 1].rstrip() + "…"
        return first[:lim]
    return first


def make_scene_line(row) -> str:
    """상황_한줄: 장면 trimmed to one sentence ≤ 90 chars (outcome incl.).

    Falls back to 해석 when 장면 is blank.
    """
    if not isinstance(row, dict):
        return ""
    line = one_sentence(row.get("장면", ""), SCENE_LINE_LIMIT)
    if line:
        return line
    return one_sentence(row.get("해석", ""), SCENE_LINE_LIMIT)


# --- build --------------------------------------------------------------------

_COPY_FIELDS = ("id", "계열", "작품", "작가", "위치", "장면", "발화",
                "말한사람", "받은사람", "의도", "해석", "주유형",
                "오해발생", "대조종류")


def _copy_row(row: dict) -> dict:
    out: dict = {}
    for key in _COPY_FIELDS:
        if key in row:
            out[key] = row.get(key)
        elif key == "오해발생":
            out[key] = False
        else:
            out[key] = ""
    src = row.get("원문출처", [])
    if isinstance(src, list):
        out["원문출처"] = list(src)
    elif isinstance(src, str) and src:
        out["원문출처"] = [src]
    else:
        out["원문출처"] = []
    ctx = row.get("맥락", {})
    if not isinstance(ctx, dict):
        ctx = {}
    out["맥락"] = {
        "관계": _as_text(ctx.get("관계", "")),
        "권력거리": _as_text(ctx.get("권력거리", "")),
    }
    if "데모적합" in row:
        out["데모적합"] = row.get("데모적합", "")
    out["문화"] = infer_culture(row)
    out["상황_한줄"] = make_scene_line(row)
    return out


def build_scenes(src, out) -> dict:
    """Filter + copy the owner's JSONL to a sorted scenes.json file.

    Keeps only 검증 == "확인" and 데모적합 in {"상","중"}. Returns counts.
    """
    try:
        text = Path(src).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read src {src}: {exc}")
    kept: list[dict] = []
    n_total = n_bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        n_total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        if not isinstance(row, dict):
            continue
        if row.get("검증") != "확인":
            continue
        if row.get("데모적합") not in ("상", "중"):
            continue
        kept.append(_copy_row(row))
    kept.sort(key=lambda s: str(s.get("id", "")))
    out_p = Path(out)
    if str(out_p.parent) and str(out_p.parent) != ".":
        out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    per_series: dict[str, int] = {}
    per_type: dict[str, int] = {}
    for s in kept:
        ks = _as_text(s.get("계열", "")) or "(none)"
        per_series[ks] = per_series.get(ks, 0) + 1
        kt = _as_text(s.get("주유형", "")) or "(none)"
        per_type[kt] = per_type.get(kt, 0) + 1
    counts = {
        "n_total": int(n_total),
        "n_kept": int(len(kept)),
        "n_dropped": int(n_total - len(kept) - n_bad),
        "n_bad_json": int(n_bad),
        "per_계열": per_series,
        "per_주유형": per_type,
    }
    print(f"kept {len(kept)}/{n_total}")
    for key in sorted(per_series):
        print(f"계열 {key}: {per_series[key]}")
    for key in sorted(per_type):
        print(f"주유형 {key}: {per_type[key]}")
    return counts


def build(src, out) -> dict:
    """Alias for build_scenes."""
    return build_scenes(src, out)


def load_scenes(path) -> list[dict]:
    """Load a scenes.json file (list, or {"scenes": [...]})."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read scenes {path}: {exc}")
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise SystemExit(f"bad scenes {path}: not a list")
    return [s for s in data if isinstance(s, dict)]


# --- blunder test ---------------------------------------------------------------

def _num(value, default=None):
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_blunder(turn) -> bool:
    """True for a blunder turn: z_resid ≤ −1.5 or standing drop ≥ 15."""
    if not isinstance(turn, dict):
        return False
    zr = _num(turn.get("z_resid"), None)
    if zr is not None and zr <= BLUNDER_Z:
        return True
    before = _num(turn.get("standing_before",
                           turn.get("before", turn.get("score_before"))), None)
    after = _num(turn.get("standing_after",
                          turn.get("after", turn.get("score_after"))), None)
    if before is not None and after is not None:
        if before - after >= BLUNDER_DROP:
            return True
    return False


def candidate_types(moves) -> list[str]:
    """Candidate 주유형 for one move kind or a list of moves (deduped)."""
    if isinstance(moves, str):
        seq = [moves]
    else:
        try:
            seq = list(moves or [])
        except TypeError:
            seq = []
    out: list[str] = []
    for mv in seq:
        if not isinstance(mv, str) or not mv:
            continue
        for cand in TYPE_FOR_BLUNDER.get(mv, TYPE_FOR_BLUNDER["other"]):
            if cand not in out:
                out.append(cand)
    if not out:
        out = list(TYPE_FOR_BLUNDER["other"])
    return out


def _type_matches(scene_type, candidates) -> bool:
    st = _as_text(scene_type).strip()
    if not st:
        return False
    for cand in candidates or []:
        c = _as_text(cand).strip()
        if not c:
            continue
        if st == c or st.startswith(c) or c.startswith(st):
            return True
        # Code-prefix match: "C2" matches "C2 체면 위협".
        code = c.split(" ")[0]
        if code and st.startswith(code):
            return True
    return False


def _culture_wanted(culture) -> str | None:
    c = _as_text(culture).strip().lower()
    if c in ("east", "동", "동양"):
        return "동"
    if c in ("west", "서", "서양"):
        return "서"
    return None


def _hash_choice(dialogue_id, turn, n) -> int:
    if n <= 0:
        return 0
    key = f"{dialogue_id}:{turn}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % n


def _moves_of(move) -> list[str]:
    if isinstance(move, dict):
        mv = move.get("moves", move.get("move", []))
        if isinstance(mv, str):
            return [mv]
        try:
            return [m for m in list(mv or []) if isinstance(m, str)]
        except TypeError:
            return []
    if isinstance(move, str):
        return [move]
    try:
        return [m for m in list(move or []) if isinstance(m, str)]
    except TypeError:
        return []


# --- pick -----------------------------------------------------------------------

def pick_scene(move=None, dialogue_id="", turn=0, scenes=None,
               culture="mix", used=None, position=None, **kwargs) -> dict | None:
    """Pick one scene for a blunder turn (deterministic, culture-aware).

    ``move`` is a move kind, a list of kinds, or a turn dict with "moves".
    ``culture`` is east|west|mix; under mix the wanted side alternates by
    ``position`` (even → east, odd → west) or, when position is None, by
    the east/west balance of ``used``. Scenes in ``used`` (ids) are never
    returned. 데모적합 상 is preferred. Falls back from type+culture to
    culture-only to type-only to any unused scene.
    """
    # Back-compat keyword spellings.
    if scenes is None and kwargs.get("scene_list") is not None:
        scenes = kwargs.get("scene_list")
    if dialogue_id == "" and kwargs.get("dialogue") is not None:
        dialogue_id = kwargs.get("dialogue")
    if turn == 0 and kwargs.get("turn_index") is not None:
        try:
            turn = int(kwargs.get("turn_index"))
        except (TypeError, ValueError):
            pass
    if turn == 0 and kwargs.get("index") is not None:
        try:
            turn = int(kwargs.get("index"))
        except (TypeError, ValueError):
            pass
    if isinstance(scenes, (str, Path)):
        try:
            scenes = load_scenes(scenes)
        except SystemExit:
            return None
    pool = [s for s in (scenes or []) if isinstance(s, dict)]
    if not pool:
        # Default built file next to this module (when committed).
        default = Path(__file__).with_name("scenes.json")
        if default.is_file():
            try:
                pool = load_scenes(default)
            except SystemExit:
                pool = []
        if not pool:
            return None
    used_ids: set = set()
    if used:
        try:
            for u in list(used):
                if isinstance(u, dict):
                    uid = u.get("id", "")
                    used_ids.add(str(uid))
                else:
                    used_ids.add(str(u))
        except TypeError:
            pass
    moves = _moves_of(move)
    if not moves and isinstance(move, dict):
        # Single move kind under "move".
        single = move.get("move", "")
        if isinstance(single, str) and single:
            moves = [single]
    cands = candidate_types(moves)
    wanted: str | None
    c = _as_text(culture).strip().lower()
    if c in ("east", "동", "동양"):
        wanted = "동"
    elif c in ("west", "서", "서양"):
        wanted = "서"
    else:  # mix: alternate across the page's notices.
        pos = None
        if position is not None:
            try:
                pos = int(position)
            except (TypeError, ValueError):
                pos = None
        if pos is not None:
            wanted = "동" if pos % 2 == 0 else "서"
        else:
            n_east = n_west = 0
            by_id = {str(s.get("id", "")): s for s in pool}
            for uid in used_ids:
                s = by_id.get(str(uid))
                if s is None:
                    continue
                if culture_of(s) == "동":
                    n_east += 1
                else:
                    n_west += 1
            wanted = "동" if n_east <= n_west else "서"

    def _prefer_top(rows: list[dict]) -> list[dict]:
        top = [s for s in rows if s.get("데모적합") == "상"]
        return top if top else rows

    def _choose(rows: list[dict]) -> dict | None:
        rows = [s for s in rows if str(s.get("id", "")) not in used_ids]
        if not rows:
            return None
        rows = _prefer_top(rows)
        rows = [s for s in rows if str(s.get("id", "")) not in used_ids]
        if not rows:
            return None
        rows.sort(key=lambda s: str(s.get("id", "")))
        return rows[_hash_choice(dialogue_id, turn, len(rows))]

    # 1. type + culture.
    if wanted is not None:
        hit = _choose([s for s in pool
                       if culture_of(s) == wanted
                       and _type_matches(s.get("주유형", ""), cands)])
        if hit is not None:
            return hit
        # 2. culture only (any type).
        hit = _choose([s for s in pool if culture_of(s) == wanted])
        if hit is not None:
            return hit
    # 3. type only (any culture).
    hit = _choose([s for s in pool
                   if _type_matches(s.get("주유형", ""), cands)])
    if hit is not None:
        return hit
    # 4. any unused scene.
    return _choose(list(pool))


def pick(*args, **kwargs) -> dict | None:
    """Alias for pick_scene."""
    return pick_scene(*args, **kwargs)


def pick_for_page(blunders, scenes=None, dialogue_id="", culture="mix",
                  used=None) -> dict:
    """Pick one scene per blunder turn in page order.

    Returns {turn_index: scene}. Under mix the wanted culture alternates
    east/west across the page's notices; scenes never repeat.
    """
    if isinstance(scenes, (str, Path)):
        try:
            pool = load_scenes(scenes)
        except SystemExit:
            return {}
    else:
        pool = [s for s in (scenes or []) if isinstance(s, dict)]
    used_ids: set = set()
    if used:
        try:
            for u in list(used):
                used_ids.add(str(u.get("id", u) if isinstance(u, dict) else u))
        except TypeError:
            pass
    out: dict = {}
    seq = list(blunders or [])
    for pos, b in enumerate(seq):
        if isinstance(b, dict):
            mv = b.get("moves", b.get("move", "other"))
            idx = b.get("index", b.get("turn", pos))
            try:
                idx_i = int(idx)
            except (TypeError, ValueError):
                idx_i = pos
        else:
            mv, idx_i = "other", pos
        scene = pick_scene(mv, dialogue_id, idx_i, pool,
                           culture=culture, used=used_ids, position=pos)
        if scene is not None:
            used_ids.add(str(scene.get("id", "")))
            out[idx_i] = scene
    return out


def pick_notices(*args, **kwargs) -> dict:
    """Alias for pick_for_page."""
    return pick_for_page(*args, **kwargs)


# --- counsel --------------------------------------------------------------------

def build_counsel_messages(scene) -> list[dict]:
    """One-shot counsel messages for a scene (system + user)."""
    scene = scene if isinstance(scene, dict) else {}
    told = _as_text(scene.get("상황_한줄", "")) or make_scene_line(scene)
    work = _as_text(scene.get("작품", ""))
    author = _as_text(scene.get("작가", ""))
    loc = _as_text(scene.get("위치", ""))
    ctx_bits = [b for b in (work, loc, author) if b]
    ctx = " · ".join(ctx_bits)
    user = f"장면: {told}\n출처: {ctx}\n한 줄 교훈을 JSON으로."
    return [
        {"role": "system", "content": COUNSEL_PROMPT},
        {"role": "user", "content": user},
    ]


def _truncate_ko(text, limit=COUNSEL_KO_LIMIT) -> str:
    s = " ".join(_as_text(text).split())
    if len(s) <= limit:
        return s
    if limit > 1:
        return s[: limit - 1].rstrip() + "…"
    return s[:limit]


def _truncate_en(text, limit=COUNSEL_EN_LIMIT) -> str:
    s = " ".join(_as_text(text).split())
    if len(s) <= limit:
        return s
    # Cut at the last space inside the limit when possible.
    cut = s[:limit].rsplit(" ", 1)
    if len(cut) == 2 and len(cut[0]) >= limit - 20:
        return cut[0].rstrip() + "…"
    if limit > 1:
        return s[: limit - 1].rstrip() + "…"
    return s[:limit]


def parse_counsel(raw) -> dict | None:
    """Parse one counsel reply; enforces the length caps. None on failure."""
    if not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    ko = obj.get("교훈", obj.get("ko", obj.get("counsel", "")))
    en = obj.get("교훈_en", obj.get("en", obj.get("counsel_en", "")))
    ko = " ".join(_as_text(ko).split())
    en = " ".join(_as_text(en).split())
    if not ko:
        return None
    return {
        "교훈": _truncate_ko(ko, COUNSEL_KO_LIMIT),
        "교훈_en": _truncate_en(en, COUNSEL_EN_LIMIT),
    }


def _call_model(client, messages, model, meta=None):
    wants_meta = meta is not None and getattr(client, "accepts_meta", False)
    if not wants_meta:
        return client.complete(messages, model, 1.0)
    try:
        return client.complete(messages, model, 1.0, meta=meta)
    except TypeError:
        return client.complete(messages, model, 1.0)


def counsel_scenes(scenes_path, model=None, budget=None, client=None,
                   meter=None) -> dict:
    """Ask once per scene for 교훈/교훈_en; cache in the same file.

    Idempotent: rows that already have a non-blank 교훈 are skipped.
    Without a model/client nothing is called and the file is left alone.
    Returns {"n_total","n_done","n_skipped","n_failed","budget_stop"}.
    """
    from .usage import BudgetExceeded

    scenes = load_scenes(scenes_path)
    n_total = len(scenes)
    n_done = n_skipped = n_failed = 0
    budget_stop = False
    if model is None and client is None:
        return {"n_total": n_total, "n_done": 0,
                "n_skipped": n_total, "n_failed": 0,
                "budget_stop": False}
    m = meter
    if m is None and budget is not None:
        from .usage import Meter

        try:
            m = Meter(budget_usd=float(budget))
        except (TypeError, ValueError):
            m = None
    own_client = False
    if client is None:
        from .reader import MuseClient

        own_client = True
        client = MuseClient(meter=m)
    model_name = model
    if model_name is None:
        from .reader import DEFAULT_MODEL

        model_name = DEFAULT_MODEL
    changed = False
    for scene in scenes:
        if _as_text(scene.get("교훈", "")).strip():
            n_skipped += 1
            continue
        if m is not None:
            try:
                m.check()
            except BudgetExceeded:
                budget_stop = True
                break
        messages = build_counsel_messages(scene)
        meta = {"layer": "scenes-counsel",
                "scene": str(scene.get("id", ""))}
        parsed = None
        for _ in range(2):
            try:
                raw = _call_model(client, messages, model_name, meta)
            except BudgetExceeded:
                budget_stop = True
                break
            except Exception:
                continue
            parsed = parse_counsel(raw)
            if parsed is not None:
                break
        if budget_stop:
            break
        if parsed is None:
            n_failed += 1
            continue
        scene["교훈"] = parsed["교훈"]
        scene["교훈_en"] = parsed["교훈_en"]
        scene["교훈_검토"] = False
        n_done += 1
        changed = True
    if changed:
        out_p = Path(scenes_path)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
    _ = own_client
    return {"n_total": int(n_total), "n_done": int(n_done),
            "n_skipped": int(n_skipped), "n_failed": int(n_failed),
            "budget_stop": bool(budget_stop)}


def counsel(*args, **kwargs) -> dict:
    """Alias for counsel_scenes."""
    return counsel_scenes(*args, **kwargs)


# --- render -----------------------------------------------------------------------

def _ro_particle(number) -> str:
    try:
        ones = abs(int(number)) % 10
    except (TypeError, ValueError):
        return "로"
    if ones in (1, 2, 4, 5, 7, 8, 9):
        return "로"
    return "으로"


def notice_line(display, before, after, lang="ko") -> str:
    """First line of the notice block (plain text, no HTML)."""
    name = display if isinstance(display, str) and display else "화자"
    try:
        b = int(before)
    except (TypeError, ValueError):
        b = 50
    try:
        a = int(after)
    except (TypeError, ValueError):
        a = 50
    if _as_text(lang).strip().lower().startswith("en"):
        return f"⚠ This was read as {b} → {a} for {name}"
    ro = _ro_particle(a)
    return f"⚠ 이 말은 {name}에게 {b} → {a}{ro} 읽혔습니다"


def source_line(scene) -> str:
    """Small source line: 『작품』 위치 · 작가 (parts optional)."""
    scene = scene if isinstance(scene, dict) else {}
    work = _as_text(scene.get("작품", "")).strip()
    loc = _as_text(scene.get("위치", "")).strip()
    author = _as_text(scene.get("작가", "")).strip()
    bits: list[str] = []
    if work:
        bits.append(f"『{work}』")
    if loc:
        bits.append(loc)
    head = " ".join(bits)
    if head and author:
        return f"{head} · {author}"
    return head or author


def scene_line(scene, lang="ko") -> str:
    """Told scene line: 상황_한줄 (English variant when --lang en)."""
    scene = scene if isinstance(scene, dict) else {}
    if _as_text(lang).strip().lower().startswith("en"):
        for key in ("상황_한줄_en", "장면_en", "scene_en"):
            cand = _as_text(scene.get(key, "")).strip()
            if cand:
                return cand
    told = _as_text(scene.get("상황_한줄", "")).strip()
    if told:
        return told
    return make_scene_line(scene)


def counsel_line(scene, lang="ko") -> str:
    """Counsel line: 교훈 (교훈_en when --lang en)."""
    scene = scene if isinstance(scene, dict) else {}
    if _as_text(lang).strip().lower().startswith("en"):
        en = _as_text(scene.get("교훈_en", "")).strip()
        if en:
            return en
    return _as_text(scene.get("교훈", "")).strip()


def render_notice(before=50, after=50, display="", alternatives=None,
                  scene=None, lang="ko", speaker="") -> str:
    """Notice block as HTML (board2 cards + play page share this).

    ``alternatives`` is a list of {text, p_better}; ``scene`` is a scene
    dict (or None). Without alternatives the block shows only the notice
    and counsel lines. No type codes, no "닮은 장면" claim.
    """
    name = display or speaker or "화자"
    lang_s = _as_text(lang).strip().lower()
    en = lang_s.startswith("en")
    try:
        b = int(before)
    except (TypeError, ValueError):
        b = 50
    try:
        a = int(after)
    except (TypeError, ValueError):
        a = 50
    parts: list[str] = ['<div class="notice">']
    parts.append("<p class=\"warn\">"
                 + html.escape(notice_line(name, b, a, lang="en" if en else "ko"))
                 + "</p>")
    alts = [x for x in (alternatives or []) if isinstance(x, dict)]
    if alts:
        head = "Other ways to say it" if en else "이렇게 말해 볼 수도"
        parts.append("<p class=\"althead\">" + html.escape(head) + "</p>")
        parts.append("<ul class=\"alts\">")
        for i, alt in enumerate(alts):
            text = _as_text(alt.get("text", "")).strip()
            if not text:
                continue
            try:
                pct = int(round(float(alt.get("p_better", 0.0)) * 100))
            except (TypeError, ValueError):
                pct = 0
            pct = max(0, min(100, pct))
            if en:
                if i == 0:
                    label = f"({pct}% chance to read better than the original)"
                else:
                    label = f"({pct}%)"
            else:
                if i == 0:
                    label = f"(원문보다 잘 읽힐 확률 {pct}%)"
                else:
                    label = f"({pct}%)"
            parts.append("<li>“" + html.escape(text) + "” "
                         + "<span class=\"pct\">" + html.escape(label)
                         + "</span></li>")
        parts.append("</ul>")
    scene = scene if isinstance(scene, dict) else None
    if scene is not None:
        told = scene_line(scene, lang="en" if en else "ko")
        tip = counsel_line(scene, lang="en" if en else "ko")
        if told and tip:
            parts.append("<p class=\"scene\">" + html.escape(told) + " "
                         + html.escape(tip) + "</p>")
        elif told:
            parts.append("<p class=\"scene\">" + html.escape(told) + "</p>")
        elif tip:
            parts.append("<p class=\"scene\">" + html.escape(tip) + "</p>")
        src = source_line(scene)
        if src:
            parts.append("<p class=\"src\"><small>"
                         + html.escape(src) + "</small></p>")
    parts.append("</div>")
    return "".join(parts)


def render_notice_html(*args, **kwargs) -> str:
    """Alias for render_notice."""
    return render_notice(*args, **kwargs)


def notice_html(*args, **kwargs) -> str:
    """Alias for render_notice."""
    return render_notice(*args, **kwargs)


def render_notice_text(before=50, after=50, display="", alternatives=None,
                       scene=None, lang="ko", speaker="") -> str:
    """Notice block as plain text (spec §D layout, for tests/logs)."""
    name = display or speaker or "화자"
    lang_s = _as_text(lang).strip().lower()
    en = lang_s.startswith("en")
    try:
        b = int(before)
    except (TypeError, ValueError):
        b = 50
    try:
        a = int(after)
    except (TypeError, ValueError):
        a = 50
    lines = [notice_line(name, b, a, lang="en" if en else "ko")]
    alts = [x for x in (alternatives or []) if isinstance(x, dict)]
    if alts:
        lines.append("Other ways to say it" if en else "이렇게 말해 볼 수도")
        for i, alt in enumerate(alts):
            text = _as_text(alt.get("text", "")).strip()
            if not text:
                continue
            try:
                pct = int(round(float(alt.get("p_better", 0.0)) * 100))
            except (TypeError, ValueError):
                pct = 0
            pct = max(0, min(100, pct))
            if en:
                label = (f"({pct}% chance to read better than the original)"
                         if i == 0 else f"({pct}%)")
            else:
                label = (f"(원문보다 잘 읽힐 확률 {pct}%)"
                         if i == 0 else f"({pct}%)")
            lines.append(f'  · "{text}"  {label}')
    scene = scene if isinstance(scene, dict) else None
    if scene is not None:
        told = scene_line(scene, lang="en" if en else "ko")
        tip = counsel_line(scene, lang="en" if en else "ko")
        if told and tip:
            lines.append(f"{told} {tip}")
        elif told:
            lines.append(told)
        elif tip:
            lines.append(tip)
        src = source_line(scene)
        if src:
            lines.append(src)
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-scenes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build", help="build scenes.json from owner JSONL")
    bp.add_argument("--src", required=True)
    bp.add_argument("--out", required=True)
    cp = sub.add_parser("counsel", help="ask once per scene for 교훈/교훈_en")
    cp.add_argument("--scenes", required=True)
    cp.add_argument("--model", default=None)
    cp.add_argument("--budget", type=float, default=5.0)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build_scenes(args.src, args.out)
    elif args.cmd == "counsel":
        stats = counsel_scenes(args.scenes, model=args.model,
                               budget=args.budget)
        print(json.dumps(stats, ensure_ascii=False))
