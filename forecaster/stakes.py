"""L2 both-sides stakes: outcome axis for BOTH sides (CaSiNo negotiation)."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import calibrate as C
from . import casino as CAS
from . import outcome as O
from . import stats as S
from .board import _call_board
from .reader import DEFAULT_MODEL
from .usage import BudgetExceeded

ITEMS = ("Food", "Water", "Firewood")
VALUES = {"High": 5, "Medium": 4, "Low": 3}

_AGENTS = (("A", "mturk_agent_1"), ("B", "mturk_agent_2"))
_OFFER_TAGS = {"propose", "counter", "accept"}


def priorities(entry) -> dict:
    """Invert one raw CaSiNo dialogue's value2issue maps to {side: {item: value}}."""
    pinfo = {}
    if isinstance(entry, dict):
        pinfo = entry.get("participant_info", {})
    if not isinstance(pinfo, dict):
        pinfo = {}
    out: dict[str, dict] = {}
    for side, agent in _AGENTS:
        rec = pinfo.get(agent, {})
        v2i = rec.get("value2issue", {}) if isinstance(rec, dict) else {}
        inv: dict[str, int] = {}
        if isinstance(v2i, dict):
            for level, issue in v2i.items():
                if level in VALUES and issue in ITEMS:
                    inv[issue] = VALUES[level]
        out[side] = inv
    return out


def load_priorities(casino_path) -> dict[int, dict]:
    """Load dialogue_id -> priorities from a raw CaSiNo file."""
    with open(Path(casino_path), encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, dict] = {}
    for d in data or []:
        if not isinstance(d, dict):
            continue
        try:
            did = int(d.get("dialogue_id"))
        except (TypeError, ValueError):
            continue
        out[did] = priorities(d)
    return out


def normalize_offer(raw, speaker) -> dict | None:
    """Normalize parsed model JSON into {"A": {item: n}, "B": {item: n}}.

    Returns None when the turn is not an offer, the speaker is not A/B,
    any mentioned count is not an int in 0..3, both sides of an item do
    not sum to 3, or an item is missing on both sides.
    """
    if speaker not in ("A", "B"):
        return None
    if not isinstance(raw, dict):
        return None
    if not raw.get("is_offer"):
        return None
    sg = raw.get("speaker_gets", {})
    og = raw.get("other_gets", {})
    if sg is None:
        sg = {}
    if og is None:
        og = {}
    if not isinstance(sg, dict) or not isinstance(og, dict):
        return None
    other = "B" if speaker == "A" else "A"
    alloc: dict[str, dict] = {speaker: {}, other: {}}
    for item in ITEMS:
        s = sg.get(item)
        o = og.get(item)
        s_given = s is not None
        o_given = o is not None
        if not s_given and not o_given:
            return None
        if s_given and (isinstance(s, bool) or not isinstance(s, int)
                        or not 0 <= s <= 3):
            return None
        if o_given and (isinstance(o, bool) or not isinstance(o, int)
                        or not 0 <= o <= 3):
            return None
        if s_given and o_given:
            if s + o != 3:
                return None
        elif s_given:
            o = 3 - s
        else:
            s = 3 - o
        alloc[speaker][item] = s
        alloc[other][item] = o
    return {"A": alloc["A"], "B": alloc["B"]}


def offer_points(offer, prio) -> dict:
    """Score a normalized offer under both sides' priorities."""
    out: dict[str, int] = {}
    for side in ("A", "B"):
        total = 0
        for item in ITEMS:
            total += int(offer[side][item]) * int(prio[side][item])
        out[side] = total
    return out


def _speaker_of(turn) -> str:
    if isinstance(turn, dict):
        return turn.get("speaker", "")
    return getattr(turn, "speaker", "")


def _text_of(turn) -> str:
    if isinstance(turn, dict):
        return turn.get("text", "")
    return getattr(turn, "text", "")


def build_offer_messages(turns, i, window=4) -> list[dict]:
    """Prompt asking whether the TARGET turn puts a package split on the table."""
    system = (
        "You read one TARGET turn of a CaSiNo negotiation over camp supplies.\n"
        "There are 3 Food, 3 Water and 3 Firewood packages in total.\n"
        "Reply with JSON only (no other text): "
        '{"is_offer": true|false, '
        '"speaker_gets": {"Food": n, ...}, "other_gets": {...}} '
        "with counts for items mentioned."
    )
    start = max(0, i - window)
    lines = [f"{_speaker_of(turns[j])}: {_text_of(turns[j])}"
             for j in range(start, i)]
    lines.append(f"TARGET turn {_speaker_of(turns[i])}: {_text_of(turns[i])}")
    user = "\n".join(lines) + (
        "\n\nDoes the TARGET turn put a complete or partial split of the "
        "3 Food, 3 Water, 3 Firewood packages on the table (including "
        "accepting or restating the other side's split)? Answer JSON only: "
        '{"is_offer": true|false, "speaker_gets": {"Food": n, ...}, '
        '"other_gets": {...}} with counts for items mentioned.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_offer(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("no JSON object")
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def extract_offers(client, turns, moves, model, max_workers=8, item_id="") -> dict[int, dict]:
    """Extract normalized offers for propose/counter/accept turns only."""
    tags_by_index: dict[int, set] = {}
    for t in moves or []:
        if isinstance(t, dict):
            idx = t.get("turn_index")
            mv = t.get("moves", []) or []
        else:
            idx = getattr(t, "turn_index", None)
            mv = list(getattr(t, "moves", []) or [])
        if isinstance(idx, int) and idx not in tags_by_index:
            tags_by_index[idx] = set(mv or [])
    targets = [i for i in range(len(turns or []))
               if tags_by_index.get(i, set()) & _OFFER_TAGS]
    if not targets:
        return {}
    workers = max_workers if isinstance(max_workers, int) and max_workers >= 1 else 1

    def _one(i: int) -> dict | None:
        try:
            messages = build_offer_messages(turns, i)
            meta = {"layer": "offer", "item": item_id, "turn": i}
            raw = _call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            return None
        try:
            obj = _extract_offer(raw)
        except (ValueError, json.JSONDecodeError):
            return None
        try:
            return normalize_offer(obj, _speaker_of(turns[i]))
        except Exception:
            return None

    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(i, pool.submit(_one, i)) for i in targets]
        for i, fut in futures:
            res = fut.result()
            if res is not None:
                out[i] = res
    return out


def outcome_track(n_turns, offers, prio) -> list[dict]:
    """Per-turn points of the latest offer at or before that turn."""
    norm: dict[int, dict] = {}
    for k, v in (offers or {}).items():
        try:
            kk = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            norm[kk] = v
    track: list[dict] = []
    latest: int | None = None
    for t in range(max(0, int(n_turns))):
        if t in norm:
            latest = t
        if latest is None:
            track.append({"A": None, "B": None, "offer_turn": None})
        else:
            pts = offer_points(norm[latest], prio)
            track.append({"A": pts["A"], "B": pts["B"], "offer_turn": latest})
    return track


def evaluate(tracks, outcomes, b=2000, seed=0) -> dict:
    """E1 exact match, E2 mid-track rho, E3 last-track rho, plus coverage."""
    k = 0
    n = 0
    rows_mid: list[dict] = []
    rows_last: list[dict] = []
    n_offer = 0
    n_total = 0
    for did, track in (tracks or {}).items():
        track = track or []
        n_total += 1
        if any(isinstance(e, dict) and e.get("offer_turn") is not None
               for e in track):
            n_offer += 1
        oc = (outcomes or {}).get(did)
        if oc is None or not track:
            continue
        try:
            rep_diff = int(oc["A"]["points"]) - int(oc["B"]["points"])
        except (KeyError, TypeError, ValueError):
            continue
        last = track[-1]
        if (isinstance(last, dict) and last.get("A") is not None
                and last.get("B") is not None):
            n += 1
            try:
                if (int(last["A"]) == int(oc["A"]["points"])
                        and int(last["B"]) == int(oc["B"]["points"])):
                    k += 1
            except (KeyError, TypeError, ValueError):
                pass
            rows_last.append({"dialogue": did,
                              "last": last["A"] - last["B"],
                              "final": rep_diff})
        mid = track[len(track) // 2]
        if (isinstance(mid, dict) and mid.get("A") is not None
                and mid.get("B") is not None):
            rows_mid.append({"dialogue": did,
                             "mid": mid["A"] - mid["B"],
                             "final": rep_diff})
    e1 = {"k": k, "n": n, "share": (k / n if n else 0.0),
          "ci": S.wilson_ci(k, n)}
    e2 = O.boot_spearman(rows_mid, "mid", "final", b=b, seed=seed)
    e3 = O.boot_spearman(rows_last, "last", "final", b=b, seed=seed)
    return {"E1": e1, "E2": e2, "E3": e3,
            "coverage": (n_offer / n_total if n_total else 0.0)}


def _jsonable(obj):
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _fmt_rho(value) -> str:
    try:
        return f"{value[0]:+.3f} [{value[1]:+.3f}-{value[2]:+.3f}]"
    except (TypeError, IndexError):
        return "n/a"


def main(argv=None) -> None:
    from .reader import MuseClient
    from .trace import CallLog, LoggingClient
    from .usage import Meter

    ap = argparse.ArgumentParser(prog="uptake-stakes")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--casino", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    items = C._load_items(out)
    casino_ids = sorted(cid for cid in items if O.dialogue_id_of(cid) is not None)

    stakes_path = out / "stakes.json"
    stakes: dict = {}
    if stakes_path.exists():
        try:
            with open(stakes_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                stakes = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            stakes = {}

    ann: dict[int, list] = {}
    for d in CAS.load_casino(args.casino):
        ann[d.get("dialogue_id")] = d.get("turns", [])
    unann: dict[int, list] | None = None

    def _turns_for(did: int):
        nonlocal unann
        if did in ann:
            return ann[did]
        if unann is None:
            unann = {}
            for d in CAS.load_casino(args.casino, annotated=False):
                unann[d.get("dialogue_id")] = d.get("turns", [])
        return unann.get(did)

    prio_map = load_priorities(args.casino)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meter = Meter(budget_usd=args.budget)
    inner = MuseClient(meter=meter)
    client = LoggingClient(inner, CallLog(out / "calls_stakes.jsonl"), run_id)

    def _save() -> None:
        with open(stakes_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(stakes), f, ensure_ascii=False, indent=2)
        tracks: dict[int, list] = {}
        for cid, rec in stakes.items():
            did = O.dialogue_id_of(cid)
            if did is not None and isinstance(rec, dict):
                tracks[did] = rec.get("track", []) or []
        outcomes = O.load_outcomes(args.casino)
        res = evaluate(tracks, outcomes)
        spend = meter.by_model()
        total = meter.cost()
        e1 = res["E1"]
        try:
            e1_txt = (f"{e1['share']:.3f} "
                      f"[{float(e1['ci'][0]):.3f}-{float(e1['ci'][1]):.3f}]")
        except (KeyError, TypeError, IndexError):
            e1_txt = "n/a"
        lines = [
            "# Stakes: outcome axis for both sides",
            "",
            f"dir={args.dir} casino={args.casino} model={args.model} "
            f"budget={args.budget}",
            f"n_items={len(casino_ids)} n_scored={len(tracks)} "
            f"coverage={res['coverage']:.3f}",
            "",
            "| metric | value [CI] | n |",
            "|---|---|---|",
            f"| E1 exact match (last offer vs reported) | {e1_txt} | "
            f"{e1.get('k', 0)}/{e1.get('n', 0)} |",
            f"| E2 mid-track diff vs reported diff | {_fmt_rho(res['E2'])} | - |",
            f"| E3 last-track diff vs reported diff | {_fmt_rho(res['E3'])} | - |",
            "",
            f"spend: ${total:.4f}",
        ]
        for model in sorted(spend):
            row = spend[model]
            lines.append(f"  {model}: {row['calls']} calls ${row['usd']:.4f}")
        lines.append("")
        with open(out / "stakes.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"scored {len(tracks)}/{len(casino_ids)} "
              f"coverage={res['coverage']:.3f} E1={e1_txt} "
              f"(k={e1.get('k', 0)}/n={e1.get('n', 0)})")
        print(f"E2={_fmt_rho(res['E2'])} E3={_fmt_rho(res['E3'])} "
              f"spend=${total:.4f}")

    try:
        for cid in casino_ids:
            if cid in stakes:
                continue
            did = O.dialogue_id_of(cid)
            turns = _turns_for(did)
            prio = prio_map.get(did)
            if not turns or prio is None:
                continue
            moves = items[cid].get("moves", []) or []
            offers = extract_offers(client, turns, moves, args.model,
                                    max_workers=args.workers, item_id=cid)
            track = outcome_track(len(turns), offers, prio)
            stakes[cid] = {
                "offers": {str(kk): vv for kk, vv in offers.items()},
                "track": track,
            }
    except BudgetExceeded:
        print("budget exceeded; saving partial results")
    _save()
