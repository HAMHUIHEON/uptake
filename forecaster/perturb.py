"""Single-turn edit sensitivity test (task46, Amendment 11).

Reuses the API wrapper, judge personas, parser and cost accounting from
``forecaster.pairwise`` instead of duplicating them.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import batch
from . import board
from . import pairwise as PW
from . import scalecheck as SC
from .reader import DEFAULT_MODEL, MuseClient
from .trace import CallLog, LoggingClient
from .usage import BudgetExceeded, Meter

# Fixed rewrite prompt (auditable exact wording). Used twice per target, once
# with style RUDE and once with style POLITE.
REWRITE_PROMPT = (
    "You rewrite a single dialogue turn into the requested style.\n"
    "Keep the propositional content exactly: the same offer, ask, numbers, "
    "names and decisions.\n"
    "Change only the interpersonal wording.\n"
    "- RUDE: same content with face-threatening wording: blunt refusal, "
    "blame, no softening.\n"
    "- POLITE: same content with redressed wording: acknowledgement, "
    "hedge, thanks.\n"
    "Write one short paragraph, in the same language as the original.\n"
    "Reply with the rewritten turn only, no quotes, no explanation."
)

RUDE_STYLE = "RUDE"
POLITE_STYLE = "POLITE"
SLIGHT_STYLE = "SLIGHT"
STRONG_STYLE = "STRONG"

# Fixed SLIGHT rewrite prompt (Amendment 13, auditable exact wording):
# one small hardening touch on the original, content kept.
SLIGHT_REWRITE_PROMPT = (
    "You rewrite a single dialogue turn with one small hardening touch.\n"
    "Keep the propositional content exactly: the same offer, ask, numbers, "
    "names and decisions.\n"
    "Make exactly one small hardening edit to the interpersonal wording "
    '(e.g. drop "please" or add "no offence"), nothing more.\n'
    "Write one short paragraph, in the same language as the original.\n"
    "Reply with the rewritten turn only, no quotes, no explanation."
)
SLIGHT_PROMPT = SLIGHT_REWRITE_PROMPT
# STRONG reuses the existing RUDE prompt (same system prompt + RUDE wording).
STRONG_REWRITE_PROMPT = REWRITE_PROMPT

# Variant levels: default stays rude,polite; dose uses slight,strong.
DEFAULT_LEVELS: tuple[str, str] = ("rude", "polite")
DOSE_LEVELS: tuple[str, str] = ("slight", "strong")

# Canonical perturb pairs and their predicted winners.
PAIRS: tuple[tuple[str, str], ...] = (
    ("orig", "rude"),
    ("orig", "polite"),
    ("rude", "polite"),
)
PAIR_IDS = ("orig_rude", "orig_polite", "rude_polite")
DOSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("orig", "slight"),
    ("orig", "strong"),
    ("slight", "strong"),
)
DOSE_PAIR_IDS = ("orig_slight", "orig_strong", "slight_strong")
DOSE_CANONICAL: dict[str, tuple[str, str]] = {
    "orig_slight": ("orig", "slight"),
    "orig_strong": ("orig", "strong"),
    "slight_strong": ("slight", "strong"),
}
DOSE_PREDICTED: dict[str, str] = {
    "orig_slight": "orig",
    "orig_strong": "orig",
    "slight_strong": "slight",
}
CANONICAL: dict[str, tuple[str, str]] = {
    "orig_rude": ("orig", "rude"),
    "orig_polite": ("orig", "polite"),
    "rude_polite": ("rude", "polite"),
    "orig_slight": ("orig", "slight"),
    "orig_strong": ("orig", "strong"),
    "slight_strong": ("slight", "strong"),
}
PREDICTED: dict[str, str] = {
    "orig_rude": "orig",
    "orig_polite": "polite",
    "rude_polite": "polite",
    "orig_slight": "orig",
    "orig_strong": "orig",
    "slight_strong": "slight",
}

VERDICT_SENSITIVE = "directionally sensitive"
VERDICT_INSENSITIVE = "insensitive at single-turn edits"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_GRADED = "graded"
VERDICT_ORDINAL_FLAT = "ordinal but flat"
VERDICT_NOT_GRADED = "not graded"


def _normalize_levels(levels=None) -> tuple[str, str]:
    """Normalize --levels to a (a, b) pair of variant keys (lowercase)."""
    if levels is None:
        return DEFAULT_LEVELS
    if isinstance(levels, str):
        parts = [p.strip().lower() for p in levels.split(",")]
        parts = [p for p in parts if p]
    else:
        try:
            parts = [str(p).strip().lower() for p in list(levels)]
        except TypeError:
            return DEFAULT_LEVELS
        parts = [p for p in parts if p]
    if len(parts) < 2:
        return DEFAULT_LEVELS
    return (parts[0], parts[1])


_NON_VARIANT_KEYS = frozenset({
    "dialogue", "turn", "orig", "text", "speaker", "z", "id",
})


def _detect_variant_levels(variants) -> tuple[str, str]:
    """Detect (a, b) variant keys from stored variants (orig excluded)."""
    seen: list[str] = []
    try:
        seq = list(variants or [])
    except TypeError:
        seq = []
    for v in seq:
        if not isinstance(v, dict):
            continue
        for k in v.keys():
            if not isinstance(k, str):
                continue
            kl = k.strip().lower()
            if not kl or kl in _NON_VARIANT_KEYS or kl == "orig":
                continue
            if kl not in seen:
                seen.append(kl)
    if "rude" in seen and "polite" in seen:
        return ("rude", "polite")
    if "slight" in seen and "strong" in seen:
        return ("slight", "strong")
    if len(seen) >= 2:
        return (seen[0], seen[1])
    return DEFAULT_LEVELS


def _pair_ids_for_levels(a: str, b: str) -> tuple[str, str, str]:
    return (f"orig_{a}", f"orig_{b}", f"{a}_{b}")


def _canonical_for_levels(a: str, b: str) -> dict[str, tuple[str, str]]:
    return {
        f"orig_{a}": ("orig", a),
        f"orig_{b}": ("orig", b),
        f"{a}_{b}": (a, b),
    }


def _predicted_for_levels(a: str, b: str) -> dict[str, str]:
    if (a, b) == ("rude", "polite"):
        return {
            "orig_rude": "orig",
            "orig_polite": "polite",
            "rude_polite": "polite",
        }
    if (a, b) == ("slight", "strong"):
        return dict(DOSE_PREDICTED)
    # Generic dose-style: orig beats both edits, milder edit beats stronger.
    return {
        f"orig_{a}": "orig",
        f"orig_{b}": "orig",
        f"{a}_{b}": a,
    }


# --- select ---------------------------------------------------------------

def load_ids(ids_file, n) -> list[str]:
    try:
        nn = int(n)
    except (TypeError, ValueError):
        nn = 20
    if nn < 0:
        nn = 0
    try:
        text = Path(ids_file).read_text(encoding="utf-8")
    except OSError:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids[:nn]


def load_mu_sd(table_path) -> tuple[float, float]:
    p = Path(table_path)
    try:
        if p.is_dir():
            p = p / "table.json"
    except OSError:
        pass
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read table {table_path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"bad table {table_path}: not an object")
    mu = sd = None
    if "mu" in data and "sd" in data:
        mu, sd = data.get("mu"), data.get("sd")
    elif isinstance(data.get("norm"), dict):
        norm = data["norm"]
        mu, sd = norm.get("mu"), norm.get("sd")
    else:
        # calibration.json style: top-level genre entries with "norm"
        for v in data.values():
            if isinstance(v, dict) and isinstance(v.get("norm"), dict):
                mu = v["norm"].get("mu")
                sd = v["norm"].get("sd")
                break
    if mu is None or sd is None:
        raise SystemExit(f"bad table {table_path}: no mu/sd")
    try:
        mu_f = float(mu)
        sd_f = float(sd)
    except (TypeError, ValueError):
        raise SystemExit(f"bad table {table_path}: mu/sd not numeric")
    return (mu_f, sd_f)


def word_count(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def pick_target(effects, mu, sd) -> dict | None:
    """Median-|z| effect with >= 6 words; ties go to the earlier turn."""
    try:
        mu_f = float(mu)
    except (TypeError, ValueError):
        mu_f = 0.0
    try:
        sd_f = float(sd)
    except (TypeError, ValueError):
        sd_f = 0.0
    cands: list[dict] = []
    for pos, e in enumerate(effects or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) else pos
        speaker = e.get("speaker", "")
        speaker = speaker if isinstance(speaker, str) else str(speaker)
        text = e.get("text", "")
        text = text if isinstance(text, str) else str(text)
        if word_count(text) < 6:
            continue
        try:
            value = float(board._value_of(e))
        except (TypeError, ValueError):
            continue
        z = 0.0 if sd_f == 0 else (value - mu_f) / sd_f
        cands.append({"turn": idx, "speaker": speaker, "text": text,
                      "z": float(z)})
    if not cands:
        return None
    cands.sort(key=lambda c: (abs(c["z"]), c["turn"]))
    # Lower median: index (len-1)//2. For odd lengths this is the middle;
    # for even lengths it matches "closest to the statistical median,
    # ties go earlier".
    return cands[(len(cands) - 1) // 2]


def select_targets(records_dir, table_path, ids_file, n, out_dir) -> list[dict]:
    mu, sd = load_mu_sd(table_path)
    ids = load_ids(ids_file, n)
    records = SC.load_dir(records_dir, ids=ids)
    targets: list[dict] = []
    for did in ids:
        rec = records.get(did)
        if not isinstance(rec, dict):
            continue
        pick = pick_target(rec.get("effects", []), mu, sd)
        if pick is None:
            continue
        targets.append({"dialogue": did, "turn": pick["turn"],
                        "speaker": pick["speaker"], "text": pick["text"],
                        "z": float(pick["z"])})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "targets.json", "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    return targets


# --- write ----------------------------------------------------------------

def parse_rewrite(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s if s else None


def build_rewrite_messages(text, style, speaker="") -> list[dict]:
    label = speaker if isinstance(speaker, str) and speaker else "TARGET"
    su = str(style).strip().upper() if isinstance(style, str) else ""
    if su == SLIGHT_STYLE:
        system = SLIGHT_REWRITE_PROMPT
        style_label = SLIGHT_STYLE
    elif su == STRONG_STYLE:
        # STRONG reuses the existing RUDE prompt; keep both labels visible.
        system = REWRITE_PROMPT
        style_label = f"{RUDE_STYLE} ({STRONG_STYLE})"
    else:
        system = REWRITE_PROMPT
        style_label = style
    user = (f"Style: {style_label}\n"
            f"TARGET turn {label}: {text}\n"
            "Rewrite:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _rewrite_one(client, messages, model, meta):
    attempts = 0
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        attempts += 1
        parsed = parse_rewrite(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def write_variants(targets, out_dir, client, model=DEFAULT_MODEL,
                   meter=None, levels=None) -> list[dict]:
    """Rewrite each target once per level; failures become null.

    Default levels are ("rude", "polite"). Dose uses ("slight", "strong"):
    SLIGHT gets one small hardening touch (SLIGHT_REWRITE_PROMPT), STRONG
    reuses the existing RUDE prompt. Stored keys follow the levels.
    """
    a_level, b_level = _normalize_levels(levels)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants: list[dict] = []
    for t in targets or []:
        if not isinstance(t, dict):
            continue
        did = t.get("dialogue", "")
        turn = t.get("turn", 0)
        turn = turn if isinstance(turn, int) else 0
        speaker = t.get("speaker", "")
        text = t.get("text", t.get("orig", ""))
        text = text if isinstance(text, str) else str(text)
        got: dict[str, str | None] = {a_level: None, b_level: None}
        try:
            for lvl in (a_level, b_level):
                if meter is not None:
                    meter.check()
                style = str(lvl).upper()
                val, _ = _rewrite_one(
                    client, build_rewrite_messages(text, style, speaker),
                    model, {"layer": "perturb-write", "item": did,
                            "dialogue": did, "turn": turn,
                            "style": style})
                got[lvl] = val
        except BudgetExceeded:
            variants.append({"dialogue": did, "turn": turn, "orig": text,
                             a_level: got[a_level], b_level: got[b_level]})
            break
        variants.append({"dialogue": did, "turn": turn, "orig": text,
                         a_level: got[a_level], b_level: got[b_level]})
    with open(out / "variants.json", "w", encoding="utf-8") as f:
        json.dump(variants, f, ensure_ascii=False, indent=2)
    return variants


def write_from_file(targets_path, out_dir, client, model=DEFAULT_MODEL,
                    meter=None, levels=None) -> list[dict]:
    try:
        with open(targets_path, encoding="utf-8") as f:
            targets = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read targets {targets_path}: {exc}")
    if not isinstance(targets, list):
        raise SystemExit(f"bad targets {targets_path}: not a list")
    return write_variants(targets, out_dir, client, model=model,
                          meter=meter, levels=levels)


# --- vote -----------------------------------------------------------------

def build_perturb_messages(context_lines, speaker, text_a, text_b,
                           receiver, listener="") -> list[dict]:
    """Shared context once + A/B candidates; same JSON contract as pairwise."""
    system = (
        "You are this listener reacting in conversations:\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You will see one conversation context and two candidate versions "
        "(A and B) of the next message. "
        "You do not know what the speaker intended. "
        "React the way this listener would, then reply with JSON only "
        "(no other text) with keys better, why.\n"
        "- better must be one of: first, second, tie — after which message is the "
        "listener more favourably disposed toward the speaker? (tie = about the same)\n"
        "- why must be one sentence saying what drove your choice"
    )
    ctx = "\n".join(context_lines or [])
    if ctx:
        ctx_block = f"Context (listener: {listener}):\n{ctx}\n\n"
    else:
        ctx_block = f"Context (listener: {listener}):\n(empty)\n\n"
    user = (
        f"{ctx_block}"
        f"Candidate A ({speaker}): {text_a}\n"
        f"Candidate B ({speaker}): {text_b}\n\n"
        "After which message is the listener more favourably disposed toward the "
        "speaker? Reply with JSON only: "
        '{"better": "first" | "second" | "tie", "why": "<one sentence>"}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _judge_one_perturb(client, messages, model, meta):
    attempts = 0
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        attempts += 1
        parsed = PW.parse_pairwise(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def _listener_map(pool) -> dict:
    out: dict[tuple, str] = {}
    for e in pool or []:
        if not isinstance(e, dict):
            continue
        out[(e.get("dialogue"), e.get("turn"))] = e.get("listener", "")
    return out


def build_presentations(variants, records, levels=None) -> tuple[list[dict], dict]:
    """Six presentations per complete target (3 pairs x 2 orders).

    Variant keys follow the stored variants: (orig,a), (orig,b), (a,b)
    where (a,b) are the two level keys present (default rude,polite;
    dose slight,strong). An explicit levels=(a,b) overrides detection.
    """
    if isinstance(records, dict):
        rec_map = {k: v for k, v in records.items()
                   if isinstance(v, dict)}
    else:
        rec_map = {}
        for r in records or []:
            if isinstance(r, dict) and isinstance(r.get("id"), str):
                rec_map[r["id"]] = r
    pool, contexts = PW.collect_pool(rec_map)
    listeners = _listener_map(pool)
    if levels is None:
        lvl_a, lvl_b = _detect_variant_levels(variants)
    else:
        lvl_a, lvl_b = _normalize_levels(levels)
    pair_ids = _pair_ids_for_levels(lvl_a, lvl_b)
    canonical = _canonical_for_levels(lvl_a, lvl_b)
    presentations: list[dict] = []
    pid = 0
    for v in variants or []:
        if not isinstance(v, dict):
            continue
        did = v.get("dialogue", "")
        turn = v.get("turn", 0)
        turn = turn if isinstance(turn, int) else 0
        orig = v.get("orig", "")
        va = v.get(lvl_a)
        vb = v.get(lvl_b)
        if va is None or vb is None:
            continue
        if not isinstance(orig, str) or not isinstance(va, str) \
                or not isinstance(vb, str):
            continue
        ctx = contexts.get(did)
        if not isinstance(ctx, dict):
            continue
        turns = ctx.get("turns", []) or []
        pos = ctx.get("pos_of", {}).get(turn)
        if not isinstance(pos, int) or pos < 0 or pos >= len(turns):
            continue
        speaker = turns[pos].speaker
        listener = listeners.get((did, turn), "")
        if not listener:
            seen: list[str] = []
            for t in turns:
                if t.speaker not in seen:
                    seen.append(t.speaker)
            others = [s for s in seen
                      if s != speaker and s != "unknown"]
            listener = others[0] if others else "listener"
        context_lines = PW.snippet_lines(turns, pos, PW.WINDOW)
        texts = {"orig": orig, lvl_a: va, lvl_b: vb}
        for pair_id in pair_ids:
            a_var, b_var = canonical[pair_id]
            for first in ("a", "b"):
                first_var = a_var if first == "a" else b_var
                second_var = b_var if first == "a" else a_var
                presentations.append({
                    "id": pid,
                    "dialogue": did,
                    "turn": turn,
                    "pair": pair_id,
                    "a": a_var,
                    "b": b_var,
                    "first": first,
                    "speaker": speaker,
                    "listener": listener,
                    "context": list(context_lines),
                    "text_first": texts[first_var],
                    "text_second": texts[second_var],
                })
                pid += 1
    # Strip judging-only helpers before returning? Keep them: the writer
    # below persists only the stable schema.
    return presentations, {"n_variants": len(variants or []),
                           "levels": [lvl_a, lvl_b],
                           "pair_ids": list(pair_ids)}


def judge_presentations(client, presentations, model=DEFAULT_MODEL,
                        workers=10, meter=None):
    by_id = {p["id"]: p for p in (presentations or [])
             if isinstance(p, dict)}
    raw: dict[int, dict] = {pid: {s: None for s in PW.STANCES}
                            for pid in by_id}
    stats = {"n_calls": 0, "n_attempted_votes": 0, "budget_stop": False}
    try:
        max_workers = int(workers)
    except (TypeError, ValueError):
        max_workers = 10
    if max_workers < 1:
        max_workers = 1
    queue: deque[tuple] = deque(
        (pid, stance) for pid in sorted(by_id) for stance in PW.STANCES)

    def _one(pid, stance):
        p = by_id[pid]
        rec = PW.pairwise_panel(pid)[PW.STANCES.index(stance)]
        messages = build_perturb_messages(
            p.get("context", []), p.get("speaker", ""),
            p.get("text_first", ""), p.get("text_second", ""),
            rec, p.get("listener", ""))
        meta = {"layer": "pairwise", "pair": pid, "stance": stance,
                "member": rec.id, "item": p.get("dialogue"),
                "dialogue": p.get("dialogue"),
                "dialogues": [p.get("dialogue")],
                "turn": p.get("turn"), "perturb_pair": p.get("pair"),
                "task": "perturb"}
        return _judge_one_perturb(client, messages, model, meta)

    in_flight: dict[cf.Future, tuple] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue or in_flight:
            while queue and len(in_flight) < max_workers \
                    and not stats["budget_stop"]:
                if meter is not None:
                    try:
                        meter.check()
                    except BudgetExceeded:
                        stats["budget_stop"] = True
                        break
                pid, stance = queue.popleft()
                fut = pool.submit(_one, pid, stance)
                in_flight[fut] = (pid, stance)
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight),
                              return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pid, stance = in_flight.pop(fut)
                try:
                    parsed, n_att = fut.result()
                except BudgetExceeded:
                    stats["budget_stop"] = True
                    continue
                except Exception:
                    continue
                stats["n_attempted_votes"] += 1
                stats["n_calls"] += n_att
                raw[pid][stance] = parsed["better"] if parsed else None
    return raw, stats


def write_vote_outputs(out_dir, presentations, raw_votes, stats, settings,
                       spend, cost, run_id="", commit="unknown") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pres = [p for p in (presentations or []) if isinstance(p, dict)]
    pres.sort(key=lambda q: q.get("id", 0)
              if isinstance(q.get("id"), int) else 0)
    n_first = n_second = n_tie = n_null = 0
    lines = []
    for p in pres:
        pid = p.get("id")
        got = raw_votes.get(pid, {}) if isinstance(raw_votes, dict) else {}
        votes: dict[str, str | None] = {}
        raw_out: dict[str, str | None] = {}
        for stance in PW.STANCES:
            v = got.get(stance)
            v = v if v in PW.VOTES else None
            raw_out[stance] = v
            votes[stance] = PW.derandomize(v, p.get("first", "a"))
            if v == "first":
                n_first += 1
            elif v == "second":
                n_second += 1
            elif v == "tie":
                n_tie += 1
            else:
                n_null += 1
        lines.append({"id": pid, "dialogue": p.get("dialogue"),
                      "turn": p.get("turn"), "pair": p.get("pair"),
                      "a": p.get("a"), "b": p.get("b"),
                      "first": p.get("first", "a"),
                      "votes": votes, "raw": raw_out})
    with open(out / "votes.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    attempted = int(stats.get("n_attempted_votes", 0) or 0)
    denom = n_first + n_second
    targets = sorted({(line.get("dialogue"), line.get("turn"))
                      for line in lines})
    run = {
        "run_id": run_id,
        "settings": dict(settings or {}),
        "commit": commit,
        "n_targets": len(targets),
        "n_presentations": len(lines),
        "n_expected": len(lines) * len(PW.STANCES),
        "n_calls": int(stats.get("n_calls", 0) or 0),
        "n_attempted_votes": attempted,
        "n_votes": n_first + n_second + n_tie,
        "n_null": n_null,
        "n_first": n_first,
        "n_second": n_second,
        "n_tie": n_tie,
        "parse_fail_rate": (n_null / attempted) if attempted else 0.0,
        "position_bias_share": (n_first / denom) if denom else 0.0,
        "spend": spend if isinstance(spend, dict) else {},
        "cost": float(cost),
        "budget_stop": bool(stats.get("budget_stop", False)),
    }
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return run


def run_vote(variants, records, out_dir, client, model=DEFAULT_MODEL,
             workers=10, meter=None, seed=0, settings=None,
             spend=None, cost=0.0, run_id="", commit="unknown",
             levels=None):
    presentations, _info = build_presentations(variants, records,
                                               levels=levels)
    raw, stats = judge_presentations(client, presentations, model=model,
                                     workers=workers, meter=meter)
    base = {"seed": seed, "model": model}
    if isinstance(settings, dict):
        base.update(settings)
    return write_vote_outputs(out_dir, presentations, raw, stats, base,
                              spend if spend is not None else {},
                              cost, run_id, commit), presentations


# --- report ---------------------------------------------------------------

def _load_json_list(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    return data if isinstance(data, list) else []


def _load_jsonl(path) -> list[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def majority_for_pair(votes_ab, predicted):
    """Majority winner among a/b/tie votes; ties become "tie"."""
    counts: dict[str, int] = {"a": 0, "b": 0, "tie": 0}
    for v in votes_ab or []:
        if v in counts:
            counts[v] += 1
    total = counts["a"] + counts["b"] + counts["tie"]
    if total == 0:
        return "tie", 0.0
    # Map a/b to variant names is done by the caller; here predicted is
    # already "a" or "b".
    best = max(counts.values())
    winners = [k for k, c in counts.items() if c == best]
    if len(winners) != 1:
        margin = 0.0
    else:
        w = winners[0]
        if w == "tie":
            margin = 0.0
        else:
            n_pred = counts.get(predicted, 0)
            other = "b" if predicted == "a" else "a"
            n_opp = counts.get(other, 0)
            margin = (n_pred - n_opp) / total
        return w, float(margin)
    n_pred = counts.get(predicted, 0)
    other = "b" if predicted == "a" else "a"
    n_opp = counts.get(other, 0)
    return "tie", float((n_pred - n_opp) / total)


def _detect_levels_from_votes(votes) -> tuple[str, str] | None:
    pairs: set[str] = set()
    for line in votes or []:
        if isinstance(line, dict) and isinstance(line.get("pair"), str):
            pairs.add(line["pair"])
    if not pairs:
        return None
    if {"orig_slight", "orig_strong", "slight_strong"} & pairs:
        return ("slight", "strong")
    if {"orig_rude", "orig_polite", "rude_polite"} & pairs:
        return ("rude", "polite")
    # Generic: look for orig_X, orig_Y, X_Y.
    orig_suffixes: list[str] = []
    for p in sorted(pairs):
        if p.startswith("orig_") and len(p) > 5:
            suf = p[5:]
            if suf and suf not in orig_suffixes:
                orig_suffixes.append(suf)
    if len(orig_suffixes) >= 2:
        a, b = orig_suffixes[0], orig_suffixes[1]
        if f"{a}_{b}" in pairs or f"{b}_{a}" in pairs:
            if f"{a}_{b}" in pairs:
                return (a, b)
            return (b, a)
        return (a, b)
    return None


def compute_report_metrics(votes, variants) -> dict:
    var_map: dict[tuple, dict] = {}
    order: list[tuple] = []
    for v in variants or []:
        if not isinstance(v, dict):
            continue
        key = (v.get("dialogue"), v.get("turn"))
        if key not in var_map:
            order.append(key)
        var_map[key] = v
    # Levels follow the stored variants; fall back to vote pair ids.
    lvl_a, lvl_b = _detect_variant_levels(variants)
    has_variant_dicts = any(isinstance(v, dict) for v in (variants or []))
    if not has_variant_dicts:
        from_votes = _detect_levels_from_votes(votes)
        if from_votes is not None:
            lvl_a, lvl_b = from_votes
    else:
        # If variants look default but votes are clearly dose (or vice
        # versa), trust the votes: they are what is being scored.
        from_votes = _detect_levels_from_votes(votes)
        if from_votes is not None and from_votes != (lvl_a, lvl_b):
            # Only override when variants carry no usable level keys.
            seen_levels = set()
            for v in (variants or []):
                if isinstance(v, dict):
                    for k in (lvl_a, lvl_b):
                        if k in v:
                            seen_levels.add(k)
            if len(seen_levels) < 2:
                lvl_a, lvl_b = from_votes
    pair_ids = _pair_ids_for_levels(lvl_a, lvl_b)
    canonical = _canonical_for_levels(lvl_a, lvl_b)
    predicted = _predicted_for_levels(lvl_a, lvl_b)
    # Group derandomized a/b/tie votes per (target, pair).
    grouped: dict[tuple, dict[str, list]] = {}
    n_first = n_second = 0
    for line in votes or []:
        if not isinstance(line, dict):
            continue
        key = (line.get("dialogue"), line.get("turn"))
        pair = line.get("pair")
        if pair not in canonical:
            continue
        bucket = grouped.setdefault(key, {pid: [] for pid in pair_ids})
        votes_d = line.get("votes", {})
        raw_d = line.get("raw", {})
        if isinstance(votes_d, dict):
            vals = [votes_d.get(s) for s in PW.STANCES]
        elif isinstance(votes_d, list):
            vals = list(votes_d)
        else:
            vals = []
        bucket[pair].extend(vals)
        if isinstance(raw_d, dict):
            for s in PW.STANCES:
                r = raw_d.get(s)
                if r == "first":
                    n_first += 1
                elif r == "second":
                    n_second += 1
        elif isinstance(raw_d, list):
            for r in raw_d:
                if r == "first":
                    n_first += 1
                elif r == "second":
                    n_second += 1
    # Targets under test: variants order, restricted to complete variants
    # when variants carry nulls; fall back to vote keys when empty.
    targets = [k for k in order
               if isinstance(var_map.get(k), dict)
               and var_map[k].get(lvl_a) is not None
               and var_map[k].get(lvl_b) is not None]
    if not targets:
        targets = sorted(grouped.keys(),
                         key=lambda k: (str(k[0]), k[1]
                                       if isinstance(k[1], int) else 0))
    per_pair_correct = {pid: 0 for pid in pair_ids}
    margins: list[float] = []
    margins_by_pair: dict[str, list[float]] = {pid: [] for pid in pair_ids}
    ordering_correct = 0
    details: dict[tuple, dict] = {}
    for key in targets:
        bucket = grouped.get(key, {pid: [] for pid in pair_ids})
        outcomes: dict[str, str] = {}
        ok_all = True
        for pid in pair_ids:
            a_var, b_var = canonical[pid]
            pred_var = predicted[pid]
            pred_ab = "a" if pred_var == a_var else "b"
            winner_ab, margin = majority_for_pair(bucket.get(pid, []),
                                                  pred_ab)
            margins.append(float(margin))
            margins_by_pair[pid].append(float(margin))
            if winner_ab == "tie":
                outcomes[pid] = "tie"
                ok_all = False
            else:
                winner_var = a_var if winner_ab == "a" else b_var
                outcomes[pid] = winner_var
                if winner_var == pred_var:
                    per_pair_correct[pid] += 1
                else:
                    ok_all = False
        details[key] = outcomes
        if ok_all:
            ordering_correct += 1
    n_targets = len(targets)
    ordering_accuracy = (ordering_correct / n_targets) if n_targets else 0.0
    mean_margin = (sum(margins) / len(margins)) if margins else 0.0
    mean_margins = {pid: (sum(vals) / len(vals) if vals else 0.0)
                    for pid, vals in margins_by_pair.items()}
    denom = n_first + n_second
    position_bias = (n_first / denom) if denom else 0.0
    out: dict = {
        "n_targets": int(n_targets),
        "ordering_correct": int(ordering_correct),
        "ordering_accuracy": float(ordering_accuracy),
        "per_pair_correct": {pid: int(per_pair_correct[pid])
                             for pid in pair_ids},
        "mean_margin": float(mean_margin),
        "mean_margins": {pid: float(mean_margins[pid]) for pid in pair_ids},
        "position_bias": float(position_bias),
        "n_first": int(n_first),
        "n_second": int(n_second),
        "targets": list(targets),
        "details": {f"{k[0]}:{k[1]}": v for k, v in details.items()},
        "levels": [lvl_a, lvl_b],
        "pair_ids": list(pair_ids),
    }
    if (lvl_a, lvl_b) == ("slight", "strong"):
        # Amendment 13: (a) slight > strong on majority in >=14/20,
        # (b) mean margin(orig vs strong) > mean margin(orig vs slight).
        n_gt = int(per_pair_correct.get("slight_strong", 0) or 0)
        mm_slight = float(mean_margins.get("orig_slight", 0.0) or 0.0)
        mm_strong = float(mean_margins.get("orig_strong", 0.0) or 0.0)
        need = 0.70 * n_targets
        cond_a = n_gt >= need
        cond_b = mm_strong > mm_slight
        if cond_a and cond_b:
            dose_verdict = VERDICT_GRADED
        elif cond_a:
            dose_verdict = VERDICT_ORDINAL_FLAT
        else:
            dose_verdict = VERDICT_NOT_GRADED
        out.update({
            "verdict": dose_verdict,
            "dose_verdict": dose_verdict,
            "slight_gt_strong": int(n_gt),
            "slight_gt_strong_correct": int(n_gt),
            "n_slight_gt_strong": int(n_gt),
            "mean_margin_orig_slight": float(mm_slight),
            "mean_margin_orig_strong": float(mm_strong),
            "mean_margin_slight_strong": float(
                mean_margins.get("slight_strong", 0.0) or 0.0),
            "dose_cond_a": bool(cond_a),
            "dose_cond_b": bool(cond_b),
        })
        return out
    # Pre-stated rule: ordering >= 0.80 and each pair >= 16/20. For fixtures
    # with n != 20 the pair bar scales to the same 0.80 share.
    need_pair = 0.80 * n_targets
    if ordering_accuracy >= 0.80 and all(
            per_pair_correct[pid] >= need_pair for pid in pair_ids):
        verdict = VERDICT_SENSITIVE
    elif ordering_accuracy < 0.60:
        verdict = VERDICT_INSENSITIVE
    else:
        verdict = VERDICT_INCONCLUSIVE
    out["verdict"] = verdict
    return out


def _cell(text, limit=120) -> str:
    s = text if isinstance(text, str) else str(text)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit]
    return s.replace("|", "\\|")


def write_report_markdown(out_path, metrics, variants) -> dict:
    var_map: dict[tuple, dict] = {}
    for v in variants or []:
        if isinstance(v, dict):
            var_map[(v.get("dialogue"), v.get("turn"))] = v
    m = metrics or {}
    n = int(m.get("n_targets", 0) or 0)
    acc = float(m.get("ordering_accuracy", 0.0) or 0.0)
    per = m.get("per_pair_correct", {}) or {}
    margin = float(m.get("mean_margin", 0.0) or 0.0)
    pbias = float(m.get("position_bias", 0.0) or 0.0)
    verdict = m.get("verdict", "")
    details = m.get("details", {}) or {}
    # Detect levels: prefer metrics, fall back to variants.
    lvl_a = lvl_b = None
    lv = m.get("levels")
    if isinstance(lv, (list, tuple)) and len(lv) == 2 \
            and all(isinstance(x, str) for x in lv):
        lvl_a, lvl_b = str(lv[0]).lower(), str(lv[1]).lower()
    else:
        per_keys = set(per.keys()) if isinstance(per, dict) else set()
        if per_keys & set(DOSE_PAIR_IDS):
            lvl_a, lvl_b = ("slight", "strong")
        elif per_keys & set(PAIR_IDS):
            lvl_a, lvl_b = ("rude", "polite")
        else:
            lvl_a, lvl_b = _detect_variant_levels(variants)
    if (lvl_a, lvl_b) not in (("rude", "polite"), ("slight", "strong")):
        # Generic levels: still render per-pair lines + table.
        pair_ids = _pair_ids_for_levels(lvl_a, lvl_b)
    elif (lvl_a, lvl_b) == ("slight", "strong"):
        pair_ids = list(DOSE_PAIR_IDS)
    else:
        pair_ids = list(PAIR_IDS)
    lines: list[str] = []
    if (lvl_a, lvl_b) == ("slight", "strong"):
        n_gt = int(m.get("slight_gt_strong",
                         per.get("slight_strong", 0) or 0) or 0)
        mm_slight = float(m.get("mean_margin_orig_slight", 0.0) or 0.0)
        # Backfill from mean_margins when the flat keys are absent.
        mm_map = m.get("mean_margins", {}) or {}
        if "mean_margin_orig_slight" not in m and isinstance(mm_map, dict):
            try:
                mm_slight = float(mm_map.get("orig_slight", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        mm_strong = float(m.get("mean_margin_orig_strong", 0.0) or 0.0)
        if "mean_margin_orig_strong" not in m and isinstance(mm_map, dict):
            try:
                mm_strong = float(mm_map.get("orig_strong", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        lines.append("# Perturbation dose-response report "
                     "(single-turn edits, Amendment 13)")
        lines.append("")
        lines.append(f"n_targets={n} ordering_accuracy={acc:.3f} "
                     f"({int(m.get('ordering_correct', 0) or 0)}/{n})")
        lines.append(f"slight > strong: {n_gt}/{n}")
        for pid in pair_ids:
            lines.append(f"{pid}: {int(per.get(pid, 0) or 0)}/{n} predicted")
        lines.append(f"mean_margin_orig_slight={mm_slight:+.3f} "
                     f"mean_margin_orig_strong={mm_strong:+.3f}")
        lines.append(f"mean_margin={margin:+.3f} "
                     f"position_bias={pbias:.3f} "
                     f"(share first among non-tie)")
        lines.append(f"verdict: {verdict}")
        lines.append("")
        lines.append("| dialogue | turn | orig | slight | strong | "
                     "orig vs slight | orig vs strong | slight vs strong |")
        lines.append("|---|---|---|---|---|---|---|---|")
    elif (lvl_a, lvl_b) == ("rude", "polite"):
        lines.append("# Perturbation report (single-turn edits)")
        lines.append("")
        lines.append(f"n_targets={n} ordering_accuracy={acc:.3f} "
                     f"({int(m.get('ordering_correct', 0) or 0)}/{n})")
        for pid in PAIR_IDS:
            lines.append(f"{pid}: {int(per.get(pid, 0) or 0)}/{n} predicted")
        lines.append(f"mean_margin={margin:+.3f} position_bias={pbias:.3f} "
                     f"(share first among non-tie)")
        lines.append(f"verdict: {verdict}")
        lines.append("")
        lines.append("| dialogue | turn | orig | rude | polite | "
                     "orig vs rude | orig vs polite | rude vs polite |")
        lines.append("|---|---|---|---|---|---|---|---|")
    else:
        lines.append("# Perturbation report (single-turn edits)")
        lines.append("")
        lines.append(f"n_targets={n} ordering_accuracy={acc:.3f} "
                     f"({int(m.get('ordering_correct', 0) or 0)}/{n})")
        lines.append(f"{lvl_a} > {lvl_b}: "
                     f"{int(per.get(pair_ids[2], 0) or 0)}/{n}")
        for pid in pair_ids:
            lines.append(f"{pid}: {int(per.get(pid, 0) or 0)}/{n} predicted")
        lines.append(f"mean_margin={margin:+.3f} position_bias={pbias:.3f} "
                     f"(share first among non-tie)")
        lines.append(f"verdict: {verdict}")
        lines.append("")
        lines.append(f"| dialogue | turn | orig | {lvl_a} | {lvl_b} | "
                     f"orig vs {lvl_a} | orig vs {lvl_b} | "
                     f"{lvl_a} vs {lvl_b} |")
        lines.append("|---|---|---|---|---|---|---|---|")
    targets = m.get("targets", []) or []
    # JSON round-trips turn tuples to lists; accept both.
    norm_targets: list[tuple] = []
    for t in targets:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            norm_targets.append((t[0], t[1]))
    if not norm_targets:
        norm_targets = sorted(var_map.keys(),
                              key=lambda k: (str(k[0]), k[1]
                                            if isinstance(k[1], int) else 0))
    for key in norm_targets:
        v = var_map.get(key, {})
        dkey = f"{key[0]}:{key[1]}"
        det = details.get(dkey, {}) if isinstance(details, dict) else {}
        if (lvl_a, lvl_b) == ("slight", "strong"):
            lines.append(
                f"| {_cell(key[0])} | {key[1]} "
                f"| {_cell(v.get('orig', ''))} "
                f"| {_cell(v.get('slight', ''))} "
                f"| {_cell(v.get('strong', ''))} "
                f"| {_cell(det.get('orig_slight', ''), 20)} "
                f"| {_cell(det.get('orig_strong', ''), 20)} "
                f"| {_cell(det.get('slight_strong', ''), 20)} |")
        elif (lvl_a, lvl_b) == ("rude", "polite"):
            lines.append(
                f"| {_cell(key[0])} | {key[1]} "
                f"| {_cell(v.get('orig', ''))} "
                f"| {_cell(v.get('rude', ''))} "
                f"| {_cell(v.get('polite', ''))} "
                f"| {_cell(det.get('orig_rude', ''), 20)} "
                f"| {_cell(det.get('orig_polite', ''), 20)} "
                f"| {_cell(det.get('rude_polite', ''), 20)} |")
        else:
            lines.append(
                f"| {_cell(key[0])} | {key[1]} "
                f"| {_cell(v.get('orig', ''))} "
                f"| {_cell(v.get(lvl_a, ''))} "
                f"| {_cell(v.get(lvl_b, ''))} "
                f"| {_cell(det.get(pair_ids[0], ''), 20)} "
                f"| {_cell(det.get(pair_ids[1], ''), 20)} "
                f"| {_cell(det.get(pair_ids[2], ''), 20)} |")
    lines.append("")
    out_p = Path(out_path)
    if str(out_p.parent) and str(out_p.parent) != ".":
        out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return m


def generate_report(votes_path, variants_path, out_md_path) -> dict:
    votes = _load_jsonl(votes_path)
    variants = _load_json_list(variants_path)
    metrics = compute_report_metrics(votes, variants)
    write_report_markdown(out_md_path, metrics, variants)
    return metrics


# --- CLI ------------------------------------------------------------------

def _normalize_effort(effort):
    try:
        return PW._normalize_effort(effort)
    except AttributeError:
        if isinstance(effort, str) and effort.lower() in ("none", "null", ""):
            return None
        return effort


def main(argv=None) -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(prog="uptake-perturb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("select")
    sp.add_argument("--records", required=True)
    sp.add_argument("--table", required=True)
    sp.add_argument("--ids-file", required=True)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--out", required=True)

    wp = sub.add_parser("write")
    wp.add_argument("--targets", required=True)
    wp.add_argument("--out", required=True)
    wp.add_argument("--model", default=DEFAULT_MODEL)
    wp.add_argument("--effort", default="low")
    wp.add_argument("--budget", type=float, default=0.5)
    wp.add_argument("--workers", type=int, default=10)
    wp.add_argument("--levels", default="rude,polite")

    vp = sub.add_parser("vote")
    vp.add_argument("--variants", required=True)
    vp.add_argument("--records", required=True)
    vp.add_argument("--out", required=True)
    vp.add_argument("--model", default=DEFAULT_MODEL)
    vp.add_argument("--effort", default="low")
    vp.add_argument("--budget", type=float, default=0.5)
    vp.add_argument("--seed", type=int, default=0)
    vp.add_argument("--workers", type=int, default=10)

    rp = sub.add_parser("report")
    rp.add_argument("--votes", required=True)
    rp.add_argument("--variants", required=True)
    rp.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "select":
        targets = select_targets(args.records, args.table, args.ids_file,
                                 args.n, args.out)
        print(f"done targets {len(targets)} out={args.out}/targets.json")
        return
    if args.cmd == "write":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        meter = Meter(budget_usd=args.budget)
        inner = MuseClient(meter=meter,
                           default_effort=_normalize_effort(args.effort))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = LoggingClient(inner, CallLog(out / "calls.jsonl"), run_id)
        try:
            with open(args.targets, encoding="utf-8") as f:
                targets = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read targets {args.targets}: {exc}")
        lvl_a, lvl_b = _normalize_levels(getattr(args, "levels", None))
        variants = write_variants(targets, out, client, model=args.model,
                                  meter=meter, levels=(lvl_a, lvl_b))
        n_null = sum(1 for v in variants
                     if v.get(lvl_a) is None or v.get(lvl_b) is None)
        print(f"done variants {len(variants)} null_targets={n_null} "
              f"levels={lvl_a},{lvl_b} spend: ${meter.cost():.4f}")
        return
    if args.cmd == "vote":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.variants, encoding="utf-8") as f:
                variants = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read variants {args.variants}: {exc}")
        if not isinstance(variants, list):
            raise SystemExit(f"bad variants {args.variants}: not a list")
        want = sorted({v.get("dialogue") for v in variants
                       if isinstance(v, dict)
                       and isinstance(v.get("dialogue"), str)})
        records = SC.load_dir(args.records, ids=want) if want else {}
        if not records:
            records = SC.load_dir(args.records)
        meter = Meter(budget_usd=args.budget)
        inner = MuseClient(meter=meter,
                           default_effort=_normalize_effort(args.effort))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = LoggingClient(inner, CallLog(out / "calls.jsonl"), run_id)
        settings = {"variants": args.variants, "records": args.records,
                    "out": args.out, "seed": args.seed,
                    "budget": args.budget, "workers": args.workers,
                    "effort": args.effort, "model": args.model}
        run, _pres = run_vote(variants, records, out, client,
                              model=args.model, workers=args.workers,
                              meter=meter, seed=args.seed, settings=settings,
                              spend=meter.by_model(), cost=meter.cost(),
                              run_id=run_id, commit=batch._git_commit())
        # Refresh spend after the run (meter mutated during judging).
        run["spend"] = meter.by_model()
        run["cost"] = float(meter.cost())
        with open(out / "run.json", "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        print(f"done presentations {run['n_presentations']} "
              f"votes {run['n_votes']}/{run['n_expected']} "
              f"parse_fail={run['parse_fail_rate']:.3f} "
              f"pos_bias={run['position_bias_share']:.3f} "
              f"budget_stop={run['budget_stop']} "
              f"spend: ${run['cost']:.4f}")
        return
    metrics = generate_report(args.votes, args.variants, args.out)
    if "slight_gt_strong" in metrics:
        print(f"done slight_gt_strong={metrics['slight_gt_strong']}/"
              f"{metrics['n_targets']} "
              f"mean_margin_orig_slight="
              f"{metrics['mean_margin_orig_slight']:+.3f} "
              f"mean_margin_orig_strong="
              f"{metrics['mean_margin_orig_strong']:+.3f} "
              f"verdict={metrics['verdict']} out={args.out}")
        return
    print(f"done ordering={metrics['ordering_accuracy']:.3f} "
          f"({metrics['ordering_correct']}/{metrics['n_targets']}) "
          f"verdict={metrics['verdict']} out={args.out}")
