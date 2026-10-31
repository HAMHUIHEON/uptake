"""Famous-misreading validation (task51, Amendment 12).

Reads the owner's scene corpus (rows with ``입력.상황``), asks the pairwise
reader stances how the original recipient took each utterance (real and
masked versions), judges the predicted readings against ``해석``/``근거``,
and reports M1/M2/M3 with the Amendment 12 verdict.

Reuses the stance definitions and API wrapper from ``forecaster.pairwise``
(``PW.STANCES``, ``PW.pairwise_panel``, ``board._call_board``) instead of
duplicating them. Direct imports are stdlib plus sibling modules only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import board
from . import pairwise as PW
from .stats import wilson_ci
from .trace import CallLog, LoggingClient
from .usage import BudgetExceeded, Meter

# Same default model as the pairwise reader (local copy so this module never
# imports the network client at module scope; main() imports it lazily).
DEFAULT_MODEL = "muse-spark-1.3-contributor"

RECEPTIONS = ("잘", "보통", "나쁘게")
JUDGMENTS = ("match", "partial", "miss")
VERSIONS = ("real", "masked")

VERDICT_READS = "reads famous scenes"
VERDICT_MEMORY = "partly from memory"
VERDICT_UNSUPPORTED = "not supported"
VERDICT_INCONCLUSIVE = "inconclusive"

JUDGE_PROMPT_VERSION = "v1"

# Auditable judge prompt (module constant). The judge sees only the actual
# reading (해석), its evidence (근거) and the reader's prediction — never the
# reader's prompt and never any other outcome text.
JUDGE_PROMPT = (
    "You grade a predicted reading of one utterance against the recipient's "
    "actual reading.\n"
    "You see: (1) the actual reading (해석), (2) its textual evidence (근거), "
    "(3) the predicted reception and reading.\n"
    "Reply with JSON only (no other text) with keys verdict, why.\n"
    "- verdict must be one of: match, partial, miss.\n"
    "  - match: the prediction says the same as the actual reading.\n"
    "  - partial: the prediction overlaps the actual reading but misses or "
    "adds a key point.\n"
    "  - miss: the prediction contradicts the actual reading or is about "
    "something else.\n"
    "- why must be one sentence saying what drove your grade"
)

READER_CONTRACT = (
    '{"reception": "잘" | "보통" | "나쁘게", "reading": "<one sentence>"}'
)


def _git_commit() -> str:
    try:
        import subprocess

        proc = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        commit = (proc.stdout or "").strip()
        return commit if proc.returncode == 0 and commit else "unknown"
    except Exception:  # noqa: BLE001 - git may be unavailable
        return "unknown"


def _normalize_effort(effort):
    try:
        return PW._normalize_effort(effort)
    except AttributeError:
        if isinstance(effort, str) and effort.lower() in ("none", "null", ""):
            return None
        return effort


# --- prepare ---------------------------------------------------------------

def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "false", "f", "0", "no", "n", "거짓", "무", "없음", "해당없음"):
            return False
        if s in ("true", "t", "1", "yes", "y", "참", "유", "발생", "있음"):
            return True
        return bool(s)
    return bool(value)


def _pick(top: dict, nested: dict, key: str) -> str:
    """Top-level value first, ``입력`` value as fallback (both coerced)."""
    if isinstance(top, dict) and _as_text(top.get(key)).strip():
        return _as_text(top.get(key))
    if isinstance(nested, dict):
        return _as_text(nested.get(key, ""))
    return ""


def _pick_input_first(top: dict, nested: dict, key: str) -> str:
    if isinstance(nested, dict) and _as_text(nested.get(key)).strip():
        return _as_text(nested.get(key))
    if isinstance(top, dict):
        return _as_text(top.get(key, ""))
    return ""


def extract_scene(row: dict, fallback_id: str = "") -> dict | None:
    """Build one scene record, or None when ``입력.상황`` is missing/blank."""
    if not isinstance(row, dict):
        return None
    nested = row.get("입력")
    nested = nested if isinstance(nested, dict) else {}
    situation = _as_text(nested.get("상황", ""))
    if not situation.strip():
        return None
    top_ctx = row.get("맥락")
    top_ctx = top_ctx if isinstance(top_ctx, dict) else {}
    in_ctx = nested.get("맥락")
    in_ctx = in_ctx if isinstance(in_ctx, dict) else {}
    ctx = dict(top_ctx)
    ctx.update(in_ctx)
    sid = _as_text(row.get("id", "")).strip() or fallback_id
    return {
        "id": sid,
        "계열": _pick(row, nested, "계열"),
        "작품": _pick(row, nested, "작품"),
        "상황": situation,
        "상황_가림": _pick_input_first(row, nested, "상황_가림"),
        "발화": _pick_input_first(row, nested, "발화"),
        "발화_가림": _pick_input_first(row, nested, "발화_가림"),
        "말한사람": _pick(row, nested, "말한사람"),
        "받은사람": _pick(row, nested, "받은사람"),
        "맥락": {
            "관계": _as_text(ctx.get("관계", "")),
            "권력거리": _as_text(ctx.get("권력거리", "")),
            "채널": _as_text(ctx.get("채널", "")),
        },
        "해석": _pick(row, nested, "해석"),
        "근거": _pick(row, nested, "근거"),
        "오해발생": _as_bool(row.get("오해발생", nested.get("오해발생", False))),
    }


def prepare_corpus(src, out_dir) -> dict:
    """Keep rows with ``입력.상황``; write ``scenes.json`` sorted by id."""
    try:
        text = Path(src).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read src {src}: {exc}")
    scenes: list[dict] = []
    n_total = n_bad = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        n_total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        scene = extract_scene(row, fallback_id=f"row-{lineno}")
        if scene is not None:
            scenes.append(scene)
    scenes.sort(key=lambda s: str(s.get("id", "")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    per_series: dict[str, int] = {}
    for s in scenes:
        key = _as_text(s.get("계열", "")) or "(none)"
        per_series[key] = per_series.get(key, 0) + 1
    return {
        "n_total": int(n_total),
        "n_kept": int(len(scenes)),
        "n_dropped": int(n_total - len(scenes)),
        "n_bad_json": int(n_bad),
        "per_계열": per_series,
    }


# --- read ------------------------------------------------------------------

def scene_texts(scene: dict, version: str) -> tuple[str, str]:
    """(상황, 발화) for a version; masked falls back to real when blank."""
    if not isinstance(scene, dict):
        return ("", "")
    if version == "masked":
        masked_s = _as_text(scene.get("상황_가림", ""))
        masked_u = _as_text(scene.get("발화_가림", ""))
        real_s = _as_text(scene.get("상황", ""))
        real_u = _as_text(scene.get("발화", ""))
        return (masked_s if masked_s.strip() else real_s,
                masked_u if masked_u.strip() else real_u)
    return (_as_text(scene.get("상황", "")), _as_text(scene.get("발화", "")))


def build_scene_messages(scene, receiver, version="real") -> list[dict]:
    situation, utter = scene_texts(scene if isinstance(scene, dict) else {},
                                   version)
    ctx = scene.get("맥락", {}) if isinstance(scene, dict) else {}
    if not isinstance(ctx, dict):
        ctx = {}
    speaker = _as_text(scene.get("말한사람", "")) if isinstance(scene, dict) else ""
    listener = _as_text(scene.get("받은사람", "")) if isinstance(scene, dict) else ""
    system = (
        "You are this recipient reacting to one utterance in a scene:\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You see the situation before the outcome and the utterance. "
        "You do not know what happens next or what the speaker intended. "
        "React the way this recipient would, then reply with JSON only "
        "(no other text) with keys reception, reading.\n"
        "- reception must be one of: 잘, 보통, 나쁘게 — how the recipient "
        "takes the utterance (well / neutrally / badly)\n"
        "- reading must be one sentence: what the recipient takes the "
        "utterance to mean"
    )
    user = (
        f"상황: {situation}\n"
        f"발화 ({speaker} → {listener}): {utter}\n"
        f"관계: {_as_text(ctx.get('관계', ''))} / "
        f"권력거리: {_as_text(ctx.get('권력거리', ''))} / "
        f"채널: {_as_text(ctx.get('채널', ''))}\n\n"
        "Reply with JSON only: " + READER_CONTRACT + "."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_scene_reading(raw) -> dict | None:
    if not isinstance(raw, str):
        return None
    try:
        start = raw.find("{")
        if start < 0:
            return None
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        if not isinstance(obj, dict):
            return None
    except (ValueError, json.JSONDecodeError):
        return None
    reception = obj.get("reception", "")
    reception = "" if reception is None else str(reception).strip()
    if reception not in RECEPTIONS:
        return None
    reading = obj.get("reading", "")
    reading = "" if reading is None else str(reading)
    return {"reception": reception, "reading": reading}


def _read_one(client, scene, receiver, version, model):
    messages = build_scene_messages(scene, receiver, version)
    meta = {"layer": "scenes-read",
            "scene": scene.get("id", "") if isinstance(scene, dict) else "",
            "stance": receiver.stance, "member": receiver.id,
            "version": version}
    attempts = 0
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        attempts += 1
        parsed = parse_scene_reading(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def read_scenes(scenes, version, out_dir, client, model=DEFAULT_MODEL,
                meter=None, workers=10, settings=None, run_id="",
                commit="unknown") -> tuple[dict, list[dict]]:
    """One call per (scene, stance); writes ``pred_<version>.jsonl``."""
    if version not in VERSIONS:
        raise ValueError(f"bad version: {version!r}")
    scenes = [s for s in (scenes or []) if isinstance(s, dict)]
    try:
        max_workers = int(workers)
    except (TypeError, ValueError):
        max_workers = 10
    if max_workers < 1:
        max_workers = 1
    results: dict[tuple, dict | None] = {}
    stats = {"n_calls": 0, "n_attempted": 0, "budget_stop": False}

    def _one(idx, stance):
        scene = scenes[idx]
        rec = PW.pairwise_panel(idx)[PW.STANCES.index(stance)]
        return _read_one(client, scene, rec, version, model)

    queue: deque[tuple] = deque(
        (idx, stance) for idx in range(len(scenes)) for stance in PW.STANCES)
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
                idx, stance = queue.popleft()
                fut = pool.submit(_one, idx, stance)
                in_flight[fut] = (idx, stance)
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight),
                              return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx, stance = in_flight.pop(fut)
                try:
                    parsed, n_att = fut.result()
                except BudgetExceeded:
                    stats["budget_stop"] = True
                    continue
                except Exception:  # noqa: BLE001 - keep going
                    continue
                stats["n_attempted"] += 1
                stats["n_calls"] += n_att
                results[(idx, stance)] = parsed
    preds: list[dict] = []
    n_null = 0
    for idx, scene in enumerate(scenes):
        for stance in PW.STANCES:
            parsed = results.get((idx, stance))
            if parsed is None:
                n_null += 1
                preds.append({"scene_id": scene.get("id", ""),
                              "stance": stance, "version": version,
                              "reception": None, "reading": None})
            else:
                preds.append({"scene_id": scene.get("id", ""),
                              "stance": stance, "version": version,
                              "reception": parsed["reception"],
                              "reading": parsed["reading"]})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"pred_{version}.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    attempted = int(stats.get("n_attempted", 0) or 0)
    run = {
        "run_id": run_id,
        "settings": dict(settings or {}),
        "commit": commit,
        "version": version,
        "n_scenes": len(scenes),
        "n_expected": len(scenes) * len(PW.STANCES),
        "n_calls": int(stats.get("n_calls", 0) or 0),
        "n_attempted": attempted,
        "n_preds": len(preds) - n_null,
        "n_null": n_null,
        "parse_fail_rate": (n_null / attempted) if attempted else 0.0,
        "budget_stop": bool(stats.get("budget_stop", False)),
    }
    with open(out / f"run_{version}.json", "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return run, preds


# --- judge -----------------------------------------------------------------

def build_judge_messages(pred, truth) -> list[dict]:
    """Judge sees 해석 + 근거 + the prediction only.

    ``pred`` holds the reader's predicted reception/reading; ``truth``
    holds 해석/근거. Scene situation, utterance, work and the reader's
    prompt are never included.
    """
    pred = pred if isinstance(pred, dict) else {}
    truth = truth if isinstance(truth, dict) else {}
    user = (
        f"실제 읽기 (해석): {_as_text(truth.get('해석', ''))}\n"
        f"근거: {_as_text(truth.get('근거', ''))}\n"
        f"예측된 받아들임: {_as_text(pred.get('reception', ''))}\n"
        f"예측된 읽기: {_as_text(pred.get('reading', ''))}\n\n"
        "Reply with JSON only: "
        '{"verdict": "match" | "partial" | "miss", "why": "<one sentence>"}.'
    )
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_judge(raw) -> dict | None:
    if not isinstance(raw, str):
        return None
    try:
        start = raw.find("{")
        if start < 0:
            return None
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        if not isinstance(obj, dict):
            return None
    except (ValueError, json.JSONDecodeError):
        return None
    verdict = obj.get("verdict", obj.get("judgment", obj.get("judgement", "")))
    verdict = "" if verdict is None else str(verdict).strip().lower()
    if verdict not in JUDGMENTS:
        return None
    why = obj.get("why", "")
    why = "" if why is None else str(why)
    return {"verdict": verdict, "why": why}


def _judge_one(client, pred, truth, model):
    messages = build_judge_messages(pred, truth)
    meta = {"layer": "scenes-judge",
            "scene": pred.get("scene_id", "") if isinstance(pred, dict) else "",
            "stance": pred.get("stance", "") if isinstance(pred, dict) else "",
            "version": pred.get("version", "") if isinstance(pred, dict) else ""}
    attempts = 0
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        attempts += 1
        parsed = parse_judge(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def judge_preds(preds, truths, out_path, client, model=DEFAULT_MODEL,
                meter=None, workers=10) -> tuple[list[dict], dict]:
    """One judge call per prediction; writes ``judge_<version>.jsonl``.

    ``out_path`` may be a file (written exactly) or a directory (the file
    name is derived from the preds version). Returns (judgments, stats).
    """
    preds = [p for p in (preds or []) if isinstance(p, dict)]
    truths = truths if isinstance(truths, dict) else {}
    try:
        max_workers = int(workers)
    except (TypeError, ValueError):
        max_workers = 10
    if max_workers < 1:
        max_workers = 1
    results: dict[int, dict | None] = {}
    stats = {"n_calls": 0, "n_attempted": 0, "budget_stop": False}

    def _one(i):
        pred = preds[i]
        truth = truths.get(pred.get("scene_id", {}), {})
        if not isinstance(truth, dict):
            truth = {}
        return _judge_one(client, pred, truth, model)

    queue: deque[int] = deque(range(len(preds)))
    in_flight: dict[cf.Future, int] = {}
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
                i = queue.popleft()
                fut = pool.submit(_one, i)
                in_flight[fut] = i
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight),
                              return_when=cf.FIRST_COMPLETED)
            for fut in done:
                i = in_flight.pop(fut)
                try:
                    parsed, n_att = fut.result()
                except BudgetExceeded:
                    stats["budget_stop"] = True
                    continue
                except Exception:  # noqa: BLE001 - keep going
                    continue
                stats["n_attempted"] += 1
                stats["n_calls"] += n_att
                results[i] = parsed
    judgments: list[dict] = []
    n_null = 0
    for i, pred in enumerate(preds):
        parsed = results.get(i)
        if parsed is None:
            n_null += 1
            judgments.append({"scene_id": pred.get("scene_id", ""),
                              "stance": pred.get("stance", ""),
                              "version": pred.get("version", ""),
                              "verdict": None, "why": ""})
        else:
            judgments.append({"scene_id": pred.get("scene_id", ""),
                              "stance": pred.get("stance", ""),
                              "version": pred.get("version", ""),
                              "verdict": parsed["verdict"],
                              "why": parsed["why"]})
    out_p = Path(out_path)
    versions = {str(p.get("version", "")) for p in preds} & set(VERSIONS)
    version = sorted(versions)[0] if len(versions) == 1 else "mixed"
    target = out_p / f"judge_{version}.jsonl" if out_p.is_dir() else out_p
    if str(target.parent) and str(target.parent) != ".":
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for j in judgments:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    attempted = int(stats.get("n_attempted", 0) or 0)
    stats.update({
        "n_preds": len(preds),
        "n_judged": len(preds) - n_null,
        "n_null": n_null,
        "parse_fail_rate": (n_null / attempted) if attempted else 0.0,
        "version": version,
        "out": str(target),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    })
    return judgments, stats


# --- report ----------------------------------------------------------------

def _load_scenes(path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read scenes {path}: {exc}")
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        data = data["scenes"]
    if not isinstance(data, list):
        raise SystemExit(f"bad scenes {path}: not a list")
    out = [s for s in data if isinstance(s, dict)]
    out.sort(key=lambda s: str(s.get("id", "")))
    return out


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


def _success_of(verdict) -> bool:
    return verdict in ("match", "partial")


def scene_success(verdicts) -> bool:
    """Majority of stances judged match or partial (>= 2 of 3)."""
    return sum(1 for v in (verdicts or []) if _success_of(v)) >= 2


def majority_verdict(verdicts) -> str:
    """Most common verdict (missing counts as miss; 1-1-1 ties → partial)."""
    vals = [v if v in JUDGMENTS else "miss" for v in (verdicts or [])]
    if not vals:
        return "miss"
    counts = {label: vals.count(label) for label in JUDGMENTS}
    best = max(counts.values())
    winners = [label for label in JUDGMENTS if counts[label] == best]
    if len(winners) == 1:
        return winners[0]
    if "partial" in winners:
        return "partial"
    return winners[0]


def majority_reception(receptions) -> str:
    """Most common reception; ties fall back to 보통; "" when all missing."""
    vals = [r for r in (receptions or []) if r in RECEPTIONS]
    if not vals:
        return ""
    counts = {label: vals.count(label) for label in RECEPTIONS}
    best = max(counts.values())
    winners = [label for label in RECEPTIONS if counts[label] == best]
    if len(winners) == 1:
        return winners[0]
    if "보통" in winners:
        return "보통"
    return winners[0]


def newcombe_diff_ci(k1, n1, k2, n2, z=1.959964) -> tuple[float, float]:
    """Newcombe 95% interval for p1 − p2 from two Wilson score intervals."""
    p1 = (k1 / n1) if n1 else 0.0
    p2 = (k2 / n2) if n2 else 0.0
    d = p1 - p2
    l1, u1 = wilson_ci(k1, n1, z=z)
    l2, u2 = wilson_ci(k2, n2, z=z)
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def amendment12_verdict(m1: float, m3: float) -> str:
    if m1 >= 0.60 and m3 <= 0.15:
        return VERDICT_READS
    if m1 >= 0.60:
        return VERDICT_MEMORY
    if m1 < 0.45:
        return VERDICT_UNSUPPORTED
    return VERDICT_INCONCLUSIVE


def _index_preds(preds) -> dict:
    """(scene_id, version) -> {stance: pred}."""
    idx: dict = {}
    for p in preds or []:
        if not isinstance(p, dict):
            continue
        key = (p.get("scene_id", ""), p.get("version", ""))
        bucket = idx.setdefault(key, {})
        stance = p.get("stance", "")
        if stance in PW.STANCES and stance not in bucket:
            bucket[stance] = p
    return idx


def _index_judgments(judgments) -> dict:
    idx: dict = {}
    for j in judgments or []:
        if not isinstance(j, dict):
            continue
        key = (j.get("scene_id", ""), j.get("version", ""))
        bucket = idx.setdefault(key, {})
        stance = j.get("stance", "")
        if stance in PW.STANCES and stance not in bucket:
            bucket[stance] = j
    return idx


def _neutral_reading(pred_by_stance) -> str:
    for stance in PW.STANCES:
        p = (pred_by_stance or {}).get(stance, {})
        if isinstance(p, dict):
            reading = p.get("reading", "")
            if isinstance(reading, str) and reading.strip():
                return reading
    return ""


def compute_metrics(scenes, preds_real, preds_masked,
                    judge_real, judge_masked) -> dict:
    scenes = [s for s in (scenes or []) if isinstance(s, dict)]
    scenes.sort(key=lambda s: str(s.get("id", "")))
    pred_idx = _index_preds(list(preds_real or []) + list(preds_masked or []))
    judge_idx = _index_judgments(list(judge_real or []) + list(judge_masked or []))
    rows: list[dict] = []
    k1 = k2 = 0
    per_series: dict[str, dict] = {}
    crosstab = {label: {"true": 0, "false": 0} for label in RECEPTIONS}
    n_no_pred = 0
    for s in scenes:
        sid = s.get("id", "")
        misunderstood = bool(s.get("오해발생", False))
        per_version: dict[str, dict] = {}
        for version in VERSIONS:
            pb = pred_idx.get((sid, version), {})
            jb = judge_idx.get((sid, version), {})
            verdicts = [(jb.get(st, {}) or {}).get("verdict")
                        for st in PW.STANCES]
            receptions = [(pb.get(st, {}) or {}).get("reception")
                          for st in PW.STANCES]
            success = scene_success(verdicts)
            per_version[version] = {
                "success": bool(success),
                "majority_verdict": majority_verdict(verdicts),
                "majority_reception": majority_reception(receptions),
                "reading": _neutral_reading(pb),
                "n_success_stances": sum(1 for v in verdicts
                                         if _success_of(v)),
            }
        if per_version["real"]["success"]:
            k1 += 1
        if per_version["masked"]["success"]:
            k2 += 1
        series = _as_text(s.get("계열", "")) or "(none)"
        bucket = per_series.setdefault(series, {"n": 0, "k1": 0, "k2": 0})
        bucket["n"] += 1
        bucket["k1"] += 1 if per_version["real"]["success"] else 0
        bucket["k2"] += 1 if per_version["masked"]["success"] else 0
        rec = per_version["real"]["majority_reception"]
        if rec in RECEPTIONS:
            crosstab[rec]["true" if misunderstood else "false"] += 1
        else:
            n_no_pred += 1
        rows.append({
            "id": sid,
            "계열": _as_text(s.get("계열", "")),
            "작품": _as_text(s.get("작품", "")),
            "발화": _as_text(s.get("발화", "")),
            "오해발생": bool(misunderstood),
            "real": per_version["real"],
            "masked": per_version["masked"],
        })
    n = len(scenes)
    m1 = (k1 / n) if n else 0.0
    m2 = (k2 / n) if n else 0.0
    m3 = m1 - m2
    ci1 = wilson_ci(k1, n)
    ci2 = wilson_ci(k2, n)
    ci3 = newcombe_diff_ci(k1, n, k2, n)
    series_out = {}
    for key in sorted(per_series):
        b = per_series[key]
        nn = b["n"]
        series_out[key] = {
            "n": int(nn),
            "k1": int(b["k1"]),
            "m1": (b["k1"] / nn) if nn else 0.0,
            "m1_ci": list(wilson_ci(b["k1"], nn)),
            "k2": int(b["k2"]),
            "m2": (b["k2"] / nn) if nn else 0.0,
            "m2_ci": list(wilson_ci(b["k2"], nn)),
            "m3": ((b["k1"] - b["k2"]) / nn) if nn else 0.0,
        }
    return {
        "n_scenes": int(n),
        "k1": int(k1),
        "m1": float(m1),
        "m1_ci": [float(ci1[0]), float(ci1[1])],
        "k2": int(k2),
        "m2": float(m2),
        "m2_ci": [float(ci2[0]), float(ci2[1])],
        "m3": float(m3),
        "m3_ci": [float(ci3[0]), float(ci3[1])],
        "m3_ci_method": "newcombe-wilson",
        "verdict": amendment12_verdict(m1, m3),
        "per_계열": series_out,
        "crosstab": {label: dict(crosstab[label]) for label in RECEPTIONS},
        "n_no_pred": int(n_no_pred),
        "rows": rows,
    }


def _cell(text, limit) -> str:
    s = text if isinstance(text, str) else str(text)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit]
    return s.replace("|", "\\|")


def _fmt(x) -> str:
    return f"{float(x):.3f}"


def write_report_markdown(out_path, metrics) -> dict:
    m = metrics or {}
    n = int(m.get("n_scenes", 0) or 0)
    k1, k2 = int(m.get("k1", 0) or 0), int(m.get("k2", 0) or 0)
    ci1 = m.get("m1_ci", [0.0, 0.0])
    ci2 = m.get("m2_ci", [0.0, 0.0])
    ci3 = m.get("m3_ci", [0.0, 0.0])
    lines: list[str] = []
    lines.append("# Scenes eval report (Amendment 12: famous misreadings)")
    lines.append("")
    lines.append(f"n_scenes={n}")
    lines.append(
        f"M1={_fmt(m.get('m1', 0.0))} [{_fmt(ci1[0])}, {_fmt(ci1[1])}] "
        f"({k1}/{n}, real, Wilson 95%)")
    lines.append(
        f"M2={_fmt(m.get('m2', 0.0))} [{_fmt(ci2[0])}, {_fmt(ci2[1])}] "
        f"({k2}/{n}, masked, Wilson 95%)")
    lines.append(
        f"M3={_fmt(m.get('m3', 0.0))} [{_fmt(ci3[0])}, {_fmt(ci3[1])}] "
        f"(M1-M2, {m.get('m3_ci_method', 'newcombe-wilson')} 95%)")
    lines.append(f"verdict: {m.get('verdict', '')}")
    lines.append("")
    lines.append("## Per-계열 breakdown")
    lines.append("")
    lines.append("| 계열 | n | M1 (real) | M2 (masked) | M3 |")
    lines.append("|---|---|---|---|---|")
    per = m.get("per_계열", {}) or {}
    for key in sorted(per):
        b = per[key]
        bci1 = b.get("m1_ci", [0.0, 0.0])
        bci2 = b.get("m2_ci", [0.0, 0.0])
        lines.append(
            f"| {_cell(key, 40)} | {int(b.get('n', 0) or 0)} "
            f"| {_fmt(b.get('m1', 0.0))} [{_fmt(bci1[0])}, {_fmt(bci1[1])}] "
            f"({int(b.get('k1', 0) or 0)}/{int(b.get('n', 0) or 0)}) "
            f"| {_fmt(b.get('m2', 0.0))} [{_fmt(bci2[0])}, {_fmt(bci2[1])}] "
            f"({int(b.get('k2', 0) or 0)}/{int(b.get('n', 0) or 0)}) "
            f"| {_fmt(b.get('m3', 0.0))} |")
    lines.append("")
    lines.append("## Reception (real, majority) vs 오해발생")
    lines.append("")
    lines.append("| reception | 오해발생 true | 오해발생 false | total |")
    lines.append("|---|---|---|---|")
    ct = m.get("crosstab", {}) or {}
    tot_t = tot_f = 0
    for label in RECEPTIONS:
        row = ct.get(label, {}) if isinstance(ct, dict) else {}
        t = int(row.get("true", 0) or 0)
        f = int(row.get("false", 0) or 0)
        tot_t += t
        tot_f += f
        lines.append(f"| {label} | {t} | {f} | {t + f} |")
    lines.append(f"| total | {tot_t} | {tot_f} | {tot_t + tot_f} |")
    if int(m.get("n_no_pred", 0) or 0):
        lines.append("")
        lines.append(f"scenes without a real-version prediction: "
                     f"{int(m.get('n_no_pred', 0))} (excluded above)")
    lines.append("")
    lines.append("## Scenes")
    lines.append("")
    lines.append("| id | 계열 | 작품 | 발화 ≤80 | real reading ≤100 | "
                 "real judgment | masked reading ≤100 | masked judgment |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in (m.get("rows", []) or []):
        if not isinstance(r, dict):
            continue
        real = r.get("real", {}) if isinstance(r.get("real"), dict) else {}
        masked = r.get("masked", {}) if isinstance(r.get("masked"), dict) else {}
        lines.append(
            f"| {_cell(r.get('id', ''), 40)} "
            f"| {_cell(r.get('계열', ''), 20)} "
            f"| {_cell(r.get('작품', ''), 60)} "
            f"| {_cell(r.get('발화', ''), 80)} "
            f"| {_cell(real.get('reading', ''), 100)} "
            f"| {_cell(real.get('majority_verdict', ''), 20)} "
            f"{int(real.get('n_success_stances', 0) or 0)}/3 "
            f"{'✓' if real.get('success') else '✗'} "
            f"| {_cell(masked.get('reading', ''), 100)} "
            f"| {_cell(masked.get('majority_verdict', ''), 20)} "
            f"{int(masked.get('n_success_stances', 0) or 0)}/3 "
            f"{'✓' if masked.get('success') else '✗'} |")
    lines.append("")
    out_p = Path(out_path)
    if str(out_p.parent) and str(out_p.parent) != ".":
        out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return m


def generate_report(scenes_path, pred_real_path, pred_masked_path,
                    judge_real_path, judge_masked_path, out_md_path) -> dict:
    scenes = _load_scenes(scenes_path)
    preds_real = _load_jsonl(pred_real_path)
    preds_masked = _load_jsonl(pred_masked_path)
    judge_real = _load_jsonl(judge_real_path)
    judge_masked = _load_jsonl(judge_masked_path)
    metrics = compute_metrics(scenes, preds_real, preds_masked,
                              judge_real, judge_masked)
    write_report_markdown(out_md_path, metrics)
    return metrics


# --- CLI -------------------------------------------------------------------

def main(argv=None) -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(prog="uptake-scenes-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--src", required=True)
    pp.add_argument("--out", required=True)

    rp = sub.add_parser("read")
    rp.add_argument("--scenes", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--model", default=DEFAULT_MODEL)
    rp.add_argument("--effort", default="low")
    rp.add_argument("--budget", type=float, default=0.5)
    rp.add_argument("--workers", type=int, default=10)
    rp.add_argument("--masked", action="store_true")
    rp.add_argument("--real", action="store_true")

    jp = sub.add_parser("judge")
    jp.add_argument("--preds", required=True)
    jp.add_argument("--scenes", required=True)
    jp.add_argument("--out", required=True)
    jp.add_argument("--model", default=DEFAULT_MODEL)
    jp.add_argument("--effort", default="low")
    jp.add_argument("--budget", type=float, default=1.0)
    jp.add_argument("--workers", type=int, default=10)

    op = sub.add_parser("report")
    op.add_argument("--out", required=True)
    op.add_argument("--scenes", default=None)
    op.add_argument("--pred-real", default=None)
    op.add_argument("--pred-masked", default=None)
    op.add_argument("--judge-real", default=None)
    op.add_argument("--judge-masked", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        counts = prepare_corpus(args.src, args.out)
        per = counts.get("per_계열", {})
        per_s = ",".join(f"{k}={per[k]}" for k in sorted(per))
        print(f"done scenes {counts['n_kept']}/{counts['n_total']} "
              f"dropped={counts['n_dropped']} "
              f"per_계열={per_s} out={args.out}/scenes.json")
        return
    if args.cmd == "read":
        if bool(args.masked) == bool(args.real):
            raise SystemExit("read needs exactly one of --masked / --real")
        version = "masked" if args.masked else "real"
        from .reader import MuseClient

        scenes = _load_scenes(args.scenes)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        meter = Meter(budget_usd=args.budget)
        inner = MuseClient(meter=meter,
                           default_effort=_normalize_effort(args.effort))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = LoggingClient(inner, CallLog(out / "calls.jsonl"), run_id)
        settings = {"scenes": args.scenes, "out": args.out,
                    "version": version, "budget": args.budget,
                    "workers": args.workers, "effort": args.effort,
                    "model": args.model}
        run, _preds = read_scenes(scenes, version, out, client,
                                  model=args.model, meter=meter,
                                  workers=args.workers, settings=settings,
                                  run_id=run_id, commit=_git_commit())
        run["spend"] = meter.by_model()
        run["cost"] = float(meter.cost())
        with open(out / f"run_{version}.json", "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        print(f"done pred_{version} {run['n_preds']}/{run['n_expected']} "
              f"parse_fail={run['parse_fail_rate']:.3f} "
              f"budget_stop={run['budget_stop']} "
              f"spend: ${run['cost']:.4f}")
        return
    if args.cmd == "judge":
        from .reader import MuseClient

        preds = _load_jsonl(args.preds)
        scenes = _load_scenes(args.scenes)
        truths = {s.get("id", ""): {"해석": s.get("해석", ""),
                                    "근거": s.get("근거", "")}
                  for s in scenes}
        out_p = Path(args.out)
        log_dir = out_p if out_p.is_dir() or args.out.endswith("/") else out_p.parent
        if str(log_dir) and str(log_dir) != ".":
            log_dir.mkdir(parents=True, exist_ok=True)
        meter = Meter(budget_usd=args.budget)
        inner = MuseClient(meter=meter,
                           default_effort=_normalize_effort(args.effort))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = Path(str(log_dir) or ".")
        client = LoggingClient(inner, CallLog(log_dir / "calls.jsonl"), run_id)
        judgments, stats = judge_preds(preds, truths, args.out, client,
                                       model=args.model, meter=meter,
                                       workers=args.workers)
        print(f"done judge_{stats['version']} {stats['n_judged']}/"
              f"{stats['n_preds']} "
              f"parse_fail={stats['parse_fail_rate']:.3f} "
              f"budget_stop={stats['budget_stop']} "
              f"spend: ${meter.cost():.4f} out={stats['out']}")
        return
    out_p = Path(args.out)
    base = out_p.parent if str(out_p.parent) and str(out_p.parent) != "." else Path(".")
    metrics = generate_report(
        args.scenes or str(base / "scenes.json"),
        args.pred_real or str(base / "pred_real.jsonl"),
        args.pred_masked or str(base / "pred_masked.jsonl"),
        args.judge_real or str(base / "judge_real.jsonl"),
        args.judge_masked or str(base / "judge_masked.jsonl"),
        args.out)
    ci1 = metrics["m1_ci"]
    ci3 = metrics["m3_ci"]
    print(f"done M1={metrics['m1']:.3f} [{ci1[0]:.3f},{ci1[1]:.3f}] "
          f"M2={metrics['m2']:.3f} M3={metrics['m3']:.3f} "
          f"[{ci3[0]:.3f},{ci3[1]:.3f}] "
          f"verdict={metrics['verdict']} out={args.out}")
