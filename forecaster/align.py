"""Transcript <-> audio alignment for the audio panel (task59).

For cases with real public audio, the text panel reads the
case transcript while the audio panel (task36) reads turns cut from the
recording. This module aligns each record turn (by ``index``) to a fresh
transcription of the audio so both panels read the same turns.

Pipeline (``uptake-align --audio WAV --record JSON --out JSON [--model M]``):

1. Transcribe the WAV with the existing transcriber
   (:mod:`forecaster.transcribe`, same model/flags as task07/36: diarization
   model ``muse-voice-transcribe-1.0`` with per-turn timestamps) to get
   segments with start/end seconds and text.
2. Greedily align each record turn, in index order, to the transcript
   segments: monotone (later turns only match later segments), maximizing
   token-overlap similarity. A match is accepted when the similarity is at
   least 0.35, else the turn is marked ``unaligned``.
3. Write the record with ``effects[i].start``/``end`` filled (seconds in the
   audio) plus a top-level ``alignment`` report
   ``{aligned, unaligned, mean_similarity, offset_hint}``.

Tokenization is lowercased alphanumeric runs (``[a-z0-9]+``), so digits and
callsign-style tokens are kept. Similarity is the
Jaccard index over token sets (0.0 when either side has no tokens).

``mean_similarity`` is the mean over aligned turns only (0.0 when nothing
aligned). ``offset_hint`` is the start seconds of the first aligned segment,
i.e. where the record content begins in the audio (0.0 when nothing aligned).

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from . import transcribe as _transcribe

SIM_THRESHOLD = 0.35
DEFAULT_MODEL = _transcribe.REQUEST_PAYLOAD["model"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; digits and callsign tokens kept."""
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def similarity(a_text: str, b_text: str) -> float:
    """Token-overlap (Jaccard) similarity in [0, 1].

    1.0 for identical token sets, 0.0 when either side has no tokens.
    """
    set_a = set(tokenize(a_text))
    set_b = set(tokenize(b_text))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def segments_from_turns(turns) -> list[dict]:
    """Normalize timestamped turns to ``{start, end, text}`` segment dicts.

    Turns without a usable (positive) start/end span are skipped: they
    cannot anchor a record turn in the audio.
    """
    segments: list[dict] = []
    for turn in turns or []:
        if isinstance(turn, dict):
            start = turn.get("start")
            end = turn.get("end")
            text = turn.get("text", "")
        else:
            start = getattr(turn, "start", None)
            end = getattr(turn, "end", None)
            text = getattr(turn, "text", "")
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if not (end_f > start_f):
            continue
        segments.append({"start": start_f, "end": end_f,
                         "text": text if isinstance(text, str) else ""})
    return segments


def align_texts(texts: list[str], segments: list[dict],
                threshold: float = SIM_THRESHOLD) -> list[dict]:
    """Greedy monotone alignment of record texts to transcript segments.

    Each text, in order, takes the highest-similarity segment at or after
    the cursor; the cursor then moves past the matched segment so later
    turns only match later segments. Ties break towards the earliest
    segment. A match below ``threshold`` leaves the turn unaligned and the
    cursor unmoved.

    Returns one dict per text: ``{segment, similarity, start, end,
    aligned}`` where ``segment`` is the matched segment position (or None).
    """
    matches: list[dict] = []
    cursor = 0
    for text in texts:
        best_j: int | None = None
        best_sim = 0.0
        for j in range(cursor, len(segments)):
            sim = similarity(text, segments[j].get("text", ""))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None and best_sim >= threshold:
            seg = segments[best_j]
            matches.append({"segment": best_j, "similarity": best_sim,
                            "start": seg["start"], "end": seg["end"],
                            "aligned": True})
            cursor = best_j + 1
        else:
            matches.append({"segment": None, "similarity": best_sim,
                            "start": None, "end": None, "aligned": False})
    return matches


def _effect_index(effect: dict, pos: int) -> int:
    idx = effect.get("index", pos)
    return idx if isinstance(idx, int) else pos


def align_record(record: dict, segments: list[dict],
                 threshold: float = SIM_THRESHOLD) -> tuple[dict, dict]:
    """Align a record's effects to transcript segments.

    Effects are aligned in ``index`` order; the output record keeps the
    original effect order with ``start``/``end`` filled on aligned turns,
    ``aligned`` (bool) and ``similarity`` (float) set on every turn, and a
    top-level ``alignment`` report. A ``board`` list mirroring ``effects``
    is updated the same way when present. The input record is not mutated.
    """
    effects = (record or {}).get("effects", []) or []
    order = sorted(range(len(effects)),
                   key=lambda pos: _effect_index(effects[pos], pos)
                   if isinstance(effects[pos], dict) else pos)
    texts = [effects[pos].get("text", "") if isinstance(effects[pos], dict)
             else "" for pos in order]
    matches = align_texts(texts, segments, threshold)

    by_pos = dict(zip(order, matches))
    out_effects: list[dict] = []
    for pos, effect in enumerate(effects):
        if not isinstance(effect, dict):
            out_effects.append(effect)
            continue
        match = by_pos[pos]
        updated = dict(effect)
        updated["aligned"] = bool(match["aligned"])
        updated["similarity"] = float(match["similarity"])
        if match["aligned"]:
            updated["start"] = float(match["start"])
            updated["end"] = float(match["end"])
        else:
            updated.pop("start", None)
            updated.pop("end", None)
        out_effects.append(updated)

    aligned_idx = sorted(
        _effect_index(effects[pos], pos) for pos in order
        if isinstance(effects[pos], dict) and by_pos[pos]["aligned"])
    unaligned_idx = sorted(
        _effect_index(effects[pos], pos) for pos in order
        if isinstance(effects[pos], dict) and not by_pos[pos]["aligned"])
    aligned_sims = [by_pos[pos]["similarity"] for pos in order
                    if isinstance(effects[pos], dict)
                    and by_pos[pos]["aligned"]]
    first_start: float | None = None
    for pos in order:
        if isinstance(effects[pos], dict) and by_pos[pos]["aligned"]:
            first_start = float(by_pos[pos]["start"])
            break
    report = {
        "aligned": len(aligned_idx),
        "unaligned": len(unaligned_idx),
        "mean_similarity": (sum(aligned_sims) / len(aligned_sims)
                            if aligned_sims else 0.0),
        "offset_hint": first_start if first_start is not None else 0.0,
        "threshold": float(threshold),
        "aligned_indices": aligned_idx,
        "unaligned_indices": unaligned_idx,
    }

    out = dict(record)
    out["effects"] = out_effects
    board = out.get("board")
    if isinstance(board, list) and len(board) == len(out_effects):
        synced: list = []
        for orig, upd in zip(board, out_effects):
            if isinstance(orig, dict) and isinstance(upd, dict):
                row = dict(orig)
                row["aligned"] = upd["aligned"]
                row["similarity"] = upd["similarity"]
                if upd.get("aligned"):
                    row["start"] = upd["start"]
                    row["end"] = upd["end"]
                else:
                    row.pop("start", None)
                    row.pop("end", None)
                synced.append(row)
            else:
                synced.append(orig)
        out["board"] = synced
    out["alignment"] = report
    return out, report


def align_audio_to_record(audio_path, record: dict, work_dir=None,
                          model: str = DEFAULT_MODEL) -> tuple[dict, dict]:
    """Transcribe ``audio_path`` and align ``record`` to the transcript."""
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="uptake-align-") as tmp:
            turns = _transcribe.transcribe_file(
                audio_path, tmp, timestamps=True, model=model)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        turns = _transcribe.transcribe_file(
            audio_path, work, timestamps=True, model=model)
    return align_record(record, segments_from_turns(turns))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-align")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--work", default=None,
                    help="chunk work dir (default: temp dir)")
    args = ap.parse_args(argv)

    with open(args.record, encoding="utf-8") as f:
        record = json.load(f)
    if not isinstance(record, dict) or not isinstance(
            record.get("effects"), list):
        ap.error("--record must be a record JSON object with an effects list")

    out_record, report = align_audio_to_record(
        args.audio, record, work_dir=args.work, model=args.model)

    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_record, f, ensure_ascii=False, indent=2)
    total = report["aligned"] + report["unaligned"]
    print(f"aligned {report['aligned']}/{total} turns "
          f"mean_sim {report['mean_similarity']:.3f} "
          f"offset {report['offset_hint']:.1f}s -> {args.out}")
    if report["unaligned_indices"]:
        print(f"unaligned turns: {report['unaligned_indices']}")
