# SPEC task22 — L9 화면: single static HTML viewer (no server)

Goal: open one analysed conversation like a Go broadcast board: the 형세 curve for both sides, pivotal-turn marks with the L7
commentary, the full turn list, and a final score panel. One self-contained .html file: inline CSS + inline JS, no external
scripts/fonts/CDNs. The only external resource allowed is an optional YouTube iframe (youtube-nocookie.com) when `--youtube-id` is given.

## New module `forecaster/viewer.py`
- `build_view(item, commentary, outcomes=None, outcome_stats=None, youtube_id=None, turns=None) -> dict` (pure, testable):
  `{"id", "title", "turns": [{index, speaker, text, moves, value, z, mark (or null), t (seconds or null)}],
    "curves": commentary["curves"], "pivots": commentary["turns"], "score": {...}, "outcomes": outcomes or null,
    "validity": outcome_stats subset or null, "youtube_id", "dataset": credit or null}`.
  - value/z from `commentary.turn_values(item, commentary["norm"])`; mark from the pivot with the same index.
  - score: per side A/B: final side curve value, counts of "!", "?", "!?" among pivots, number of turns.
  - outcomes (CaSiNo): `outcome.load_outcomes(...)[dialogue_id]` → shown as "reported by participants after the negotiation".
  - validity: from an `outcome.json` (uptake-outcome): C1 and C2 `[point, lo, hi]` and n, shown as a one-line note
    ("on N held-out dialogues the curve tracks how much the other side liked the speaker: ρ = x [lo, hi]").
  - dataset credit: when item id starts with "casino-", the `datasets[0]` entry of references.json.
  - t: when youtube_id and `turns` (list of transcribe.Turn with chunk) are given: t = chunk·540 + 540·(chars of earlier turns in the
    same chunk / total chars of that chunk), rounded down to int; shown with "≈". Otherwise null.
- `render_html(view) -> str`:
  - Data embedded once as `<script type="application/json" id="data">…</script>`; every "<" in the JSON written as "<"
    so turn text can never close the tag. All text inserted with textContent (no innerHTML with data).
  - Layout (top → bottom on phones, two columns ≥ 900px): header (title, dataset credit, note "model reading of moves, not a judgment
    of people"); 형세 chart as inline SVG drawn by JS: line A, line B, balance (A−B) dashed, zero line, x = turn index, markers with
    the mark glyph at pivotal turns (click → scroll to that turn and highlight); optional YouTube iframe (clicking a turn with t seeks by
    reloading the iframe src with `start=t`); turn list (index, speaker, text, move chips, small centered value bar, pivot badge);
    pivot cards (mark, label ko/en, line ko/en, reference "author (year), title"); score panel (per side final value and mark counts;
    outcomes if present; validity note if present).
  - Language toggle KO/EN switching labels and commentary lines. Colors as CSS variables with a dark-mode media query.
  - Body max width ~1100px, 16px side padding, no horizontal scroll at 375px width.
- CLI `uptake-viewer --item FILE --commentary FILE --out FILE.html [--casino PATH] [--outcome FILE] [--youtube-id ID] [--turns FILE]`.
  Register in pyproject.

## Tests `tests/test_task22.py` (Muse writes)
- render_html output has no `<script src`, no `<link rel="stylesheet"`, no http(s) URL except youtube-nocookie when youtube_id given.
- a turn text containing `</script><b>x` survives JSON embedding: the raw HTML does not contain `</script><b>x`.
- view contains pivots with their reference titles; score counts match pivots; t computed per the formula on a toy 2-chunk turn list.
- no iframe in HTML when youtube_id is None.
