# SPEC task21 — L7 해설: one-line Go-broadcast commentary on pivotal turns

Goal: for one analysed conversation (item JSON from `batch.run_item`), pick the turns where the position (형세) swung most
and write one short broadcast-style line per turn in Korean and English, each grounded in exactly ONE reference from
`forecaster/references.json` (curated table — never cite anything else; do not edit the table).

## Inputs
- `item`: dict with `id`, `moves` (list of {turn_index, speaker, moves}), `effects` (list with index, speaker, text,
  relationship, face, disagreement, n_readings ...).
- `norm`: `{"mu": float, "sd": float}` (from `calibrate.fit_board_norm(..., method="z")` on a fit set).
- optional `markov`: table from `predict.fit_markov` (fit set) for the 특이수 rule.

## New module `forecaster/commentary.py`
- `load_references(path=None) -> list[dict]`: reads `references.json` next to this module (`references` list).
- `turn_values(item, norm) -> list[dict]`: per effect in order: `{"index", "speaker", "text", "moves" (from item["moves"] by turn_index,
  first entry; [] if missing), "value" (board._value_of − mu), "z" (value / sd; 0.0 if sd == 0), "disagreement"}`.
- `side_curves(item, norm) -> dict`: `{"A": [...], "B": [...], "balance": [...]}`, one entry per effect: each side's cumulative sum
  of centered value over that side's OWN turns (carried forward on the other side's turns); balance = A − B. Speakers other
  than A/B are ignored for A/B but still produce an entry (carry forward).
- `mark_for(z, surprise) -> str | None`: z ≥ 1.0 → "!" (묘수); z ≤ −1.0 → "?" (실수); else surprise True → "!?" (특이수); else None.
- `surprise(prev_class, cur_class, markov, threshold=0.05) -> bool`: markov is None → False; else `markov[prev_class][cur_class] < threshold`.
  Classes via `board.actual_class(moves)`; first turn's prev_class = "proceed".
- `pick_turns(item, norm, markov=None, k=3) -> list[dict]`: candidates = turns with a mark. Order by |z| descending, ties by index;
  take the top k; additionally include at most one "!?" turn (the first one by index not already picked). Return sorted by index,
  each entry = turn_values entry + `"mark"`, `"label_ko"` ({"!": "묘수", "?": "실수", "!?": "특이수"}), `"label_en"`
  ({"!": "strong move", "?": "mistake", "!?": "unusual move"}), `"curve_before"`, `"curve_after"` (the speaker's side curve value
  before/after this turn).
- `choose_reference(turn, refs, used=()) -> dict`: candidates = refs whose `marks` contain turn["mark"] and whose `moves`
  intersect turn["moves"]; score = size of the intersection; prefer ids not in `used`; ties → table order.
  If none, fall back to refs with "*" in moves and the mark in marks (→ austin_uptake). Never returns None.
- `template_line(turn, ref) -> dict`: deterministic, no model call:
  ko = f"{turn['index']}수 {speaker} {label_ko}({mark}) — {move words} 뒤 형세 {curve_before:+.1f}→{curve_after:+.1f}. {ref['idea_ko']}"
  en = f"Turn {index}, {speaker}: {label_en} ({mark}) — after {move words}, position {before:+.1f}→{after:+.1f}. {ref['idea_en']}"
  where move words = the turn's moves joined with ", " (or "a move" / "수" if empty). Returns `{"ko", "en", "source": "template"}`.
- `llm_line(client, turn, ref, model, meta=None) -> dict`: one chat call. System prompt: Go/baduk TV commentator; one line per
  language, ≤ 120 characters each, describe what the move did to the position and why, using ONLY the given reference idea; may
  name the reference author in short form; no other names, books, papers or years. Ask for JSON `{"ko": "...", "en": "..."}`.
  Validation → fall back to `template_line` (source "template_fallback") when: parse fails, a field is missing/empty, any field is
  longer than 160 chars, or the text contains a 4-digit year other than the reference's year, or the text contains a quotation of a
  title in quotes/『』 that is not the reference title. Success → source "llm".
  Client interface: `client.complete(messages, model, temperature)` like `board._call_board` (pass meta when `client.accepts_meta`).
- `comment(item, norm, markov=None, k=3, client=None, model="muse-spark-1.3-contributor", refs=None) -> dict`:
  `{"id", "norm": {"mu","sd"}, "curves": side_curves, "turns": [picked turn + "reference": {id, author, title, year, venue} +
  "line_ko", "line_en", "source"]}`. References chosen in index order with `used` accumulating. client None → template lines.
- CLI `uptake-comment --item FILE --fit DIR [--k 3] [--llm] [--budget 0.5] [--model ...]`: norm = fit_board_norm(z) and markov
  (base_rates + fit_markov on `predict.with_context` hits) from the fit DIR's items of the item's genre; with `--llm` build
  `reader.MuseClient(meter=usage.Meter(budget_usd=budget))`; writes `<item dir>/commentary_<id>.json` (or `--out`), prints the lines.
  Register in pyproject.

## Tests `tests/test_task21.py` (Muse writes)
- every reference in references.json has id, author, title, year, idea_en, idea_ko, moves, marks; ids unique.
- mark_for thresholds; side_curves carry-forward and balance on a 4-turn toy; pick_turns order/k and the extra "!?".
- choose_reference prefers larger move overlap, avoids `used`, falls back to austin_uptake.
- llm_line with a fake client: valid JSON → source "llm"; a line containing "(1999)" or 『Fake Book』 → template_fallback; bad JSON → template_fallback.
- comment(...) with client None returns only references whose ids exist in the table.
