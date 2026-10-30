# SPEC task46 — `uptake-perturb`: single-turn edit sensitivity test (Amendment 11)

Implements Amendment 11 in `specs/EVAL_PLAN_overnight_2.md`. New module `forecaster/perturb.py`, entry point
`uptake-perturb` with subcommands:

## `uptake-perturb select --records DIR --table FILE --ids-file FILE --n 20 --out DIR`
- Takes the first `n` ids from the ids file, loads each pairwise record, computes z = (value − mu)/sd per effect from the
  table, picks the median-|z| turn (ties → earlier turn; skip turns with < 6 words), writes `OUT/targets.json`:
  `[{dialogue, turn, speaker, text, z}]`.

## `uptake-perturb write --targets FILE --out DIR --model M --effort low --budget B`
- For each target asks the model twice with a fixed prompt (kept in the module, with the exact wording in a constant so it
  is auditable): RUDE and POLITE rewrites of `text` that keep the propositional content (same offer/ask/numbers) and change
  only the interpersonal wording. One short paragraph each, same language as the original. Writes `OUT/variants.json`:
  `[{dialogue, turn, orig, rude, polite}]` and `OUT/calls.jsonl` (reuse the existing API wrapper and cost accounting used by
  `pairwise.py`; parse failures → retry once, then record `null` and skip the target).

## `uptake-perturb vote --variants FILE --records DIR --out DIR --model M --effort low --budget B --seed 0`
- For each target builds the three pairs (orig,rude), (orig,polite), (rude,polite) in both presentation orders and asks the
  three stances used by `pairwise.py` (reuse its judge prompt and parser; the two items are the same turn position in the
  same dialogue context, so show the shared context once and the two candidate turn texts as A/B). Writes `OUT/votes.jsonl`
  and `OUT/run.json` (counts, cost, parse-failure rate, position bias).

## `uptake-perturb report --votes FILE --variants FILE --out FILE.md`
- Majority vote per pair (ties → not predicted). Metrics exactly as Amendment 11: ordering accuracy (polite > orig > rude),
  per-pair predicted-direction counts (x/20), mean margin, position bias; verdict line per the pre-stated thresholds.
  Markdown table of the 20 targets with the three texts (truncated to 120 chars) and the three pair outcomes.

Rules: stdlib only; no network in tests (mock the API wrapper the way `tests/test_task34.py` does). Tests in
`tests/test_task46.py`: select picks the median-|z| turn on a toy record; write stores both variants and handles a null;
vote builds 6 presentations × 3 stances per target and records position; report computes the metrics on a fixture with a
known answer (e.g., 17/20 ordering → 0.85, verdict "directionally sensitive"). `pytest -q` stays green.
