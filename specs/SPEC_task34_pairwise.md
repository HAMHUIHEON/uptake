# SPEC task34 — pairwise-comparison reader and Bradley–Terry ratings (Amendment 7, P7-A)

New module `forecaster/pairwise.py`, CLI `uptake-pairwise` (register in pyproject). Stdlib only.

## `uptake-pairwise run --src DIR --out DIR [--k 4] [--seed 0] [--budget 1.5] [--workers 10] [--effort low] [--model ...]`
- Load records from `--src` (same loader as `rescore`/`scalecheck`; they only supply turns, moves,
  speakers, genre). Pool = every effect turn (dialogue id, turn index, speaker, listener).
- Pairing: for each turn, sample `k` partner turns uniformly from other dialogues of the same
  genre (random.Random(seed)); dedupe unordered pairs; each pair gets a random presentation order
  (`first`/`second`) stored in the pair record.
- Prompt: for each of the three board stances (reuse the receiver personas from `board`), one call
  per pair. The user message shows snippet 1 and snippet 2, each = the 6-turn window ending at the
  judged message plus "the message to judge" marked, with the listener role stated. Ask: after
  which message is the listener more favourably disposed toward the speaker? Answer JSON
  `{"better": "first" | "second" | "tie", "why": "<one sentence>"}`.
- Log every call to `OUT/calls.jsonl` with `meta = {"layer": "pairwise", "pair": id, "stance", ...}`
  exactly like `rescore` (so `scalecheck.board_cost` can price it). Honour `--budget` through the
  usual meter; stop cleanly when exhausted and record what was done.
- Write `OUT/pairs.jsonl`: one line per pair with turn refs, order, the three votes (de-randomized
  to turn-a / turn-b / tie), parse failures as `null` votes. Write `OUT/run.json` (counts, cost,
  parse-failure rate, position-bias share = fraction of "first" among non-tie votes).

## `uptake-pairwise fit --pairs DIR --src DIR --out DIR [--half-seed 1]`
- Bradley–Terry on all non-null votes (tie = 0.5 win each side), MM updates (Hunter 2004) with a
  uniform prior of 0.5 pseudo-win against a virtual reference turn for connectivity; 200
  iterations or convergence 1e-6; ratings centred to mean 0 (log scale).
- `OUT/ratings.json`: rating per (dialogue, turn index) and reception_pw per (dialogue, side) =
  mean rating of that side's turns.
- Split-half: split pairs randomly in two halves (half-seed), fit each, Spearman between the two
  rating vectors over turns present in both → `OUT/reliability.json` (also report n turns).
- `OUT/records/`: a copy of each source record where every effect's `value` is replaced by its
  rating (`n_readings` = number of votes it took part in; effects with no rating keep
  `n_readings` 0) and top-level `scale = "pairwise"`. This lets `uptake-scale residual` /
  `outcome` compute C1/R1 unchanged.

## fine100 split-half (for the comparison, same module)
- `uptake-pairwise reliability-fine100 --src DIR --out FILE`: per turn, mean of readings from
  stances {neutral, trusting} vs reading from {suspicious}, and {neutral, suspicious} vs
  {trusting}; Spearman over turns for each split, report both and their mean.

## tests `tests/test_task34.py` (Muse writes)
- BT recovers a planted ordering from synthetic votes; ties handled; centred ratings.
- pairing: every turn appears in ≥ k pairs, no within-dialogue pairs, orders ≈ balanced.
- vote de-randomization correct for both presentation orders.
- emitted records keep the keys scalecheck/outcome read; `scale == "pairwise"`.
- fine100 split-half on toy readings.
- `pytest -q` must stay green.
