# SPEC task23 — L2 양쪽 판돈: outcome axis for BOTH sides (CaSiNo negotiation)

Why: the relational curve (listener reception) is one stake. In negotiation each side also has an outcome stake (points).
CaSiNo gives both participants' private priorities (participant_info[agent]["value2issue"] = {"High": item, "Medium": item,
"Low": item}); reported points_scored = Σ packages received × {High: 5, Medium: 4, Low: 3}. Items: Food, Water, Firewood, 3 packages each.

## New module `forecaster/stakes.py`
- `ITEMS = ("Food", "Water", "Firewood")`, `VALUES = {"High": 5, "Medium": 4, "Low": 3}`.
- `priorities(entry) -> {"A": {item: value}, "B": {...}}` from one raw CaSiNo dialogue (A = mturk_agent_1, B = mturk_agent_2).
- `load_priorities(casino_path) -> dict[int, dict]` dialogue_id → priorities.
- `normalize_offer(raw, speaker) -> dict | None`: raw = parsed model JSON `{"is_offer": bool, "speaker_gets": {...}, "other_gets": {...}}`.
  Not an offer → None. For each item: if only one side's count is given, the other = 3 − count; counts must be ints 0..3 and sum to 3,
  else None. Returns `{"A": {item: n}, "B": {item: n}}` (speaker's share goes to the speaker's letter; the other letter gets the rest).
  Speakers other than "A"/"B" → None.
- `offer_points(offer, prio) -> {"A": int, "B": int}` = Σ_item offer[X][item] · prio[X][item].
- `build_offer_messages(turns, i, window=4) -> list[dict]`: system + user; the user shows up to `window` previous turns and the target turn
  ("A: …"/"B: …"), and asks: does the TARGET turn put a complete or partial split of the 3 Food, 3 Water, 3 Firewood packages on the
  table (including accepting/restating the other's split)? Answer JSON only: {"is_offer": true|false, "speaker_gets": {"Food": n, ...},
  "other_gets": {...}} with counts for items mentioned.
- `extract_offers(client, turns, moves, model, max_workers=8, item_id="") -> dict[int, dict]`: call the model only for turns whose
  move tags (by turn_index) include "propose", "counter" or "accept"; parse JSON (tolerate code fences/extra text by taking the first
  {...}); normalize; keep successes. Client interface as in `board._call_board` (meta when `accepts_meta`; meta layer = "offer").
  BudgetExceeded propagates.
- `outcome_track(n_turns, offers, prio) -> list[dict]`: per turn index 0..n−1: points of the latest offer at or before that turn,
  `{"A": int|None, "B": int|None, "offer_turn": int|None}`.
- `evaluate(tracks, outcomes, b=2000, seed=0) -> dict` (tracks: dialogue_id → track list; outcomes from `outcome.load_outcomes`):
  E1 = exact match of last track entry (A and B both equal reported points) over dialogues whose last entry is not None:
  `{"k", "n", "share", "ci": stats.wilson_ci}`; E2 = `outcome.boot_spearman` on rows {"dialogue", "mid": A−B at index n//2 (skip None),
  "final": reported A−B}; E3 = boot_spearman of last-entry A−B vs reported A−B. Also `"coverage"` = share of dialogues with ≥ 1 offer.
- CLI `uptake-stakes --dir DIR --casino PATH [--budget 1.0] [--model muse-spark-1.3-contributor] [--workers 4]`:
  for every casino item in DIR (calibrate._load_items), rebuild turns with `casino.load_casino(path)` (annotated dialogues) or
  `annotated=False` when not found, extract offers with a metered, logged client (`reader.MuseClient(meter=usage.Meter(budget_usd=budget))`
  wrapped in `trace.LoggingClient(..., trace.CallLog(DIR/"calls_stakes.jsonl"), run_id)`), write `DIR/stakes.json`
  ({id: {"offers": …, "track": …}}, resumable: skip ids already present) and `DIR/stakes.md` (E1–E3 + coverage + spend).
  On BudgetExceeded: save what is done and stop. Register in pyproject.

## Tests `tests/test_task23.py` (Muse writes; fake clients only)
- priorities/offer_points reproduce the example: A prio {Firewood: 5, Food: 4, Water: 3}, A gets 3 Firewood + 1 Food → 19;
  B prio {Firewood: 5, Water: 4, Food: 3}, B gets 3 Water + 2 Food → 18.
- normalize_offer completes missing sides, rejects counts > 3 or sums ≠ 3, maps speaker "B" correctly.
- extract_offers only calls the client for propose/counter/accept turns; bad JSON skipped.
- outcome_track carries the latest offer forward; evaluate E1 share on a toy set.
