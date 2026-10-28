# Pre-registered evaluation plan — overnight run 2 (written before any run, 2026-10-28 23:58 KST)

Purpose: test on data nobody has looked at (1) whether calibration / pooling gains found on overnight_1 transfer,
and (2) whether the position curve (형세) means anything outside the model — does it track what the
negotiators themselves reported at the end (마지막 검증)? All rules below are fixed before the runs.

## Data
- **Fit set** = overnight_1 negotiation items (200 CaSiNo dialogues, seed 2). All parameters (base rates, λ, Markov table,
  pool weights a/b, board mean μ and sd) are fit here and then frozen.
- **Held-out set** = a fresh CaSiNo sample: 300 dialogues, seed 3, excluding every dialogue already used
  (moves_casino30, moves_casino30_holdout, manifest_smoke, overnight_1 manifest). Same settings as overnight_1:
  panel n=3, k=1, effort low, model muse-spark-1.3-contributor (public data). No repeat sample (stability already OK in run 1).
- YouTube: no new audio is available locally (batch-2 list has 1 partial download only) → no new YouTube data tonight.
  YouTube stays report-only, private, never committed.
- Budget: hard cap $15 via `uptake-batch --budget 15`. Expected ≈ $3 (overnight_1: ≈ $0.0095 per CaSiNo dialogue).

## Part A — re-analysis of overnight_1 (no model calls)
`uptake-calibrate` and `uptake-predict` at the current commit, plus the Part C outcome analysis on the 200 fit dialogues
(labelled exploratory: these dialogues are also where μ is fit).

## Part B — next-move forecast, fit → held-out transfer
Models (all scored by ln p(actual next class), clusters = dialogues, cluster bootstrap b=2000, seed 0):
`raw_panel` (panel shares with add-α smoothing, no calibration), `base` (fit-set base rates), `panel_base` (λ blend),
`markov` (previous class → next class), `pool` (geometric pool of markov and panel).

| # | Hypothesis | Metric | Supported if |
|---|---|---|---|
| B1 | Calibration fixes the raw panel | panel_base − raw_panel | CI lower > 0 |
| B2 | Markov is a real baseline | markov − base | CI lower > 0 |
| B3 (primary) | Panel adds information beyond Markov | pool − markov | CI lower > 0 |
| B4 | Calibrated panel alone vs Markov | panel_base − markov | reported, no verdict |

Decision rules: B3 supported → keep "panel + Markov" as the next-move hint and report the gain with its CI.
B3 not supported → the panel is not sold as a next-move forecaster; next-move hint = Markov only; panel is used for
reading effects (형세, commentary). Either way the number is reported.

## Part C — outcome validity of the position curve (마지막 검증, new)
CaSiNo has post-negotiation self-reports per participant: points_scored (0–36), satisfaction and opponent_likeness
(5-point: Extremely dis- … Extremely, mapped −2…+2).
Per turn, value v = relationship + face as read by the listener panel; centered c = v − μ (μ from the fit set).
For participant X: **reception(X)** = mean c over X's own turns (how X's moves landed with the other side).

| # | Hypothesis | Metric | Supported if |
|---|---|---|---|
| C1 (primary) | X's moves landing well → the other side likes X more | Spearman(reception(X), likeness reported by X's opponent), 2 rows per dialogue, cluster bootstrap by dialogue | CI lower > 0 on held-out |
| C2 | Dialogue-level 형세 tracks joint satisfaction | Spearman(mean c over all turns, mean satisfaction of both) | CI lower > 0 on held-out |
| C3 (exploratory) | Side advantage tracks points | Spearman(reception(A) − reception(B), points(A) − points(B)) | reported only |
| C4 | Panel beats a move-tag baseline for C1 | tag score(X) = share of X's turns tagged warm (empathize, praise, repair, smalltalk) − share tagged hostile (pressure, challenge, reject); also turn-count baseline. Paired bootstrap of ρ_board − ρ_tag | reported; "panel adds value for outcome reading" only if CI lower > 0 |
| C5 (exploratory) | Big mistakes cost liking | mean opponent likeness for participants with ≥1 turn at z ≤ −2 vs none, difference with CI | reported only |

Decision rules: C1 supported → the 형세 curve is shown in the demo as externally validated ("tracks how much the other
side ended up liking you, ρ = x [CI]"). C1 not supported → the curve is shown as a reading aid only, no validity claim.
If C1 holds but C4 shows the tag baseline is as good → say so; the panel's added value is then explanation, not accuracy.

## Stop rules
- Batch health check after 20 dialogues: failed_rate > 0.20 or error rate > 0.10 → stop, report.
- Spend reaches $15 → batch stops by itself; analyze whatever finished, report n.

## What is not tested tonight
- Pivotal-turn (판세가 갈린 턴) accuracy: no human turning-point labels exist. Pivotal turns are used for commentary (L7)
  and shown in the viewer (L9) as model readings, not validated claims.

## Amendment 1 (2026-10-29 00:02 KST, before the batch started at 00:03)
- The loader keeps only CaSiNo dialogues with strategy annotations (396 of 1,030). After exclusions only **134**
  annotated dialogues remain, so the held-out set is all 134 (seed irrelevant), not 300. Rules above unchanged.
- If time and budget allow, a second held-out set of unannotated CaSiNo dialogues (no gold move labels needed for
  Parts B and C) is run after a loader option is added; results are reported separately as "held-out 2".

## Amendment 2 (2026-10-29 00:17 KST — after Part C was run on the fit set, BEFORE any held-out result was seen)
Exploratory check of pivotal-turn detection (판세가 갈린 턴), since no human labels exist:
- D1: the pivotal turn = `board.turning_point` (largest |value| from turn 2). Its centered value is attributed to its speaker.
  Spearman(pivot value of X, likeness reported by X's opponent) over participants who own a pivotal turn
  vs the same statistic using one random turn of X (mean over 200 random draws, seed 0). Reported only.
- D2: share of dialogues whose pivotal turn's speaker received the lower opponent likeness when pivot value < 0
  (higher when > 0), with Wilson CI vs 0.5. Reported only.
These are labelled exploratory in the report; no verdict.

## Amendment 3 (2026-10-29 00:38 KST — L2 both-sides stakes, before any L2 code or run)
Motivation from the fit set (exploratory): C3 = −0.19 [−0.33, −0.05] — the side whose moves land better tends to score FEWER
points. The relational curve alone cannot be "형세" for a negotiation; an outcome axis per side is needed.
L2 (negotiation): extract each offer on the table (model call only on turns tagged propose/counter), score it for BOTH sides with
the CaSiNo point values (High 5 / Medium 4 / Low 3 per package), carry the latest offer forward.
- E1: for dialogues whose final offer exists, share where the final on-table points equal both participants' reported
  points_scored exactly (Wilson CI). ≥ 0.70 → offer extraction usable.
- E2: Spearman(points A − B of the on-table offer at the middle turn, reported points A − B), dialogue bootstrap CI. Reported.
- E3: does adding the outcome axis change C3? Spearman(offer-points diff at the end, points diff) next to C3. Reported.
Run on the held-out set only; budget for L2 calls ≤ $1 (inside the $15 cap).

## Amendment 4 (2026-10-29 00:42 KST — after a fit-set table of mean value by move type, before held-out results)
Fit-set table: mean centered value is strongly ordered by move type (accept +0.69, smalltalk +0.53 … reject −1.26, pressure −1.74).
So the panel value may be mostly "which move was it". Stronger baseline than C4's warm/hostile share:
- C6: tag-value baseline = for each move, mean centered value over fit-set turns containing it; a turn's expected value = mean of its
  moves' table values (unknown move → 0); X's score = mean over X's turns. Paired bootstrap of ρ(reception) − ρ(tag-value) vs
  opponent likeness on held-out. CI lower > 0 → "panel reading carries outcome information beyond the move type". Otherwise the
  curve could be produced from move tags alone (cheaper), and the panel's role is explanation.

## Amendment 5 (2026-10-29 01:15 KST — prototype of the redesign "형세·판세 as one model", before it is run)
Idea: like a Go engine's value network, make the curve a prediction of the final result. Value model fit on the fit set only
(ridge regression, stdlib), features of participant X from the prefix up to turn t: mean centered reception of X, of the other side,
tag-value of X, of the other side, number of X's turns so far. Target: likeness reported by X's opponent (−2…+2).
Evaluate on held-out (and held-out 2 when done):
- V1: Spearman(prediction at the final turn, opp likeness) and paired difference vs C1 (reception only). Reported; "learned value better"
  only if paired CI lower > 0.
- V2: early warning — Spearman at the turn where 50% of the dialogue's turns are done. CI lower > 0 → the curve says something before the end.
- V3: pivot by the value model = turn with the largest |Δ predicted value| for either side; D1-style Spearman of that Δ (attributed to the
  speaker) vs opp likeness, next to D1. Reported only.
