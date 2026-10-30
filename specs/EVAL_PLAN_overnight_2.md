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

## Amendment 6 (2026-10-29 07:55 KST — scale resolution + residual test, written BEFORE any amendment-6 run or new analysis)
Motivation: C6 not supported on both held-out sets (panel curve ≈ move-type average table), and the panel scale is coarse
(relationship −2..+2 used as −1/0/+1 in practice; face 0..2 → only ≤ 0; "!!" impossible). Approved advice: finer resolution,
residual test, power analysis. Same goal as before: one 형세 model, statistical evaluation as the core.

**Logprob check (done before this text, 2 rejected requests, $0):** the Meta API refuses `logprobs` for muse-spark-1.3-contributor
("logprobs is not supported with reasoning"; effort `none` not offered, `minimal`/`low` also refused). → option (a) expected value over
scale-point probabilities is not possible; option (b) is used.

### Conditions (board panel only; move tags reused from the existing held-out records, no re-tagging)
| id | scale | n | turn value |
|---|---|---|---|
| P0 current | relationship −2..+2, face_threat 0..2 (as run) | 3 | rel + (−face) (existing readings, no new calls) |
| P1 current n=5 | same prompt | 5 | same |
| P2 fine100 | relationship_after 0..100 (50 = unchanged, 0 much colder, 100 much warmer); face 0..100 (50 = no effect, 0 severe threat to my face, 100 strongly affirms my face) | 3 | (rel−50)/25 + (face−50)/25 |
| P3 seven | relationship_shift −3..+3; face_effect −3..+3 (negative = threat, positive = affirms) | 3 | (2/3)·rel + (2/3)·face |
Same model (muse-spark-1.3-contributor), k=1, effort low, window 6, same stance mix, temperature 1.0, same listener rule.

### Step 1 — pilot (paired)
40 dialogues sampled from held-out 2 (overnight_2b, 300) with `random.Random(6).sample(sorted ids, 40)`; the same 40 in every condition.
Per condition, on these 40: distinct turn values; distinct per-reading values; share of readings at each axis endpoint;
z of turn values using the condition's own mean/sd over the 40 dialogues → share of turns with z ≥ +2 ("!!" possible?) and z ≤ −2,
max z; parse-failure rate; Spearman of turn values vs P0 on the same turns; ICC(1) of member values within turns and
Spearman–Brown reliability for n=3 and n=5; cost per dialogue (metered); C1 and C6 difference with CIs (reported only, n=40 underpowered).
**Selection rule (fixed now):** a finer scale "improves resolution" if parse failure ≤ 5% AND distinct turn values ≥ 2× P0 AND
share of turns with z ≥ +2 ≥ 0.5%. If both P2 and P3 qualify, S* = the one with higher Spearman–Brown reliability at n=3; tie → P2.
If neither qualifies → no full run; report pilot only. n=5 (P1) is evaluated in the pilot only (observed vs predicted reliability gain);
the full run uses n=3 for budget reasons.

### Step 2 — full run (only if S* exists)
S* on all 300 held-out 2 dialogues; then held-out 1 (overnight_2, 134) if metered spend so far + estimate ≤ $7.5.
Fit set for μ and the move-type table = overnight_1 as before (Spearman-based metrics are invariant to μ; the table is scale-independent
as a baseline). Metrics, dialogue-level cluster bootstrap b=2000, seed 0, 95% percentile CI:
- C1(S*) = Spearman(reception, opp likeness).
- **C6diff(S*)** = ρ(reception S*) − ρ(tag-value), paired. 
- ΔS* = ρ(reception S*) − ρ(reception P0), paired, same dialogues. Reported.
- **R1 (residual, primary)** = partial Spearman ρ(reception, opp likeness | tag-value): rank the three variables, regress the ranks of
  reception and of likeness on the rank of tag-value (with intercept), Pearson of the two residual vectors. Computed for P0 on held-out 1
  and held-out 2 (existing data, no calls) and for S*.
- R2 (reported) = same partial Spearman with extended controls: tag-value, X's turn count, and X's share of turns containing each move type
  (move types present in ≥ 1% of fit-set turns).
- R3 (reported) = OLS on rank-standardised variables, likeness ~ tag-value + reception: coefficient of reception and ΔR² over tag-value alone, with CIs.
- R4 (reported, P0 only, same scale as the table) = Spearman(mean over X's turns of [turn value − μ − table value of the turn's moves], opp likeness).

### Step 3 — power (no calls)
From the held-out 2 bootstrap replicates of C6diff: SE₃₀₀ = sd of replicates; dialogues needed for 80% power (two-sided α = 0.05, i.e. CI lower > 0)
n = 300·(SE₃₀₀·(1.960+0.842)/δ)² for δ = +0.03 and +0.06 (assumes SE ∝ 1/√n). Same for P0 and S*. Also for R1.

### Decision rules
- **Adopt the finer scale** for the 형세 readings if C6diff(S*) CI lower > 0 OR R1(S*) CI lower > 0 on held-out 2 (held-out 1 reported as replication).
- Panel role: if R1(P0) CI lower > 0 on both held-out sets → the coarse panel already carries outcome information beyond the move type;
  keep the panel term in the value model. If R1 is not supported for P0 and S* → curve = move tags + value table (cheap); panel is used for
  explanation only, and the value model (amendment 5, agreed as the 형세 direction) drops the reception features.
- Several metrics are tested; only C6diff(S*) and R1 carry decisions. Everything else is reported without a verdict.
- Budget: $8 metered for all amendment-6 calls (hard cap via the meter); stop and report whatever finished if reached.

## Amendment 7 (2026-10-29 18:13 KST, written before any pairwise call) — pairwise-comparison pilot; value-model 2×2 reporting

Motivation (from the Amendment 6 counts, 13,635 fine100 readings): judges use only 30 of the 101
values, 90% are multiples of 5, and the five values 45/50/55/60/65 cover 68% of readings. The
grid is not the bottleneck; absolute scoring is. Pairwise judgment ("which of these two messages
left the listener warmer toward the speaker?") is the standard way to get a fine ordering from
coarse absolute judges.

### P7-A pairwise pilot (feasibility + reliability; not a result test)
- Data: the same 40 pilot dialogues as Amendment 6 (ids in data/private/scale6/p2_fine100).
- Pool: every effect turn of those dialogues (speaker, listener, 6-turn window as in the board
  prompt). Each turn is paired with k = 4 turns sampled uniformly from OTHER dialogues of the same
  genre (cross-dialogue pairs, so ratings share one scale across dialogues; within-dialogue pairs
  would lose the level that C1/R1 need). Presentation order randomized per pair (seed 0).
- Judges: the same three receiver stances as the board panel (neutral / trusting / suspicious),
  one reading each per pair, effort low, same model as the board.
- Rating: Bradley–Terry (ties = half win each, MM iterations, small uniform prior for
  connectivity). reception_pw(dialogue, side) = mean rating of that side's turns.
- Metrics (all reported):
  1. split-half reliability: Spearman between ratings fitted on two random halves of the pairs
     (seed 1), vs the same statistic for fine100 computed by splitting the 3 judges' readings
     (judge 1+2 mean vs judge 3; and the reverse) on the same 40 dialogues;
  2. position bias: share of "first" wins before de-randomization (expect ≈ 0.5; report);
  3. $/dialogue and parse-failure rate;
  4. R1(pairwise) and R1(fine100) on these 40 dialogues with bootstrap CIs — REPORTED ONLY,
     not used for the decision (n = 40, as in Amendment 6).
- Decision rule (pre-stated): run the full pairwise pass on held-out 2 (300 dialogues) only if
  split-half reliability(pairwise) ≥ reliability(fine100) AND parse failure ≤ 5% AND cost ≤ 2×
  fine100 ($0.0132/dialogue). Otherwise stop and report. Budget cap: pilot $1.5.
- If the full pass runs (Amendment 8 to be written first): judgment metric R1 with CI on
  held-out 2 and 1, compared to R1(fine100) on the same dialogues; same "lower bound > 0" rule.

### P7-B value-model refit (task32) — reporting only
- Featuresets legacy {rec, tagv, n} vs resid {tagv, resid, n}; scales v2 (fit overnight_1 → eval
  overnight_2b, overnight_2) and fine100 (fit scale6/fit_fine100 → eval full_ho2_fine100,
  full_ho1_fine100). V1, V1−C1, V2, V3 with CIs in a 2×2 table. No acceptance rule: the resid
  featureset is a reparameterization chosen for interpretability (marks = residual), not a claim
  of better fit. Fit-set fine100 re-read: 200 dialogues, cap $1.6.

### P7-C commentary marks (task33) — no test
- Marks move from z(value) to z(residual) with "!!"/"??" at |z_resid| ≥ 2. Demo-facing; no metric.

## Amendment 8 (2026-10-29 18:27 KST, written before any call) — distribution reading via structured output; Muse model ladder; audio panel (demo)

Context: Amendment 6 counts show absolute judges round to multiples of 5 (30 of 101 values used).
logprobs are unavailable on Muse Spark with reasoning. The Model API supports structured output
(`response_format: json_schema`). Judging rubric weighs "Meta tool use beyond basic integration".

### P8-A distribution reading ("dist5") — pilot then full, pre-stated rule
- Reading: the same listener persona, window and question as fine100, but the judge returns a
  probability distribution over five bins of the listener's relationship-after / face-after
  (0–20, 20–40, 40–60, 60–80, 80–100), enforced by a strict JSON schema. Point value = expected
  value over bin midpoints (10/30/50/70/90), normalized exactly like fine100 ((EV − 50)/25) so the
  two scales are comparable; spread = sd of the distribution (reported as confidence).
- Pilot: the same 40 dialogues (ids file data/private/scale6/pilot_ids.txt), n = 3 stances,
  model muse-spark-1.3-contributor, effort low. Report: distinct turn values, z ≥ +2 share, ICC(1),
  parse failures (should be 0 with the schema), $/dialogue, split-half reliability (judges 2 vs 1,
  as in task34) side by side with fine100 on the same dialogues; R1 reported only (n = 40).
- Go rule for the full run: parse failures ≤ 5% AND reliability(dist5) ≥ reliability(fine100)
  AND cost ≤ 1.5× fine100. If go: full run on held-out 2 (300) and held-out 1 (134); judgment
  metric R1 with CI; **adopt dist5 as default only if R1(dist5) lower bound > 0 on both sets and
  R1(dist5) point estimate ≥ R1(fine100)**; otherwise keep fine100 and report. Cap  total.
- Structured output is also switched on for fine100 (same prompt, integer fields in a strict
  schema) and its parse-failure/cost delta is reported; no decision attached.

### P8-B Muse model ladder — reporting only
- fine100, n = 3, same 40 pilot dialogues, models muse-spark-1.1 and muse-spark-1.2 (1.3-contributor
  = existing p2_fine100). Report per model: ICC(1), split-half reliability, distinct values, parse
  failures, latency, $/dialogue, R1 (n = 40, reported only). No model change unless a cheaper model
  matches 1.3-contributor's reliability within its CI — and even then only after a full-set
  replication (would be a new amendment). Cap .

### P8-C audio panel — demo only, no metric
- Listeners read the turn from the audio segment (muse-spark-1.2, audio input) plus the transcript
  window; outputs the same record format with modality "audio". Used on self-recorded role-play
  only (no real third parties); shown in the viewer as a second curve against the text panel.
  Agreement between the two panels is reported descriptively; nothing is decided from it.

#### P8-B note (18:31 KST, after the first runs)
muse-spark-1.1 and 1.2 run in Standard mode at 1.25/4.25 $/M tokens vs 0.10/0.20 for the
contributor variants, so the $0.5-per-model caps stopped after 5 (1.1) and 3 (1.2) dialogues
($1.05 spent; the P8-B cap of $1 is used up). Change, recorded before any further call: add
muse-spark-1.2-contributor (same price class as 1.3-contributor, ≈$0.3 for 40 dialogues) as the
like-for-like rung. Completing 1.1 and 1.2 Standard on all 40 would cost ≈$4 and ≈$7; this
needs an explicit cap raise (owner decision) and is otherwise reported as partial.

#### P8-B owner decision (18:33 KST)
- Drop muse-spark-1.1 (partial 5/40 reported as-is, no further calls). Decide on completing muse-spark-1.2 Standard only after the 1.2-contributor rung is reported.

#### P8-B result (18:49 KST) — reporting only, numbers from analysis/ladder.md, ladder11.md, rel_*.json
| model (fine100, n=3, pilot 40) | distinct turn values | ICC(1) | split-half ρ | $/dialogue | C1 [CI] (n=40, report only) | conv. ρ vs 1.3c |
|---|---|---|---|---|---|---|
| muse-spark-1.3-contributor (P2) | 102 | 0.352 | 0.687 | 0.0066 | +0.512 [+0.281, +0.689] | — |
| muse-spark-1.2-contributor | 141 | 0.343 | 0.690 | 0.0059 | +0.553 [+0.318, +0.722] | +0.908 |
| muse-spark-1.1 (partial, 5 dialogues) | 38 | 0.097 | 0.410 (n=59 turns) | 0.0775 | +0.292 [−0.558, +0.729] | +0.800 |
Reading: 1.2-contributor and 1.3-contributor are interchangeable as listeners on this pilot (reliability
within 0.01, convergent ρ 0.91); 1.1 is both less reliable and ~12× dearer per dialogue. Per the
pre-stated rule no model change follows. muse-spark-1.2 Standard: not completed (owner decision).

#### P7-B result (19:18 KST) — value model 2×2, numbers from value_compare.md files (legacy = resid to 3 decimals everywhere: reparameterization confirmed)
| fit→eval | scale | V1 [CI] | V1−C1 [CI] | V2 [CI] | V3 [CI] |
|---|---|---|---|---|---|
| overnight_1→overnight_2b | v2 | +0.418 [+0.336-+0.495] | +0.033 [-0.005-+0.070] | +0.153 [+0.061-+0.245] | +0.055 [-0.061-+0.168] |
| overnight_1→overnight_2 | v2 | +0.397 [+0.275-+0.513] | +0.037 [-0.018-+0.088] | +0.190 [+0.052-+0.321] | +0.045 [-0.119-+0.215] |
| fit_fine100→full_ho2_fine100 | fine100 | +0.466 [+0.389-+0.540] | +0.047 [+0.006-+0.090] | +0.227 [+0.136-+0.311] | +0.117 [-0.002-+0.230] |
| fit_fine100→full_ho1_fine100 | fine100 | +0.462 [+0.337-+0.577] | +0.011 [-0.057-+0.079] | +0.281 [+0.142-+0.405] | +0.127 [-0.038-+0.299] |
Reading: on the 0–100 scale the value model predicts the final outcome better than on −2..+2 (V1 +0.47 vs +0.42 on both held-out sets; V2 +0.23/+0.28 vs +0.15/+0.19), and on held-out 2 it now beats raw reception (V1−C1 lower bound > 0). Reporting only, as pre-stated; fit-set re-read cost $1.31.

#### P8-A pilot result (19:33 KST, analysis/dist5.md, rel_p4_dist5.json) and go decision
| condition (40 dialogues, n=3, 1.3-contributor) | distinct turn values | z ≥ +2 share | ICC(1) | split-half ρ | parse fail | $/dialogue | C1 [CI] (report only) | conv. ρ vs fine100 |
|---|---|---|---|---|---|---|---|---|
| fine100 (P2) | 102 | 0.030 | 0.352 | 0.687 | 0 | 0.0066 | +0.512 [+0.281, +0.689] | — |
| dist5 (P4, structured) | 339 | 0.009 | 0.331 | **0.715** | 0 | 0.0080 (1.21×) | +0.513 [+0.282, +0.694] | +0.913 |
Mean reported spread (sd of the judge's distribution) 17.8 points (range 14–21). Go rule: parse ≤ 5% ✓,
reliability ≥ fine100 ✓ (0.715 vs 0.687), cost ≤ 1.5× ✓ → **full run on held-out 2 and held-out 1
proceeds** (same settings; expected ≈ $3.5, within the $5 cap). Adoption still requires R1(dist5)
lower bound > 0 on both sets and R1 point ≥ R1(fine100), as pre-stated.

#### P8-A full-run result (22:31 KST, analysis/residual_dist5_ho2.md, residual_dist5_ho1.md, rel_full_*.json) — decision
| set | R1(dist5) [CI] | R1(fine100) | dC1 (dist5−fine100) [CI] | C6diff(dist5) [CI] | split-half dist5 / fine100 | $/dialogue dist5 |
|---|---|---|---|---|---|---|
| held-out 2 (300) | **+0.268 [+0.187, +0.340]** | +0.257 | +0.005 [−0.019, +0.028] | +0.060 [−0.003, +0.125] | 0.723 / 0.699 | 0.0079 |
| held-out 1 (134) | **+0.299 [+0.175, +0.418]** | +0.293 | +0.005 [−0.035, +0.049] | +0.090 [+0.002, +0.182] | 0.699 / 0.689 | 0.0079 |
Pre-stated adoption rule: R1(dist5) lower bound > 0 on both sets ✓ and R1(dist5) point ≥ R1(fine100) on both ✓
→ **dist5 (distribution reading via structured output) becomes the default board scale.** Honest reading: the
gain over fine100 in outcome validity is within noise (ΔR1 ≈ +0.01, dC1 CI includes 0); what dist5 adds is
resolution (339 vs 102 turn values on the pilot), slightly higher reliability on both sets, zero parse
failures, and a per-reading spread usable as confidence — at 1.2× cost. Spend P8-A: $0.32 + $2.38 + $1.06 =
$3.76 of the $5 cap. Follow-ups (owner decision, not run tonight): re-read the fit set on dist5 for
commentary/value-model norms (≈$1.6, would exceed the P8-A cap by ≈$0.4 → needs a cap raise), make dist5
the batch default (small Muse task), regenerate demo pages with confidence bands.

#### P7-A pairwise pilot result (10/30 00:05 KST, pairwise_pilot/run.json, pairwise_fit/reliability.json, analysis/residual_pairwise.md) — decision
| metric | pairwise (40 dialogues, k=4, 3 stances) | fine100 (same 40) | rule |
|---|---|---|---|
| pairs / votes | 1,867 / 5,601 (ties 8) | — | — |
| parse failures | 0 | 0 | ≤ 5% ✓ |
| position bias (share "first" wins) | 0.504 | — | ≈ 0.5 ✓ (reported) |
| split-half reliability ρ | **0.763** (462 turns) | 0.687 | ≥ fine100 ✓ |
| $/dialogue | **0.0239** | 0.0066 | ≤ 2× (0.0132) **✗ (3.6×)** |
| R1 (n = 40, report only) | +0.206 [−0.034, +0.431] | +0.274 [+0.040, +0.482] | reported only |
| dC1 pairwise − fine100 | −0.068 [−0.171, +0.034] | — | reported only |
Pre-stated go rule needs all three of parse ✓, reliability ✓, cost ✓ → **cost fails → no full run; stop and report.**
Reading: pairwise judging is the most consistent reader we have tried (0.76 vs 0.69–0.72) at ~3.6× the price,
and on 40 dialogues it is not better than absolute scoring at predicting the outcome (dC1 CI straddles 0,
n = 40 as pre-stated). A cheaper variant (k = 2, two stances) would need a new amendment before any call.
Spend: $0.96 of the $1.5 cap.

## Amendment 9 (2026-10-30 08:05 KST, written before any call) — pairwise full run on held-out 2 as a validity test (owner cap raise)
Owner decision (10/30 00:10): the P7-A cost rule stopped the automatic full run; the owner chose to fund one
full pass anyway, as a validity test, because pairwise reading was the most reliable reader in the pilot
(split-half 0.763 vs dist5 0.715). Same settings as the pilot (k = 4, three stances, muse-spark-1.3-contributor,
effort low, seed 0), source held-out 2 (300 dialogues, overnight_2b records for turns/moves). Cap $8.
- Judgment metric: R1(pairwise) on held-out 2 with bootstrap CI (same-scale partial Spearman on BT ratings,
  table from fit_fine100), compared with R1(dist5) = +0.268 [+0.187, +0.340] and R1(fine100) = +0.257 on the
  same 300 dialogues. Also report split-half reliability on 300, position bias, $/dialogue.
- Rule: pairwise becomes the default reader only if R1(pairwise) lower bound > 0 AND point ≥ R1(dist5) AND
  split-half ≥ dist5's 0.723. Otherwise dist5 stays and the pairwise result is reported as "most consistent
  reader, not more valid, 3–4× cost". No held-out 1 run unless held-out 2 passes (would be a further amendment).

#### P9-A pairwise full-run result (10/30 17:27 KST, pairwise_ho2/run.json, pairwise_ho2_fit/reliability.json, analysis/residual_pairwise_ho2.md) — decision
| metric | pairwise (held-out 2, 300 dialogues, k=4, 3 stances) | dist5 (same 300) | rule |
|---|---|---|---|
| pairs / votes | 13,871 / 41,613 | — | — |
| parse failures | 0 | 0 | — |
| position bias (share "first") | 0.510 | — | ≈ 0.5 ✓ |
| split-half reliability ρ | **0.789** (3,409 turns) | 0.723 | ≥ 0.723 ✓ |
| R1 (n = 600 rows) | **+0.344 [+0.267, +0.414]** | +0.268 [+0.187, +0.340] | lower bound > 0 ✓, point ≥ 0.268 ✓ |
| dC1 pairwise − dist5 | +0.048 [+0.008, +0.087] | — | reported (CI excludes 0) |
| C1 / R4 | +0.472 / +0.462 | — | reported |
| $/dialogue | **0.0244** ($7.31 total, 20.2M prompt + 26.4M completion tokens) | 0.0079 | 3.1× (reported) |
All three pre-stated conditions hold → **pairwise becomes the default reader** per Amendment 9. Caveats: single
held-out set (held-out 1 not run — a further amendment); cost is 3.1× dist5; the fit table (fit_fine100) is the
fine100 value model, so R1 here is "pairwise ratings read through the fine100 table", same as the dist5 row.

## Amendment 10 (2026-10-30 19:50 KST, written before any call) — pairwise confirmation on held-out 1
Held-out 2 passed all three Amendment 9 conditions (R1 +0.344 [+0.267, +0.414] vs dist5 +0.268; split-half 0.789 vs 0.723;
parse 0; position bias 0.510). Before pairwise replaces dist5 in the pipeline, run the same reader once on held-out 1
(134 dialogues, overnight_1 records; k = 4, three stances, muse-spark-1.3-contributor, effort low, seed 0). Cap $4.
- Report: R1(pairwise) with bootstrap CI vs R1(dist5) = +0.299 [+0.18, +0.42] on the same set, split-half, position bias, $/dialogue.
- Rule: confirmation holds if R1(pairwise) lower bound > 0 AND point ≥ R1(dist5) point on held-out 1. If it holds, pairwise is
  the default reader for both held-out sets and the demo pages are regenerated on it (task43). If it fails, held-out 2 result
  stands as "most valid on one set", dist5 remains the default, and no further pairwise spend without a new amendment.
