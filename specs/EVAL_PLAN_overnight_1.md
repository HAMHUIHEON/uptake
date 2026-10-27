# Pre-registered evaluation plan — overnight run 1 (written before the run, 2026-10-28 00:10 KST)

Purpose: decide in advance what each number means, what counts as a problem, and where to dig — so the morning is analysis, not archaeology.

## Data
- negotiation: CaSiNo, 200 dialogues, seed 2, excluding the 60 dialogues already used (dev 30 + held-out 30). Public → contributor model.
- youtube: all locally transcribed clips (9-min segments; genres: comedy podcast, long interview, news panel, talk show, debate, haggling). Private, internal only → contributor model by owner's decision (cost); never committed.
- Settings: panel n=3, k=1, effort low, inner workers 8, conversation workers 4, budget cap $15.

## Questions, metrics, thresholds (fixed now)
| # | Question | Metric | "OK" | "Problem" → first drill-down |
|---|---|---|---|---|
| Q1 | Move record still accurate on new negotiation data? | CaSiNo micro F1 (projected), CI over dialogues | CI overlaps held-out 0.624 [0.569–0.677] | CI entirely below → per-label confusion + 20 random wrong turns with raw output |
| Q2 | Does the move set cover each genre? | other_rate, failed_rate per genre | other ≤ 0.20, failed ≤ 0.05 | read 20 'other' turns per genre; list missing move types |
| Q3a | Is move tagging stable run-to-run? | kappa on first move (repeat sample 10) | ≥ 0.60 | per-label disagreement table |
| Q3b | Is the position curve stable run-to-run? | Spearman between curves (repeat sample 10) | ≥ 0.70 | variance split: between panel members vs between runs, per turn; test n=6 on 3 conversations |
| Q4 | Do next-turn forecasts beat climatology? | skill_mean (nats) with CI, per genre | CI lower bound > 0 | predicted-vs-actual class confusion; share of 'proceed'; hits by turn position |
| Q5 | Cost and speed | $ per conversation, calls/min, retries, latency p50/p95 | ≤ $0.05/conv, no budget stop | retries by status code; slowest calls |

Smoke run (3 conversations, before this plan): kappa 1.00, curve Spearman 0.17 (1 repeat), $0.065, 7 min — Q3b is the expected weak spot.

## Health check (stop rule)
After the first 20 conversations: if failed_rate > 0.20 or any error rate > 0.10 → stop and report; do not burn the night.

## What must be logged per call (to answer the drill-downs)
run id, git commit, layer (tag | board | judge), item id, turn index, listener, panel member/stance, model, effort, prompt hash, raw response text, parsed ok, prompt/completion tokens, latency, retries and status codes.

## Morning deliverable
`uptake-analyze` produces report.md: the Q1–Q5 table with OK/Problem verdicts computed from the thresholds above, each Problem followed by its drill-down output, plus 30 random raw traces per genre for reading.
