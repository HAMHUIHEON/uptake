# SPEC task43 — pairwise ratings as the default reader (commentary + viewer v2)

Owner decision (2026-10-30 evening): Amendment 9 passed on held-out 2 (R1 +0.344 vs dist5 +0.268, split-half 0.789),
so pairwise Bradley–Terry ratings become the default reader. Amendment 10 (held-out 1) runs overnight; this task makes
the pipeline able to read pairwise records end to end so the demo pages can be regenerated the moment it confirms.

## 1. Scale handling in `commentary.py`
- A record with top-level `scale == "pairwise"` carries per-effect `value` on the BT log scale (centred 0, typical
  sd ≈ 0.6) instead of 0–100. `norm` (mean/sd by move kind) and `z_resid` must be computed on that scale, from a fit
  table built on pairwise records. Add `--fit` acceptance of a directory whose `table.json` says `"scale": "pairwise"`;
  refuse (clear error) when a 0–100 table is paired with a pairwise record or vice versa, unless `--same-scale` is passed
  (then rank-based: convert both to within-record ranks before computing residuals — document this in the output JSON as
  `"scale_note"`).
- Standing (잔고) shown in commentary lines: for pairwise records use `50 + 25·(value − mean)/sd` clipped to 0–100 so the
  wording ("잔고 53 → 45") stays on a 0–100 feel; put `"standing_scale": "pairwise→0-100 via z"` in the JSON.
- Mark thresholds (`!!` / `??` at |z_resid| ≥ 2, `!`/`?` at ≥ 1) unchanged.

## 2. Fit table for pairwise
- `uptake-value fit` (valuemodel) gains `--scale pairwise`: reads pairwise records, writes `table.json` with
  `"scale": "pairwise"`, mean/sd by move kind, and the value-model coefficients. Input for now: the held-out 2 pairwise
  records (`data/private/scale6/pairwise_ho2_fit/records`) — the owner will run it; the task only needs the code path and
  a fixture-based test.

## 3. Viewer v2
- `viewer2.py` reads `scale` from the record: pairwise → the tide uses z_resid exactly as now (already scale-free), the
  side standings use the 0–100 conversion above, hover shows the raw rating with 2 decimals and the label "쌍대 평점".
- Add a one-line badge under the title: "읽기 척도: 쌍대 비교(BT 평점)" or "읽기 척도: 0–100" so demo viewers know which
  reader produced the page.

## 4. Batch default
- `uptake-batch` keeps dist5 as its reading mode (pairwise is a separate run via `uptake-pairwise run`); only add a
  `--reader pairwise` flag that errors with a pointer to `uptake-pairwise` so nobody silently mixes scales.

## tests `tests/test_task43.py`
- commentary on a toy pairwise record + pairwise table: residuals computed, standings in 0–100, scale_note absent;
  mismatched scales → error; `--same-scale` → rank path with scale_note.
- valuemodel fit with `--scale pairwise` on fixtures writes table.json with `"scale": "pairwise"`.
- viewer2 on a pairwise record: HTML contains the scale badge text and "쌍대 평점" hover label; 0–100 record still shows
  the old badge.
- `pytest -q` stays green.
