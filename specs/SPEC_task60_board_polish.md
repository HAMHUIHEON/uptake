# SPEC task60 — board2 polish after the first 8 public-record cases (10/31 morning)

Owner context: the overnight run built boards for 8 cases (aviation ATC ×4, diplomacy, medical, court debate, drama) with
`--case`, `--panel` (task57 misread), alerts (task58) and scenes. Reviewing them exposed six fixes. No new features.

1. **Misread = conflict, not "outside the invited set".** `board.ACTION_TO_MOVES` was written for negotiation chat; on
   ATC/medical/diplomacy records it marks ordinary uptake as misread (1549, a success case, reads 17/22 어긋남): an
   instruction labelled `other` invites only withdraw/ignore, `inform` invites only proceed, `ask` invites only
   ask_clarification — so "the listener complied" or "the listener answered" counts as a misreading. Change
   `misread.reading_for_turn`:
   - invited sets widen: `inform` → {proceed, comply, ask_clarification}; `ask`/`probe` → {ask_clarification, comply, proceed}
     (an answer is uptake); `other`/`smalltalk` → every action (no misread possible); `propose` → {negotiate, comply, refuse,
     proceed}; `reject` → {refuse, negotiate, ask_clarification, withdraw}; the rest unchanged.
   - `misread` is true only when the majority reading is in the **conflict set** of the move: `CONFLICTS = {accept/concede:
     {refuse, escalate, retaliate, ignore}, propose/counter/appeal_*: {ignore, retaliate, escalate}, ask/probe: {ignore,
     refuse, escalate}, inform: {refuse, ignore, retaliate, escalate}, pressure/challenge: {ignore, withdraw},
     repair/empathize/praise: {retaliate, escalate, refuse, ignore}, reject: {comply, ignore}}`; severity = share of readings
     in the conflict set (0 when the move has none).
   - Keep the row fields (`invited`, `majority`, `share`), add `conflict: [...]`.
   Expected effect: 1549 drops to a few 어긋남 (the callsign confusion turns), while Avianca's "running out of fuel" →
   read as proceed/ignore and Lear's "Nothing" → read as refuse stay flagged.
2. **Multi-speaker boards.** board2 currently shows two score boxes and a two-colour tide (first two speakers). When a
   dialogue has ≥ 3 speakers: one score box per speaker (sorted by turn count, max 6), the tide keeps the two most active
   speakers but the legend names them, and a small "다른 화자" line lists the rest with their final standing. Lear 1.1 has
   LEAR, GONERIL, REGAN, CORDELIA, KENT; the 1549 record has A/B/C.
3. **First-alert card.** The first turn at 경계 or above (alerts.first_alert) is always rendered as the first "결정적 순간"
   card, labelled "첫 경보", even when its |Δ| is not among the top-3.
4. **Hide "결과 없음".** The score box line that prints the CaSiNo outcome label is omitted when the record carries no
   outcome (case records). Same for the verdict's outcome clause.
5. **Case verdict.** With `--case`, the "한 문장" line is the case `outcome` text (first sentence), and the generated
   standings sentence moves under it in smaller type. Without `--case`, unchanged.
6. **Defaults.** `uptake-pairwise run` default `--k 8` (Amendment 14 decision; `--k 4` stays available).

tests (`tests/test_task60.py`): severity 0 when the move has no conflict set, 1549-style comply/answer not misread, Lear "Nothing" (reject→refuse) still misread; ≥3-speaker score boxes and legend; first-alert card
present; no "결과 없음" when outcome is None; case verdict uses outcome; pairwise default k. `pytest -q` green, including
every earlier test file (task57/58 fixtures must still pass with the new conflict rule — adjust fixtures only if they
relied on the artifact).
