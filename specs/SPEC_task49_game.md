# SPEC task49 — layer 3 "다음 수 맞히기" game page

Standalone page: `uptake-game --item RECORD --commentary JSON --out HTML [--names JSON]`. New module
`forecaster/game.py`. Self-contained HTML, light theme, mobile-first (max-width 480px), Korean UI, stdlib only. The tide is
hidden until the end.

Flow:
1. Intro card: "두 사람이 캠핑 물품을 나누는 협상입니다. 발언을 하나씩 보고, 상대가 이 말을 어떻게 받아들일지 맞혀 보세요."
   Buttons: 시작.
2. For each turn in order: show speaker + text; the player taps one of three: 좋게 받아들여짐 / 보통 / 나쁘게. Then reveal:
   the reader's verdict (from z_resid: ≥1 좋게, ≤−1 나쁘게, else 보통), the relationship score change for the listener
   ("B의 관계 점수 50 → 8"), and the one-line reason from the commentary. Score +1 on a match. Running tally "나 3 / 해설자 5"
   (the commentator is always right by definition — say so once in the intro).
3. End card: player's hit rate, the full simplified tide (same as task48 section 3) now revealed with the player's
   guesses as small marks (✓/✗) on each turn, the actual outcome line, and a share text box the player can copy:
   "나는 협상의 흐름을 15수 중 9수 맞혔다 — uptake".
4. Everything in page JS; no server; `localStorage` optional for last score (wrapped in try/catch).

tests (`tests/test_task49.py`): HTML contains intro, three buttons, per-turn data table with the reader's verdict and
score change, end-card share text template with the turn count; verdict thresholds (z_resid 1.2 → 좋게, −0.3 → 보통,
−1.5 → 나쁘게); `pytest -q` stays green.
