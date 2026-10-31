# SPEC task53 — "다시 말해 보기" (layer 3 main): rewrite a decisive turn, the reader compares it with the original

Why: guessing the reader's verdict feels like a test; rewriting a blunder and getting a live comparison is play, and it
uses the reader's measured sensitivity (Amendment 11) directly. Stdlib only, two pieces:

## 1. `uptake-play-server --records DIR --fit DIR --commentary-dir DIR --port 8765 --model M --budget B`
`forecaster/play_server.py`: `http.server` on 127.0.0.1 only. Routes:
- `GET /` → the play page (below) for a dialogue chosen by `?id=casino-878`; `GET /api/dialogue?id=` → turns + decisive turn
  indices + standings (from the commentary JSON).
- `POST /api/compare` {id, turn, text} → runs the pairwise judge (reuse `pairwise.py`/`perturb.py` vote code: the shared
  context up to that turn, candidates A = original, B = player's text, both orders × 3 stances = 6 votes) and returns
  {p_better: votes_for_player/6, votes: [...], margin, cost_usd, model}. Reject texts > 400 chars or empty. Per-process
  budget cap; when exceeded return 402 with a message the page shows.
- Log every comparison to `--log FILE` (jsonl: ts, id, turn, text, votes, cost). Never log keys.

## 2. Play page (served; also writable as a static file by `uptake-play --item ... --out` with the API URL embedded)
Mobile-first, light theme, Korean. Flow:
1. Shows the dialogue up to the decisive turn (context turns collapsed, last 3 visible), then the original turn marked
   "이 말에서 B의 관계 점수가 50 → 8로 떨어졌다".
2. Text box: "같은 뜻을 당신이라면 어떻게 말할까요?" + 버튼 "자에게 물어보기". While waiting: "읽는 중… (약 5초)".
3. Result card: "당신의 말이 원문보다 잘 받아들여질 확률 {p}%" with the 6 votes as dots (stance labels 중립/우호/의심), one line
   "원문보다 {부드럽다/비슷하다/거칠다}" from the margin, and a "다시 써 보기" button. Keep a list of the player's attempts with
   their p (best on top). Share text: "나는 casino-878의 1수를 {p}%짜리 말로 고쳤다 — uptake".
4. Footer: "모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다 · 한 번 묻는 비용 약 {cost}".
5. Up to 3 decisive turns per dialogue, selectable as tabs.

tests (`tests/test_task53.py`): server routes with the API mocked (dialogue payload shape; compare returns p_better in
[0,1] and 6 votes; empty/too-long text → 400; budget exceeded → 402); page HTML contains the prompt, the result template and
the share template; static export embeds the API URL. `pytest -q` stays green. No network in tests.
