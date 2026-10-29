# SPEC task36 — audio panel: listeners read the turn from the recording (Amendment 8, P8-C, demo)

New module `forecaster/audiopanel.py`, CLI `uptake-audio-panel`. Stdlib + `ffmpeg` via subprocess.

## input
- `--audio FILE` (wav/mp3/mp4) and `--turns FILE` (jsonl from the L0 transcribe step). Each turn
  needs `start` and `end` in seconds. If the transcribe output lacks them, extend
  `forecaster/transcribe.py` (task07) to keep the turn-level timestamps that
  `muse-voice-transcribe-1.0` returns (`--timestamps` on by default) and make `uptake-audio-panel`
  exit with a message pointing to that when they are missing.

## reading
- For turn i: cut `[max(0, start_i − pre), end_i]` (`--pre 20` s default) with ffmpeg to 16 kHz mono
  wav; send inline as `input_audio` (base64, format wav) together with the text window (same
  builder as the board, judged turn marked) and the same three listener personas. Model default
  `muse-spark-1.2` (docs: 1.3 audio quality may drop). Scale `fine100` or `dist5` (task35), always
  structured. One call per stance, `n` readings.
- Record format = the board record (`effects` with the same keys) plus top level
  `"modality": "audio"`, `"scale"`, `"model"`, `"audio_seconds"` (sum of clip lengths). Calls logged
  with `meta.layer = "audio"`, clip seconds in `meta`. Honour `--budget` (count tokens as usual and
  add audio seconds to the run summary; price is per hour of audio).

## compare and show
- `--text-record FILE`: write `compare.json` — per turn (index, text value, audio value, Δ),
  Spearman over turns, mean |Δ|, and the three largest Δ turns.
- viewer: `--audio-record FILE` overlays the audio-panel curve (dashed) on the position chart,
  hover shows text/audio values and Δ, legend labels "text panel" / "audio panel (Muse Spark 1.2,
  hears the clip)".

## tests `tests/test_task36.py` (Muse writes; no API, no real ffmpeg: stub `subprocess.run`)
- clip boundaries (pre-roll clamp at 0); missing timestamps → clear exit; base64 payload shape;
  record keys match the board record; compare.json math on toy records; viewer renders the second
  curve and legend.
- `pytest -q` must stay green.

- Multi-party meetings (AMI has four speakers A–D): the listener for each turn is chosen exactly as the board does for multi-speaker turn files (`_addressed_to` / next speaker); no new rule.
