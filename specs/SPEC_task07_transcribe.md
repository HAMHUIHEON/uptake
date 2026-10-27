# SPEC — task07: audio → speaker-labelled turns (Muse Voice Transcribe)

API facts (from dev.meta.ai docs): `POST https://api.meta.ai/v1/asr/transcribe`, `multipart/form-data` with parts `request` (JSON: `{"model":"muse-voice-transcribe-1.0","audioEncoding":"WAV","mode":"DIARIZATION"}`, content-type application/json) and `audio` (WAV). Header `Accept: text/plain` returns one turn per line like `A:Hello there` (unattributed: `unknown:...`). Limits: WAV only, ≤ 10 minutes, ≤ 32 MB body. 429 → retry with exponential backoff.

## `forecaster/transcribe.py`
- `@dataclass Turn(speaker: str, text: str, chunk: int = 0)`
- `parse_diarized_text(text: str, chunk: int = 0) -> list[Turn]`: one line per turn; split at the FIRST colon; strip whitespace; skip blank lines and lines without a colon; merge consecutive lines from the same speaker (join with a single space).
- `ffmpeg_cmd(src, dst, start_s: int, dur_s: int) -> list[str]`: returns exactly
  `["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start_s), "-t", str(dur_s), "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]`
- `audio_duration(src) -> float` via ffprobe.
- `to_wav_chunks(src, out_dir, chunk_s=540) -> list[Path]`: run ffmpeg for each 540 s window; names `chunk_000.wav`, ...
- `transcribe_wav(path, post=None, max_retries=4) -> str`: performs the HTTP call (httpx client with `truststore.SSLContext`, same key env vars as MuseClient). `post` is an injectable callable `(url, headers, files) -> (status_code, text)` for tests; default uses httpx. Retry 429 and 5xx with exponential backoff (1, 2, 4, 8 s); raise RuntimeError after retries. Never log the key.
- `transcribe_file(src, work_dir, post=None) -> list[Turn]`: chunk, transcribe each, parse with the chunk index. Speakers are only consistent **within** a chunk; keep labels as returned and keep `chunk` so later steps can reconcile.
- `save_turns(path, turns)` / `load_turns(path)` JSONL.

## CLI
`uptake-transcribe AUDIO --out turns.jsonl [--work DIR]` — prints number of turns and speakers per chunk. Register entry point.

## Privacy
Audio, chunks and transcripts live only under `data/private/` (gitignored). Add `data/` to .gitignore.
