# SPEC task59 — align a case transcript to its audio and read it with the audio panel

Goal: for cases with real public audio (usair1549 first), compare the audio reading with the text reading on the same turns.
New entry `uptake-align --audio WAV --record JSON --out JSON [--model M]`:
1. Run the existing transcriber (`forecaster/transcribe.py`, same model/flags as task07/36) on the WAV to get segments with
   start/end seconds and text.
2. Align each record turn (by `index`) to the transcript: greedy monotone alignment maximizing token-overlap similarity
   (lowercased, digits kept, callsign tokens kept); accept a match if similarity ≥ 0.35, else mark the turn `unaligned`.
   Output: the record with `effects[i].start/end` filled (seconds in the audio) and an alignment report
   {aligned, unaligned, mean_similarity, offset_hint}.
3. The aligned record must be accepted by `uptake-audio-panel` (task36) unchanged.
tests (`tests/test_task59.py`): alignment on a toy transcript (monotone, threshold), unaligned marking, report shape;
transcriber mocked. `pytest -q` green.
