# Uptake

Read the board of any conversation.

The name comes from J. L. Austin's speech-act term: *uptake* is how an
utterance is actually taken up by the listener. Uptake forecasts that
moment — how your message will land — before you send it.

## What it does

- **Forecast how a message will land before you send it.** Describe the
  situation and your draft, and get back where it is likely to be
  understood, and where it is likely to be misread.
- **Replay a conversation turn by turn, like a Go/chess broadcast.**
  A position curve tracks how the conversation is going, moves carry
  annotations (`!!`, `!`, `=`, `?!`, `?`, `??`), and turning points mark
  where things shifted.
- **Test the next move.** Compare a rewrite against the original on the same
  simulated listener panel, with a paired confidence interval (multiple
  candidate replies: in progress).

## How it works

- **L0 input.** Plain text, or audio transcribed with speaker diarization
  via Muse Voice Transcribe (`muse-voice-transcribe-1.0`).
- **L1 move record.** Every turn is tagged with one of 20+ move types
  (e.g. proposing, conceding, probing) using `muse-spark-1.3-contributor`.
- **L3 simulated listener panel.** A panel of simulated listeners reads
  each turn from situational stances — roles and stakes in the situation,
  not personality types (`muse-spark-1.3-contributor`).
- **L4 effect gap by stake.** The gap between intended and received meaning
  is scored on four stakes — outcome, relationship, face, understanding —
  broken into 13 sub-stakes, with cluster-bootstrap 95% CIs.
- **L5 board.** A position curve over turns, annotated moves, turning
  points, and a forecast check: the actual next turn scores the forecast
  (`muse-spark-1.3-contributor` reads, `muse-spark-1.3` judges whether the
  core meaning survived).

## Measured so far

- **Move record**, on CaSiNo negotiation dialogues (CC BY 4.0), held-out
  30 dialogues: micro F1 0.624 [0.569–0.677], macro F1 0.544
  [0.481–0.596]; majority baseline 0.289.
- **Control messages** (messages that should communicate fine): false-alarm
  rate measured and being reduced (first run 2/6). Work in progress.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export META_API_KEY
```

Forecast a bundled example case:

```sh
forecast --case examples/v2/deadline_slack.json --v2
```

Transcribe audio, tag moves, and build the board:

```sh
uptake-transcribe --out turns.json meeting.wav
uptake-moves --turns turns.json --out moves.json
uptake-board --turns turns.json --moves moves.json --out board.json
```

Evaluate the move tagger on held-out dialogues:

```sh
uptake-eval-moves --casino <casino-dir> --n 30 --out eval.json
```

## Honest limits

- Simulated listeners are not real people. Treat forecasts as hints, not
  verdicts.
- Numbers come with confidence intervals and small samples. Read the CI,
  not just the point estimate.
- Not a tool for judging individuals' personality. It reads situations,
  not people.
- Record only conversations you take part in, and get consent first.

## Background

Built for the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026).
Code written with Muse Code.
