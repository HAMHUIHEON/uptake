# Uptake

Reads important conversations turn by turn, showing where what each speaker wanted in outcome, relationship, face, and understanding diverged from what actually happened, and what remains against each speaker's own goals.

The name comes from J. L. Austin's speech-act term: *uptake* is how an
utterance is actually taken up by the listener. Uptake is an analysis
tool for high-stakes conversations — diplomacy, air traffic
control, safety, negotiation — that shows where a misdelivery arose,
what was wanted versus what happened on each stake, and what could
have been said differently.
It reads the utterance, not the person.

Each turn is read as wanted vs happened: what the speaker wanted in
context on each of four stakes — outcome, relationship, face,
understanding — and what the later turns show actually happened on
that stake, judged after the reply is seen. The same turn may gain on
one stake and lose on another. A miscommunication is when what was
wanted did not happen, whether intended or not. There is no outside
right answer; each speaker is judged against their own goal in that
conversation, never scored as a person.

## Example

In Avianca 052 (avianca052), the crew's "we will run out of fuel" was
read as routine and the controller's plan did not change — an
urgent-read-as-routine moment. Uptake flags such moments from the
utterance and the turns around it, read after the reply is seen, and
shows what the crew wanted, what happened instead, and what
rephrasing could have been tried. The same report shape covers
success and recovery cases, not only tragedies.

## What it does

- **Show where misdelivery arose.** For each turn: how the speaker
  steered the conversation, how each listener received it, what was
  wanted vs what happened on each stake, and whether the gap looks
  intentional.
- **Show what was wanted on each stake.** Alerts come only from stake
  gaps of missed, partly, or cost — one per turn and stake, each with
  the reply-turn quote it was judged from. Relationship movement alone
  never raises an alert.
- **Show what could have been said differently.** Alternatives for the
  decisive turn are compared against the original on the same
  simulated listener panel, paired, with confidence intervals and
  negative controls (identical-vs-identical near 0.5, a deliberately
  worse rewrite below 0.5).
- **Track goals, not personalities.** Each speaker's goal in that
  conversation, when their strategy shifted, and the plus/minus left
  against their own goal when the talk ended. No per-person scores.

## How it works

- **L0 input.** Plain text, or audio transcribed with speaker diarization
  via Muse Voice Transcribe (`muse-voice-transcribe-1.0`).
- **L1 move record.** Every turn is tagged with move types (e.g.
  proposing, conceding, probing) using `muse-spark-1.3-contributor`.
- **L3 simulated listener panel.** A panel of simulated listeners reads
  each turn from situational stances — roles and stakes in the situation,
  not personality types (`muse-spark-1.3-contributor`).
- **L4 wanted-vs-happened gap.** Each turn is read on four stakes —
  outcome, relationship, face, understanding — with wanted, happened,
  gap (met, partly, missed, cost, not_relevant, unclear), and the
  reply-turn quote as evidence, judged after the reply is seen.
  A general urgent-read-as-routine rule describes the decisive read;
  relationship movement alone never raises an alert.
- **L5 report.** Goals and shifts, turning points (how each turn was
  steered and how it was received), the alert timeline, the
  intended-said-read triple, alternatives, a one-line perspective, the
  end balance, and self-reliability (`muse-spark-1.3-contributor`
  reads; `muse-spark-1.3` judges whether the core meaning survived).

## Measured so far

- **Move record**, on CaSiNo negotiation dialogues (CC BY 4.0), held-out
  30 dialogues: micro F1 0.624 [0.569–0.677], macro F1 0.544
  [0.481–0.596]; majority baseline 0.289.
- **Reliability** (pre-registered, measured on the method study, not on
  any single case): split-half reliability 0.863 (k=8); outcome
  R1 +0.277 on held-out set 1, +0.344 on held-out set 2;
  perturbation direction 16/20 (0.8). Per-case reports reuse these
  values and say so in their limits.
- **Control messages** (messages that should communicate fine): false-alarm
  rate measured and being reduced (first run 2/6). Work in progress.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export META_API_KEY
```

Build one case report (cached; prints `SPEND=<usd>` last, stops past
`--budget`):

```sh
.venv/bin/uptake-report --case avianca052 --out data/private/goal_run/avianca052 --budget 1.0
```

Read only the first N eligible turns (small trial):

```sh
.venv/bin/uptake-report --case avianca052 --out /tmp/uptake_trial --budget 1.0 --max-turns 5
```

Transcribe audio, tag moves, and build the analysis:

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

- This tool reads where an utterance did not land, after the reply is
  seen; it does not claim to know the right answer, and it makes no
  claim about what will happen next.
- After utterances read as not landing, re-requests or divergence are
  reported as more common only when a measured value exists for that
  claim; otherwise the report says the value was not measured.
- Simulated listeners are not real people. Treat panel reads as hints, not
  verdicts.
- Numbers come with confidence intervals and small samples. Read the CI,
  not just the point estimate.
- Not a tool for judging individuals' personality or ability.
  It reads the utterance, not the person.
- Analysis of recorded conversations only, not advice. Record only
  conversations you take part in, and get consent first.

## Background

Built for the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026).
