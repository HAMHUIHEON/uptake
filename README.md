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

- **Show wanted vs actual on items.** Each turn gets wanted vs actual
  values on items. Each item carries a measured or reference grade.
- **Show alerts and standing from measured items only.** Alerts and
  standing use measured items only.
- **Show a move mark per turn.** A move mark per turn is computed from
  measured items. It is left blank when no value exists.
- **Show stakes with sources.** Recorded stakes are shown with their
  source. Otherwise the weight is a model estimate marked as such.
- **Show a full-conversation judgment.** A full-conversation model
  judgment picks the decisive turn and scenes. It is labeled reference,
  not measurement.
- **Show what could have been said differently.** Alternatives are
  compared against the original on one criterion, paired, with an
  identical control, a worse control and a middle control.
- **Track goals, not personalities.** Each speaker's goal in that
  conversation, when their strategy shifted, and what remains against
  their own goal. No per-person scores.

## How it works

- **Input.** Plain text, or audio transcribed with speaker diarization
  via Muse Voice Transcribe (`muse-voice-transcribe-1.0`).
- **Move record.** Every turn is tagged with move types (e.g.
  proposing, conceding, probing) using `muse-spark-1.3-contributor`.
- **Per-turn item reading.** Each turn gets wanted vs actual values on
  items, read after the reply is seen, using
  `muse-spark-1.3-contributor`. Each item carries a measured or
  reference grade.
- **Alerts, standing, and marks.** Alerts and standing use measured
  items only. A move mark per turn is computed from measured items.
  It is left blank when no value exists.
- **Stakes.** Recorded stakes are shown with their source. Otherwise
  the weight is a model estimate marked as such.
- **Judgment.** A full-conversation model judgment picks the decisive
  turn and scenes, using `muse-spark-1.3`. It is labeled reference,
  not measurement.
- **Alternatives.** Alternatives are compared against the original on
  one criterion, paired, with an identical control, a worse control
  and a middle control.
- **Report.** Goals and shifts, turning points, the alert timeline,
  alternatives, the end balance, and reliability notes. Each claim
  says where its value came from.

## Measured so far

- **Move record**, on CaSiNo negotiation dialogues (CC BY 4.0), held-out
  30 dialogues: micro F1 0.624 [0.569–0.677], macro F1 0.544
  [0.481–0.596]; majority baseline 0.289.
- **Item grades.** Item grades come from repeat reading and an
  external-answer comparison. The commitment item stays reference
  after a retest.
- **Alternatives.** Alternatives separate better from worse direction
  but do not yet separate middle from good wording.
- **Not measured yet.** Whether end balances match self-reports,
  whether quiet conversations stay quiet, and held-out cases have not
  been measured yet.

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
