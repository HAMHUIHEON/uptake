"""L1 game record: classify each turn into moves."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .reader import DEFAULT_MODEL
from .usage import BudgetExceeded

MOVES = ["smalltalk", "inform", "ask", "propose", "counter", "concede", "accept", "reject",
         "probe", "appeal_need", "appeal_fairness", "empathize", "cooperate", "pressure",
         "praise", "challenge", "repair", "deflect", "joke", "reveal_preference", "other"]

CASINO_MAP = {"small-talk": "smalltalk", "self-need": "appeal_need", "other-need": "appeal_need",
              "no-need": "reveal_preference", "elicit-pref": "ask", "uv-part": "challenge",
              "vouch-fair": "appeal_fairness", "showing-empathy": "empathize",
              "promote-coordination": "cooperate", "non-strategic": "other"}

MOVE_EXAMPLES: dict[str, list[str]] = {
    "cooperate": ["Let's find a split that works for both of us"],
    "appeal_fairness": ["It's only fair we each get two"],
    "ask": ["What matters most to you?"],
    "concede": ["Okay, you can have the firewood"],
    "reveal_preference": ["We don't really need water"],
    "propose": ["I'll take 2 food and you take 3 water"],
}

MOVE_DEFS = {
    "smalltalk": "greeting or off-task chat that builds rapport",
    "inform": "states facts or one's own situation plainly",
    "ask": "asks for the other's preferences, needs or limits",
    "propose": "puts a concrete offer or deal on the table",
    "counter": "responds to an offer with a modified offer",
    "concede": "gives up something wanted or drops a claim",
    "accept": "agrees to the proposed deal",
    "reject": "refuses the proposed deal",
    "probe": "tests the other side's reaction without committing",
    "appeal_need": "justifies a claim by one's own (or one's group's) needs",
    "appeal_fairness": "justifies a claim by fairness or equal treatment",
    "empathize": "acknowledges the other's feelings or situation",
    "cooperate": "frames the exchange as joint problem-solving",
    "pressure": "deadline, threat or ultimatum",
    "praise": "compliments or vouches for the other side",
    "challenge": "questions or pushes back on the other's claim",
    "repair": "apologizes or fixes a misunderstanding",
    "deflect": "avoids the question or changes topic",
    "joke": "humor meant to lighten the mood",
    "reveal_preference": "reveals what one values or does not need (e.g. 'we have plenty of water'), signalling room to trade",
    "other": "anything that fits none of the above",
}


@dataclass
class MoveTag:
    turn_index: int
    speaker: str
    moves: list[str] = field(default_factory=list)
    addressed_to: str = ""
    quote: str = ""
    failed: bool = False


def build_move_messages(turns, i, window=4) -> list[dict]:
    lines = ["You classify negotiation turns into moves.", "Moves:"]
    for m in MOVES:
        line = f"- {m}: {MOVE_DEFS[m]}"
        examples = MOVE_EXAMPLES.get(m)
        if examples:
            line += f' (e.g. "{examples[0]}")'
        lines.append(line)
    lines.append("Label ask only when the speaker actually seeks the other's preferences or information, not for rhetorical questions.")
    lines.append("One utterance can carry a deal move AND a strategy (e.g. propose + appeal_need).")
    system = "\n".join(lines) + (
        "\nReply with JSON only (no other text): "
        '{"moves": [1-3 labels, most important first], '
        '"addressed_to": "<speaker label or \'all\'>", '
        '"quote": "<short exact phrase from the target turn>"}'
    )
    start = max(0, i - window)
    ctx = [f"{turns[j].speaker}: {turns[j].text}" for j in range(start, i)]
    target = turns[i]
    user_lines = []
    if ctx:
        user_lines.append("Previous turns:")
        user_lines.extend(ctx)
    user_lines.append(f"TARGET turn {target.speaker}: {target.text}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _extract_obj(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def parse_moves(raw, turn_text) -> dict | None:
    try:
        data = _extract_obj(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    moves_raw = data.get("moves", [])
    if not isinstance(moves_raw, list):
        return None
    moves: list[str] = []
    seen: set[str] = set()
    for m in moves_raw:
        if not isinstance(m, str):
            continue
        label = m.strip().lower()
        if label in MOVES and label not in seen:
            seen.add(label)
            moves.append(label)
            if len(moves) >= 3:
                break
    if not moves:
        return None
    quote = data.get("quote", "")
    if not isinstance(quote, str):
        quote = ""
    if quote and quote not in turn_text:
        quote = ""
    addressed = data.get("addressed_to", "all")
    if not isinstance(addressed, str):
        addressed = str(addressed)
    if not addressed:
        addressed = "all"
    return {"moves": moves, "addressed_to": addressed, "quote": quote}


def _tag_one(client, turns, i, model, window, reasoning_effort=None) -> MoveTag:
    messages = build_move_messages(turns, i, window)
    turn = turns[i]
    for _ in range(2):
        try:
            if reasoning_effort is None:
                raw = client.complete(messages, model, 0.0)
            else:
                try:
                    raw = client.complete(messages, model, 0.0,
                                          reasoning_effort=reasoning_effort)
                except TypeError:
                    raw = client.complete(messages, model, 0.0)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        parsed = parse_moves(raw, turn.text)
        if parsed is not None:
            return MoveTag(
                turn_index=i,
                speaker=turn.speaker,
                moves=parsed["moves"],
                addressed_to=parsed.get("addressed_to", "all"),
                quote=parsed.get("quote", ""),
                failed=False,
            )
    return MoveTag(turn_index=i, speaker=turn.speaker, moves=["other"], failed=True)


def tag_conversation(client, turns, model=DEFAULT_MODEL, window=4, max_workers=8,
                   reasoning_effort=None) -> list[MoveTag]:
    if not turns:
        return []
    workers = max_workers if isinstance(max_workers, int) and max_workers >= 1 else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_tag_one, client, turns, i, model, window, reasoning_effort)
                   for i in range(len(turns))]
        results = [f.result() for f in futures]
    results.sort(key=lambda t: t.turn_index)
    return results


def main(argv=None) -> None:
    import dataclasses

    from .transcribe import load_turns

    ap = argparse.ArgumentParser(prog="uptake-moves")
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--effort", default="low")
    args = ap.parse_args(argv)

    from .reader import MuseClient

    turns = load_turns(args.turns)
    tags = tag_conversation(MuseClient(), turns, model=args.model,
                            window=args.window, max_workers=args.workers,
                            reasoning_effort=args.effort)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tags:
            f.write(json.dumps(dataclasses.asdict(t), ensure_ascii=False) + "\n")
    print(f"{len(tags)} moves")
