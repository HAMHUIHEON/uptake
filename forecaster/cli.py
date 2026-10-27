import argparse
import dataclasses
import json

from . import score
from .forecast import run_forecast
from .reader import DEFAULT_MODEL, MuseClient
from .types import Intent, Receiver


def _load_case(path: str) -> tuple[str, Intent, Receiver]:
    with open(path) as f:
        case = json.load(f)
    return (
        case["message"],
        Intent(**case["intent"]),
        Receiver(**case["receiver"]),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="forecast")
    ap.add_argument("--case", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    message, intent, receiver = _load_case(args.case)
    forecast = run_forecast(MuseClient(), message, intent, receiver,
                            n=args.n, k=args.k, seed=args.seed,
                            model=args.model, b=args.b)
    if args.as_json:
        print(json.dumps(dataclasses.asdict(forecast), ensure_ascii=False, indent=2))
        return

    missed = [r for r in forecast.readings if score.is_miss(r, intent)]
    print(f"Miss rate: {forecast.miss_rate:.3f} (95% CI {forecast.ci_low:.3f}-{forecast.ci_high:.3f})")
    print("Per-receiver rates:")
    for rid in sorted(forecast.per_receiver):
        print(f"  {rid}: {forecast.per_receiver[rid]:.3f}")
    print("Top triggers:")
    for trigger, count in forecast.top_triggers:
        print(f"  {trigger!r}: {count}")
    print("Example missed glosses:")
    for r in missed[:3]:
        print(f"  - {r.gloss}")
