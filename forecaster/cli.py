import argparse
import dataclasses
import json
from datetime import datetime, timezone

from . import score
from .forecast import run_forecast
from .reader import DEFAULT_MODEL, MuseClient
from .rewrite import run_rewrite
from .runlog import save_readings
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save", default=None)
    ap.add_argument("--rewrite", default=None)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    message, intent, receiver = _load_case(args.case)

    if args.rewrite is not None:
        res = run_rewrite(MuseClient(), message, args.rewrite, intent, receiver,
                          n=args.n, k=args.k, seed=args.seed,
                          b=args.b, model=args.model,
                          max_workers=args.workers)
        before, after = res["before"], res["after"]
        point, low, high = res["diff"]
        if args.save:
            save_readings(args.save, after.readings, meta={
                "message": args.rewrite,
                "intent": dataclasses.asdict(intent),
                "receiver": dataclasses.asdict(receiver),
                "model": args.model,
                "n": args.n,
                "k": args.k,
                "seed": args.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if args.as_json:
            print(json.dumps({
                "before": dataclasses.asdict(before),
                "after": dataclasses.asdict(after),
                "diff": list(res["diff"]),
                "panel": [r.id for r in res["panel"]],
            }, ensure_ascii=False, indent=2))
            return
        print(f"Before miss rate: {before.miss_rate:.3f} "
              f"(95% CI {before.ci_low:.3f}-{before.ci_high:.3f})")
        print(f"After miss rate: {after.miss_rate:.3f} "
              f"(95% CI {after.ci_low:.3f}-{after.ci_high:.3f})")
        print(f"Paired diff (after - before): {point:.3f} "
              f"(95% CI {low:.3f}-{high:.3f})")
        return

    forecast = run_forecast(MuseClient(), message, intent, receiver,
                            n=args.n, k=args.k, seed=args.seed,
                            model=args.model, b=args.b,
                            max_workers=args.workers)
    if args.save:
        save_readings(args.save, forecast.readings, meta={
            "message": message,
            "intent": dataclasses.asdict(intent),
            "receiver": dataclasses.asdict(receiver),
            "model": args.model,
            "n": args.n,
            "k": args.k,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
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
