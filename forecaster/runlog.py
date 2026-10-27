import dataclasses
import json

from .types import Reading


def save_readings(path, readings: list[Reading], meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in readings:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")


def load_readings(path) -> tuple[dict, list[Reading]]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return {}, []
    meta = json.loads(lines[0]).get("meta", {})
    readings = [Reading(**json.loads(line)) for line in lines[1:] if line.strip()]
    return meta, readings
