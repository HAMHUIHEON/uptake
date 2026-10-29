"""Strict JSON schemas for structured panel output (task35, Amendment 8 P8-A)."""
from __future__ import annotations

from .labels import NEXT_ACTIONS

DIST5_BINS = ("0-20", "20-40", "40-60", "60-80", "80-100")
DIST5_MIDPOINTS = (10, 30, 50, 70, 90)


def _action_prop() -> dict:
    return {
        "type": "string",
        "enum": list(NEXT_ACTIONS),
        "description": "listener's next action toward the speaker",
    }


def _understood_prop() -> dict:
    return {
        "type": "string",
        "description": "one sentence saying what the listener thinks the speaker meant",
    }


def _int_prop(desc: str) -> dict:
    return {"type": "integer", "description": desc}


def _dist_prop(desc: str) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "number",
            "description": "probability mass in one bin, 0 to 1",
        },
        "description": desc,
    }


def panel_schema(scale: str) -> dict:
    """Strict-subset JSON schema for one panel reading on `scale`.

    Root object, additionalProperties false, every property required, no
    min/max keywords (ranges live in the one-line descriptions instead).
    """
    if scale == "v2":
        props = {
            "next_action": _action_prop(),
            "relationship_shift": _int_prop(
                "integer from -2 (much colder) to 2 (much warmer)"),
            "face_threat": _int_prop(
                "integer from 0 (none) to 2 (severe)"),
            "understood": _understood_prop(),
        }
    elif scale == "fine100":
        props = {
            "next_action": _action_prop(),
            "relationship_after": _int_prop(
                "integer from 0 (much colder) to 100 (much warmer), 50 unchanged"),
            "face_effect": _int_prop(
                "integer from 0 (severe threat) to 100 (strongly affirms), 50 no effect"),
            "understood": _understood_prop(),
        }
    elif scale == "seven":
        props = {
            "next_action": _action_prop(),
            "relationship_shift": _int_prop(
                "integer from -3 (much colder) to 3 (much warmer), 0 unchanged"),
            "face_effect": _int_prop(
                "integer from -3 (severe threat) to 3 (strongly affirms), 0 no effect"),
            "understood": _understood_prop(),
        }
    elif scale == "dist5":
        props = {
            "next_action": _action_prop(),
            "p_rel": _dist_prop(
                "array of 5 numbers over bins 0-20/20-40/40-60/60-80/80-100 summing to 1"),
            "p_face": _dist_prop(
                "array of 5 numbers over bins 0-20/20-40/40-60/60-80/80-100 summing to 1"),
            "understood": _understood_prop(),
        }
    else:
        raise ValueError(f"bad scale: {scale!r}")
    return {
        "type": "object",
        "properties": props,
        "required": sorted(props),
        "additionalProperties": False,
    }


def response_format(scale: str) -> dict:
    """OpenAI-style response_format requesting `panel_schema(scale)`."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"panel_{scale}",
            "strict": True,
            "schema": panel_schema(scale),
        },
    }
