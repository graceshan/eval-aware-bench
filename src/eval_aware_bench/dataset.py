"""Load the paired EvalAwareBench prompts from the packaged upstream templates."""

import json
from collections.abc import Sequence
from hashlib import sha256
from importlib.resources import files
from itertools import product
from typing import TypedDict

import yaml
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from eval_aware_bench.render import render

_BASELINE_CONFIG = {
    "F1": "realistic",
    "F2": "realistic",
    "F3": "casual",
    "F4": "realistic",
    "F5": "realistic",
    "F6": "realistic",
    "F7": "prose",
    "F8": "realistic",
}
_VARIED_LEVELS = dict.fromkeys(_BASELINE_CONFIG, "varied") | {
    "F3": "formal",
    "F7": "template",
}
_CONFIGS = {"baseline": _BASELINE_CONFIG} | {
    factor: _BASELINE_CONFIG | {factor: level} for factor, level in _VARIED_LEVELS.items()
}


class TaskVariant(TypedDict):
    task_id: str
    yaml_sha256: str
    user_sha256: dict[str, str]


def load_task_inventory() -> dict[str, dict[str, TaskVariant]]:
    """Return the pinned task IDs and fingerprints of their YAMLs and rendered user prompts."""
    path = files("eval_aware_bench") / "data" / "single_cue_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))["pairs"]


def _select(field: str, selection: str | list[str] | None, available: Sequence[str]) -> list[str]:
    if selection is None or selection == "all":
        return list(available)
    values = [selection] if isinstance(selection, str) else selection
    if (
        not isinstance(values, (list, tuple))
        or not values
        or not all(isinstance(v, str) for v in values)
    ):
        raise ValueError(f"{field} requires a name, a nonempty list of names, or 'all'.")
    unknown = [v for v in values if v not in available]
    if unknown:
        raise ValueError(f"Unknown {field}: {unknown}. Valid names: {', '.join(available)}.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {field}: {values}.")
    return [v for v in available if v in values]


class PromptRecord(TypedDict):
    id: str
    task_id: str
    pair_id: str
    valence: str
    condition: str
    factor_config: dict[str, str]
    system_prompt: str
    user_prompt: str


def load_records(
    pair_ids: str | list[str] | None = None,
    valences: str | list[str] | None = None,
    conditions: str | list[str] | None = None,
) -> list[PromptRecord]:
    """Render the selected pairs, valences, and conditions in upstream order.

    Each filter accepts a name or list of names. An omitted filter, None, or
    "all" selects every value for that dimension. The default is 1,800 records:
    100 pairs, both valences, and baseline plus F1 through F8.
    """
    inventory = load_task_inventory()
    pairs = _select("pair_ids", pair_ids, sorted(inventory))
    variants = _select("valences", valences, ["capability", "safety"])
    cues = _select("conditions", conditions, list(_CONFIGS))

    data = files("eval_aware_bench") / "data"
    system_prompt = (data / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()

    records: list[PromptRecord] = []
    for pair, valence in product(pairs, variants):
        asset = data / "tasks" / pair / f"{valence}.yaml"
        raw = asset.read_bytes()
        if sha256(raw).hexdigest() != inventory[pair][valence]["yaml_sha256"]:
            raise ValueError(f"SHA-256 mismatch for data/tasks/{pair}/{valence}.yaml")
        template = yaml.safe_load(raw)
        task_id = template["task_id"]
        for condition in cues:
            factors = _CONFIGS[condition].copy()
            records.append(
                PromptRecord(
                    id=f"{task_id}_{condition}",
                    task_id=task_id,
                    pair_id=pair,
                    valence=valence,
                    condition=condition,
                    factor_config=factors,
                    system_prompt=system_prompt,
                    user_prompt=render(template, factors),
                )
            )
    return records


def record_to_sample(record: PromptRecord) -> Sample:
    return Sample(
        id=record["id"],
        input=[
            ChatMessageSystem(content=record["system_prompt"]),
            ChatMessageUser(content=record["user_prompt"]),
        ],
        metadata={
            "task_id": record["task_id"],
            "pair_id": record["pair_id"],
            "valence": record["valence"],
            "condition": record["condition"],
            "factor_config": record["factor_config"],
        },
    )
