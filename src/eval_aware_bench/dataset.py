"""Load the paired EvalAwareBench prompts from the packaged upstream templates."""

from importlib.resources import files
from typing import TypedDict

import yaml

from eval_aware_bench.render import render

_PAIR_ID = "lyrics_purple_rain"
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


class PromptRecord(TypedDict):
    id: str
    task_id: str
    pair_id: str
    valence: str
    condition: str
    factor_config: dict[str, str]
    system_prompt: str
    user_prompt: str


def load_records() -> list[PromptRecord]:
    """Render baseline and F8 prompts for each valence of the Purple Rain pair."""
    data = files("eval_aware_bench") / "data"
    system_prompt = (data / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip()
    pair_dir = data / "tasks" / _PAIR_ID

    records: list[PromptRecord] = []
    for valence in ("capability", "safety"):
        template = yaml.safe_load((pair_dir / f"{valence}.yaml").read_text(encoding="utf-8"))
        task_id = template["task_id"]
        for condition, factors in (
            ("baseline", _BASELINE_CONFIG.copy()),
            ("F8", _BASELINE_CONFIG | {"F8": "varied"}),
        ):
            records.append(
                PromptRecord(
                    id=f"{task_id}_{condition}",
                    task_id=task_id,
                    pair_id=_PAIR_ID,
                    valence=valence,
                    condition=condition,
                    factor_config=factors,
                    system_prompt=system_prompt,
                    user_prompt=render(template, factors),
                )
            )
    return records
