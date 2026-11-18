import json
from pathlib import Path

import pytest
from inspect_ai.dataset import Sample

from eval_aware_bench.dataset import PromptRecord, load_records, record_to_sample


def test_record_ids_and_factor_settings() -> None:
    pair = "lyrics_purple_rain"
    baseline = {
        "F1": "realistic",
        "F2": "realistic",
        "F3": "casual",
        "F4": "realistic",
        "F5": "realistic",
        "F6": "realistic",
        "F7": "prose",
        "F8": "realistic",
    }
    records = load_records()
    assert len(records) == 4
    assert {record["id"] for record in records} == {
        f"{pair}_{valence}_{condition}"
        for valence in ("capability", "safety")
        for condition in ("baseline", "F8")
    }
    for record in records:
        assert record["pair_id"] == pair
        assert record["task_id"] == f"{pair}_{record['valence']}"
        assert record["id"] == f"{record['task_id']}_{record['condition']}"
        expected = baseline if record["condition"] == "baseline" else baseline | {"F8": "varied"}
        assert record["factor_config"] == expected


def test_messages_match_pinned_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).with_name("lyrics_purple_rain_reference.json")
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    for record in load_records():
        assert record["system_prompt"] == expected["system_prompt"]
        assert (
            record["user_prompt"]
            == expected["user_prompts"][record["valence"]][record["condition"]]
        )


@pytest.mark.parametrize("record", load_records(), ids=lambda record: record["id"])
def test_record_to_sample(record: PromptRecord) -> None:
    sample = record_to_sample(record)

    assert isinstance(sample, Sample)
    assert sample.id == record["id"]
    assert isinstance(sample.input, list)
    assert [(message.role, message.content) for message in sample.input] == [
        ("system", record["system_prompt"]),
        ("user", record["user_prompt"]),
    ]
    assert sample.metadata == {
        "task_id": record["task_id"],
        "pair_id": record["pair_id"],
        "valence": record["valence"],
        "condition": record["condition"],
        "factor_config": record["factor_config"],
    }
    assert sample.target == ""
