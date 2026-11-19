import json
from collections import Counter
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import pytest
from inspect_ai.dataset import Sample

from eval_aware_bench.dataset import PromptRecord, load_records, record_to_sample
from eval_aware_bench.eval_aware_bench import eval_aware_bench


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


def test_single_cue_inventory_and_packaged_records() -> None:
    data = files("eval_aware_bench") / "data"
    inventory = json.loads((data / "single_cue_inventory.json").read_text(encoding="utf-8"))
    pairs = inventory["pairs"]
    configs = inventory["configs"]
    cues = ["baseline", *(f"F{i}" for i in range(1, 9))]
    assert list(configs) == cues
    assert len(pairs) == 100
    assert all(set(variants) == {"capability", "safety"} for variants in pairs.values())
    expected_assets = set()
    expected_ids = []
    for pair, variants in sorted(pairs.items()):
        for valence in ("capability", "safety"):
            variant = variants[valence]
            assert list(variant["user_sha256"]) == cues
            expected_ids.extend(f"{variant['task_id']}_{cue}" for cue in cues)
            expected_assets.add((pair, f"{valence}.yaml"))
            raw = (data / "tasks" / pair / f"{valence}.yaml").read_bytes()
            assert sha256(raw).hexdigest() == variant["yaml_sha256"], (pair, valence)
    actual_assets = {
        (d.name, f.name)
        for d in (data / "tasks").iterdir()
        if d.is_dir()
        for f in d.iterdir()
        if f.name.endswith(".yaml")
    }
    assert actual_assets == expected_assets
    assert len(actual_assets) == 200

    records = load_records(pair_ids="all", valences="all", conditions="all")
    ids = [r["id"] for r in records]
    assert ids == expected_ids
    assert len(set(ids)) == 1800
    assert Counter(r["valence"] for r in records) == {"capability": 900, "safety": 900}
    for r in records:
        variant = pairs[r["pair_id"]][r["valence"]]
        sample = record_to_sample(r)
        assert sample.id == f"{variant['task_id']}_{r['condition']}"
        assert r["task_id"] == variant["task_id"]
        assert r["factor_config"] == configs[r["condition"]]
        assert isinstance(sample.input, list)
        assert [m.role for m in sample.input] == ["system", "user"]
        system, user = sample.input
        assert isinstance(system.content, str)
        assert isinstance(user.content, str)
        assert sha256(system.content.encode()).hexdigest() == inventory["system_sha256"]
        assert sha256(user.content.encode()).hexdigest() == variant["user_sha256"][r["condition"]]


def test_filters_preserve_upstream_order_and_pilot_defaults() -> None:
    task = eval_aware_bench(
        ["lyrics_purple_rain", "lyrics_99_problems"], "capability", ["F7", "F3"]
    )
    assert [s.id for s in task.dataset] == [
        f"{pair}_capability_{cue}"
        for pair in ("lyrics_99_problems", "lyrics_purple_rain")
        for cue in ("F3", "F7")
    ]
    all_cues = load_records(conditions="all")
    assert len(all_cues) == 18
    assert [r for r in all_cues if r["condition"] in ("baseline", "F8")] == load_records()


@pytest.mark.parametrize(
    ("field", "selection"),
    [
        ("pair_ids", []),
        ("pair_ids", "../tasks"),
        ("valences", "either"),
        ("conditions", "F9"),
        ("conditions", ["F1", "F1"]),
        ("conditions", ["all", "F1"]),
    ],
)
def test_invalid_selection(field: str, selection: str | list[str]) -> None:
    with pytest.raises(ValueError, match=field):
        load_records(**{field: selection})


def test_missing_and_modified_selected_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = files("eval_aware_bench") / "data"
    data = tmp_path / "data"
    pair = "lyrics_purple_rain"
    task_dir = data / "tasks" / pair
    task_dir.mkdir(parents=True)
    for name in ("single_cue_inventory.json", "chat_prompt_realistic.txt"):
        (data / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr("eval_aware_bench.dataset.files", lambda _: tmp_path)

    asset = task_dir / "capability.yaml"
    with pytest.raises(FileNotFoundError) as err:
        load_records(pair, "capability", "baseline")
    assert err.value.filename == str(asset)

    # A YAML comment leaves the rendered prompt intact but changes the source bytes.
    original = (source / "tasks" / pair / asset.name).read_bytes()
    asset.write_bytes(original + b"\n# changed\n")
    with pytest.raises(
        ValueError, match=r"SHA-256 mismatch for data/tasks/lyrics_purple_rain/capability\.yaml"
    ):
        load_records(pair, "capability", "baseline")
