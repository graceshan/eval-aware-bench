"""Optional independent verification against the original checkout. Hashes only."""

import json
import os
import subprocess
from hashlib import sha256
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import pytest

from eval_aware_bench.dataset import load_records


def test_inventory_against_pinned_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = os.environ.get("EVALAWAREBENCH_UPSTREAM")
    if checkout is None:
        pytest.skip("Set EVALAWAREBENCH_UPSTREAM to verify the independent upstream reference.")
    root = Path(checkout).resolve()
    data = files("eval_aware_bench") / "data"
    catalog = json.loads((data / "single_cue_inventory.json").read_text(encoding="utf-8"))
    revision = "2961c17ebcb35fd15d5c614c4874d1f567b85d1b"
    assert (
        subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        == revision
    )
    assert catalog["revision"] == revision
    subprocess.run(
        [
            "git",
            "-C",
            root,
            "diff",
            "--quiet",
            revision,
            "--",
            "run",
            "eval_bench",
            "prompts",
            "utils.py",
            "LICENSE",
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(root))
    original = import_module("run.run_eval_bench")
    assert original.__file__ is not None
    assert Path(original.__file__).resolve() == root / "run/run_eval_bench.py"
    configs = original.build_configs()
    assert configs == list(catalog["configs"].items())
    package = files("eval_aware_bench")
    for source, asset in (
        ("eval_bench/render.py", package / "render.py"),
        ("prompts/chat_prompt_realistic.txt", data / "chat_prompt_realistic.txt"),
        ("prompts/judgePromptEvidenceBoth.txt", data / "judgePromptEvidenceBoth.txt"),
        ("LICENSE", package / "LICENSE.upstream"),
    ):
        assert (root / source).read_bytes() == asset.read_bytes(), source
    system = original.read_file(str(root / "prompts/chat_prompt_realistic.txt")).strip()
    assert sha256(system.encode()).hexdigest() == catalog["system_sha256"]
    assert (data / "chat_prompt_realistic.txt").read_text(encoding="utf-8").strip() == system
    tasks = original.load_all_tasks()
    assert len(tasks) == 200
    assert [(t["task_name"], t["valence"], t["task_id"]) for t in tasks] == [
        (pair, valence, variant["task_id"])
        for pair, variants in catalog["pairs"].items()
        for valence, variant in variants.items()
    ]
    records = iter(load_records(pair_ids="all", valences="all", conditions="all"))
    for item in tasks:
        pair, valence = item["task_name"], item["valence"]
        ref = catalog["pairs"][pair][valence]
        assert item["task_id"] == ref["task_id"]
        raw = (root / "eval_bench" / "tasks" / pair / f"{valence}.yaml").read_bytes()
        assert sha256(raw).hexdigest() == ref["yaml_sha256"]
        assert (data / "tasks" / pair / f"{valence}.yaml").read_bytes() == raw
        for condition, config in configs:
            record = next(records)
            assert record["id"] == f"{item['task_id']}_{condition}"
            assert record["system_prompt"] == system
            expected = sha256(original.render(item["template"], config).encode()).hexdigest()
            assert sha256(record["user_prompt"].encode()).hexdigest() == expected, record["id"]
            assert ref["user_sha256"][condition] == expected, record["id"]
    assert next(records, None) is None
