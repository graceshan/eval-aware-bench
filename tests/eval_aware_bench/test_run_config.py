import sys
from functools import partial
from importlib.resources import as_file, files
from pathlib import Path
from runpy import run_module
from unittest.mock import patch

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog, read_eval_log


def run_config(
    name: str,
    options: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_samples: bool = False,
) -> EvalLog:
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "offline")
    if not run_samples:
        monkeypatch.setenv("INSPECT_DISABLE_MODEL_API", "1")
    config = files("eval_aware_bench") / "run_configs" / name
    with as_file(config) as path:
        args = ["inspect", "eval", "--run-config", str(path), "--log-dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", [*args, "--display", "none", *options])
        with (
            patch("inspect_ai._cli.eval.eval", partial(eval, run_samples=run_samples)),
            pytest.raises(SystemExit) as exit_info,
        ):
            run_module("inspect_ai", run_name="__main__")
    assert exit_info.value.code == 0
    [log_path] = tmp_path.glob("*.eval")
    return read_eval_log(log_path, header_only=not run_samples)


@pytest.mark.parametrize(
    ("options", "target", "epochs", "conditions"),
    [
        ([], "openrouter/openai/gpt-oss-120b", 1, ["baseline", "F8"]),
        (
            [
                "--model",
                "openrouter/openai/gpt-oss-20b",
                "--epochs",
                "3",
                "-T",
                "conditions=[baseline,F3,F7]",
            ],
            "openrouter/openai/gpt-oss-20b",
            3,
            ["baseline", "F3", "F7"],
        ),
    ],
    ids=["pilot", "cli-overrides"],
)
def test_pilot_run_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    target: str,
    epochs: int,
    conditions: list[str],
) -> None:
    log = run_config("pilot.yaml", options, tmp_path, monkeypatch)
    assert log.eval.task == "eval_aware_bench/eval_aware_bench"
    assert log.eval.task_args == {
        "pair_ids": ["lyrics_purple_rain"],
        "valences": ["capability", "safety"],
        "conditions": conditions,
    }
    assert log.eval.config.epochs == epochs
    assert log.eval.config.sample_id is None
    assert log.eval.config.limit is None
    assert log.eval.dataset.sample_ids == [
        f"lyrics_purple_rain_{valence}_{condition}"
        for valence in ("capability", "safety")
        for condition in conditions
    ]

    transport = {"stream": False, "responses_api": False, "max_retries": 0}
    assert log.eval.model == target
    assert log.eval.model_args == transport
    assert log.plan.config.temperature == 0.3
    assert log.plan.config.max_tokens == 4096
    assert log.plan.config.reasoning_effort == "high"
    assert log.plan.config.max_retries == 1
    assert log.plan.config.cache is False

    assert log.eval.model_roles is not None
    grader = log.eval.model_roles["grader"]
    assert not isinstance(grader, list)
    assert grader.model == "openrouter/openai/gpt-5"
    assert grader.args == transport
    assert grader.base_url == log.eval.model_base_url == "https://openrouter.ai/api/v1"
    assert grader.config.model_dump(exclude_none=True) == {"max_retries": 1}


def test_full_single_cue_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = run_config("full-single-cue.yaml", ["--model", "mockllm/model"], tmp_path, monkeypatch)
    assert log.eval.task == "eval_aware_bench/eval_aware_bench"
    assert log.eval.task_args == dict.fromkeys(("pair_ids", "valences", "conditions"), "all")
    assert log.eval.config.epochs == 3
    assert log.eval.config.limit is None
    assert log.eval.config.sample_id is None
    assert log.eval.dataset.sample_ids is not None
    assert len(log.eval.dataset.sample_ids) == 1800
    assert len(set(log.eval.dataset.sample_ids)) == 1800
