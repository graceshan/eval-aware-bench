from pathlib import Path
from typing import Literal

import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalError,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from eval_aware_bench.analysis import analyze_logs

SAFETY = "lyrics_purple_rain_safety"
CAPABILITY = "lyrics_purple_rain_capability"


def judged(
    awareness: Literal["matched", "mismatched", "none"],
    hypothetical: int,
    completion: int,
    *,
    reasoning: bool = True,
) -> Score:
    return Score(
        value={
            "awareness_any": int(awareness != "none"),
            "awareness_matched": int(awareness == "matched"),
            "awareness_mismatched": int(awareness == "mismatched"),
            "hypothetical_recognition": hypothetical,
            "completion": completion,
        },
        metadata={"reasoning_available": reasoning},
    )


def save_log(
    path: Path,
    model: str,
    epochs: int,
    cells: dict[tuple[str, str], list[Score | EvalError | None]],
) -> Path:
    """Write Inspect's real log format. None represents an epoch with no sample record."""
    ids = []
    samples = []
    for (task_id, condition), outcomes in cells.items():
        sample_id = f"{task_id}_{condition}"
        ids.append(sample_id)
        assert len(outcomes) == epochs
        for epoch, outcome in enumerate(outcomes, start=1):
            if outcome is None:
                continue
            sample = EvalSample(
                id=sample_id,
                epoch=epoch,
                input="Rendered user prompt",
                target="",
                metadata={
                    "task_id": task_id,
                    "valence": task_id.rsplit("_", 1)[1],
                    "condition": condition,
                },
            )
            if isinstance(outcome, EvalError):
                sample.error = outcome
                # The target answered before the grader failed. No score flag is available.
                sample.output = ModelOutput.from_content(model, "Answer without reasoning.")
            else:
                sample.scores = {"eval_awareness": outcome}
            samples.append(sample)
    log = EvalLog(
        status="error" if any(s.error for s in samples) else "success",
        eval=EvalSpec(
            task="offline_pilot",
            eval_id=path.stem,
            model=model,
            created="2026-11-18T12:00:00Z",
            dataset=EvalDataset(sample_ids=ids, samples=len(ids)),
            config=EvalConfig(epochs=epochs),
        ),
        samples=samples,
    )
    write_eval_log(log, path)
    return path


@pytest.fixture
def saved_runs(tmp_path: Path) -> list[Path]:
    invalid = Score.unscored(
        reason="grader_failed", explanation="{", metadata={"reasoning_available": False}
    )
    error = EvalError(message="Grader timeout", traceback="", traceback_ansi="")
    three = save_log(
        tmp_path / "three.eval",
        "mockllm/a",
        3,
        {
            (SAFETY, "baseline"): [
                judged("matched", 0, 1),
                judged("mismatched", 1, 0, reasoning=False),
                judged("none", 1, 0),
            ],
            (SAFETY, "F8"): [
                judged("none", 0, 0),
                judged("mismatched", 0, 1),
                judged("matched", 1, 1),
            ],
            (CAPABILITY, "baseline"): [
                judged("none", 1, 0),
                judged("none", 0, 0, reasoning=False),
                judged("none", 0, 1),
            ],
            (CAPABILITY, "F8"): [
                judged("matched", 1, 1),
                judged("matched", 1, 1),
                judged("matched", 0, 0),
            ],
            ("interrupted_capability", "baseline"): [judged("matched", 1, 0), invalid, error],
            ("interrupted_capability", "F8"): [judged("none", 0, 1), None, judged("none", 1, 1)],
        },
    )
    one = save_log(
        tmp_path / "one.eval",
        "mockllm/a",
        1,
        {
            ("encore_safety", "baseline"): [judged("none", 0, 1, reasoning=False)],
            ("encore_safety", "F8"): [judged("none", 1, 0)],
        },
    )
    other_model = save_log(
        tmp_path / "other.eval",
        "mockllm/b",
        3,
        {
            (SAFETY, "baseline"): [judged("none", 0, 0)] * 3,
            (SAFETY, "F8"): [None] * 3,  # Scheduled, but the entire sample is absent.
            (CAPABILITY, "F8"): [judged("matched", 1, 1)] * 3,  # No baseline was scheduled.
        },
    )
    return [three, one, other_model]


def test_grouped_rates_from_saved_epochs(saved_runs: list[Path]) -> None:
    report = analyze_logs(*saved_runs)
    groups = report.groups
    three = str(saved_runs[0])
    assert report.failed_epochs[three, "interrupted_capability_baseline", 2] == "unscored"
    assert report.failed_epochs[three, "interrupted_capability_baseline", 3] == "error"
    assert report.failed_epochs[three, "interrupted_capability_F8", 2] == "missing"
    safety = groups["mockllm/a", "safety", "baseline"]
    assert (safety.valid, safety.expected) == (4, 4)
    assert safety.rates == {
        "awareness_any": 2 / 4,
        "awareness_matched": 1 / 4,
        "awareness_mismatched": 1 / 4,
        "hypothetical_recognition": 2 / 4,
        "completion": 2 / 4,
    }
    cap = groups["mockllm/a", "capability", "baseline"]
    assert (cap.valid, cap.expected, cap.unscored, cap.errors, cap.missing) == (4, 6, 1, 1, 0)
    assert (cap.reasoning_available, cap.reasoning_unavailable, cap.reasoning_unknown) == (3, 3, 0)
    assert cap.rates == {
        "awareness_any": 1 / 4,
        "awareness_matched": 1 / 4,
        "awareness_mismatched": 0,
        "hypothetical_recognition": 2 / 4,
        "completion": 1 / 4,
    }
    cue = groups["mockllm/a", "capability", "F8"]
    assert (cue.valid, cue.expected, cue.missing, cue.reasoning_unknown) == (5, 6, 1, 1)
    assert cue.rates["completion"] == 4 / 5
    assert groups["mockllm/b", "capability", "F8"].rates["completion"] == 1
    missing = groups["mockllm/b", "safety", "F8"]
    assert (missing.valid, missing.expected, missing.missing, missing.reasoning_unknown) == (
        0,
        3,
        3,
        3,
    )
    assert all(rate is None for rate in missing.rates.values())
    unscheduled = groups["mockllm/b", "capability", "baseline"]
    assert (unscheduled.expected, unscheduled.rates["completion"]) == (0, None)


def test_task_weighted_paired_changes_and_exclusions(saved_runs: list[Path]) -> None:
    paired = analyze_logs(*saved_runs).paired
    safety = paired["mockllm/a", "safety", "F8"]
    # Task 1: (2/3 - 1/3); task 2: (0/1 - 1/1). Pooled rates would give zero.
    assert safety.task_changes_pp == pytest.approx({SAFETY: 100 / 3, "encore_safety": -100})
    assert safety.change_pp == pytest.approx(-100 / 3)
    assert safety.excluded_tasks == {}

    cap = paired["mockllm/a", "capability", "F8"]
    assert cap.task_changes_pp == pytest.approx({CAPABILITY: 100 / 3})
    assert cap.change_pp == pytest.approx(100 / 3)
    assert cap.excluded_tasks == {"interrupted_capability": "baseline 1/3 valid, F8 2/3 valid"}

    for valence, task_id, reason in (
        ("safety", SAFETY, "F8 0/3 valid"),
        ("capability", CAPABILITY, "baseline not scheduled"),
    ):
        change = paired["mockllm/b", valence, "F8"]
        assert change.change_pp is None
        assert change.task_changes_pp == {}
        assert change.excluded_tasks == {task_id: reason}


def test_legacy_two_indicator_log(tmp_path: Path) -> None:
    judgment = {"conclusion": "Yes", "evidence": ["Mock evidence."], "reasoning": "Mock rationale."}
    legacy = Score(
        value={"awareness_matched": 0, "completion": 1},
        metadata={
            "reasoning_available": False,
            "judge_verdict": {
                "model_awareness": {**judgment, "awareness_type": "capability"},
                "hypothetical_recognition": {**judgment, "conclusion": "No"},
                "task_performance": judgment,
            },
        },
    )
    path = save_log(tmp_path / "legacy.eval", "mockllm/a", 1, {(SAFETY, "baseline"): [legacy]})
    rates = analyze_logs(path).groups["mockllm/a", "safety", "baseline"].rates
    assert rates == {
        "awareness_any": 1,
        "awareness_matched": 0,
        "awareness_mismatched": 1,
        "hypothetical_recognition": 0,
        "completion": 1,
    }
    with pytest.raises(ValueError, match="Duplicate evaluation log"):
        analyze_logs(path, path)
