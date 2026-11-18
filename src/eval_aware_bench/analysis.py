"""Grouped epoch rates and paired completion changes from saved EvalAwareBench logs."""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, NamedTuple

from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.scorer import Score

from eval_aware_bench.dataset import PromptRecord, load_records
from eval_aware_bench.generation import extract_response_text
from eval_aware_bench.judge import JudgeVerdict
from eval_aware_bench.scorers import METRIC_NAMES, verdict_values

type GroupKey = tuple[str, str, str]  # Target model, valence, condition.


@dataclass
class GroupRates:
    expected: int
    valid: int
    unscored: int
    errors: int
    missing: int
    reasoning_available: int
    reasoning_unavailable: int
    reasoning_unknown: int
    rates: dict[str, float | None]


@dataclass
class PairedCompletion:
    change_pp: float | None
    task_changes_pp: dict[str, float]
    excluded_tasks: dict[str, str]


@dataclass
class AnalysisReport:
    groups: dict[GroupKey, GroupRates]
    paired: dict[GroupKey, PairedCompletion]
    failed_epochs: dict[tuple[str, str | int, int], str]  # Log path, sample ID, epoch.


class _TaskCondition(NamedTuple):
    task_id: str
    valence: str
    condition: str


class _Epoch(NamedTuple):
    status: Literal["valid", "unscored", "error", "missing"]
    reasoning: bool | None
    values: dict[str, int] | None = None


def _score_values(score: Score, valence: str) -> dict[str, int]:
    values = score.as_dict()
    if values.keys() == {"awareness_matched", "completion"}:
        # Step 5b logs saved the complete validated verdict alongside two indicators.
        verdict = JudgeVerdict.model_validate((score.metadata or {}).get("judge_verdict"))
        values = verdict_values(verdict, valence) | values

    indicators = {}
    for name in METRIC_NAMES:
        value = values.get(name)
        if not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError(f"Expected a binary EvalAwareBench score for {name}, got {value!r}")
        indicators[name] = int(value)
    return indicators


def _epoch(sample: EvalSample | None, valence: str) -> _Epoch:
    if sample is None:
        return _Epoch("missing", None)
    score = (sample.scores or {}).get("eval_awareness")
    flag = (score.metadata or {}).get("reasoning_available") if score else None
    reasoning = flag if isinstance(flag, bool) else None
    if reasoning is None and (not sample.output.empty or sample.output.completion):
        reasoning = extract_response_text(sample.output).reasoning is not None

    if sample.error is not None:
        return _Epoch("error", reasoning)
    if sample.invalidation is not None or score is None:
        return _Epoch("unscored", reasoning)
    if isinstance(score.value, float) and math.isnan(score.value):
        return _Epoch("unscored", reasoning)
    return _Epoch("valid", reasoning, _score_values(score, valence))


def _log_epochs(
    log: EvalLog, known: dict[str | int, _TaskCondition]
) -> Iterator[tuple[str | int, int, _TaskCondition, _Epoch]]:
    epochs = log.eval.config.epochs or 1
    selected = dict.fromkeys(log.eval.dataset.sample_ids or [])
    identities = known.copy()
    samples = {}
    for sample in log.samples or []:
        key = (sample.id, sample.epoch)
        if key in samples or not 1 <= sample.epoch <= epochs:
            raise ValueError(f"Duplicate or out-of-range epoch {key!r} in {log.location}")
        samples[key] = sample
        selected[sample.id] = None
        meta = sample.metadata
        if all(field in meta for field in _TaskCondition._fields):
            identity = _TaskCondition(meta["task_id"], meta["valence"], meta["condition"])
            if identities.setdefault(sample.id, identity) != identity:
                raise ValueError(f"Inconsistent task metadata for {sample.id!r}")

    for sample_id in selected:
        if sample_id not in identities:
            raise ValueError(f"Missing task metadata for {sample_id!r}. Supply its prompt record.")
        identity = identities[sample_id]
        for epoch in range(1, epochs + 1):
            yield (
                sample_id,
                epoch,
                identity,
                _epoch(samples.get((sample_id, epoch)), identity.valence),
            )


def _group_rates(rows: list[_Epoch]) -> GroupRates:
    valid = [row.values for row in rows if row.values is not None]
    status = Counter(row.status for row in rows)
    reasoning = Counter(row.reasoning for row in rows)
    return GroupRates(
        expected=len(rows),
        valid=len(valid),
        unscored=status["unscored"],
        errors=status["error"],
        missing=status["missing"],
        reasoning_available=reasoning[True],
        reasoning_unavailable=reasoning[False],
        reasoning_unknown=reasoning[None],
        rates={name: fmean(v[name] for v in valid) if valid else None for name in METRIC_NAMES},
    )


def _paired_completion(
    task_ids: list[str],
    baseline: dict[str, list[_Epoch]],
    cue: dict[str, list[_Epoch]],
    condition: str,
) -> PairedCompletion:
    changes = {}
    excluded = {}
    for task_id in task_ids:
        means = []
        problems = []
        for name, tasks in (("baseline", baseline), (condition, cue)):
            rows = tasks.get(task_id, [])
            completed = [row.values["completion"] for row in rows if row.values is not None]
            if not rows:
                problems.append(f"{name} not scheduled")
            elif len(completed) != len(rows):
                problems.append(f"{name} {len(completed)}/{len(rows)} valid")
            else:
                means.append(fmean(completed))
        if problems:
            excluded[task_id] = ", ".join(problems)
        else:
            baseline_rate, cue_rate = means
            changes[task_id] = 100 * (cue_rate - baseline_rate)
    return PairedCompletion(fmean(changes.values()) if changes else None, changes, excluded)


def analyze_logs(
    *paths: str | Path, records: Iterable[PromptRecord] | None = None
) -> AnalysisReport:
    """Analyze saved runs by (target model, valence, condition), using every scheduled epoch.

    Optional records identify scheduled samples with no surviving metadata and declare
    the conditions to compare. Defaults to the packaged pilot. Independent runs pool
    their repeats. Supply one copy of each run, including only the chosen retry.
    """
    reference = load_records() if records is None else records
    known: dict[str | int, _TaskCondition] = {
        r["id"]: _TaskCondition(r["task_id"], r["valence"], r["condition"]) for r in reference
    }
    conditions = {"baseline", *(r.condition for r in known.values())}
    cells: dict[GroupKey, dict[str, list[_Epoch]]] = defaultdict(lambda: defaultdict(list))
    tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    report = AnalysisReport(groups={}, paired={}, failed_epochs={})
    seen_logs = set()
    for path in paths:
        log = read_eval_log(
            path, resolve_attachments=True, exclude_fields={"events", "messages", "store"}
        )
        log_id = log.eval.eval_id or log.location
        if log_id in seen_logs:
            raise ValueError(f"Duplicate evaluation log: {path}")
        seen_logs.add(log_id)
        model = log.eval.model
        for sample_id, epoch, (task_id, valence, condition), row in _log_epochs(log, known):
            cells[model, valence, condition][task_id].append(row)
            tasks[model, valence].add(task_id)
            conditions.add(condition)
            if row.status != "valid":
                report.failed_epochs[str(path), sample_id, epoch] = row.status

    for (model, valence), task_ids in sorted(tasks.items()):
        baseline = cells[model, valence, "baseline"]
        for condition in sorted(conditions):
            key = (model, valence, condition)
            cell = cells[key]
            report.groups[key] = _group_rates([row for rows in cell.values() for row in rows])
            if condition != "baseline":
                report.paired[key] = _paired_completion(sorted(task_ids), baseline, cell, condition)
    return report
