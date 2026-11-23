"""Analyze repeated judgments in saved EvalAwareBench Inspect logs.

A sample identifies a task under one cue condition (including baseline). An epoch
is one repeat of that sample. From each log's schedule we classify every repeat
as valid, unscored, errored, or missing. We calculate rates by target model,
valence (safety or capability), and condition, then pair each task's baseline
and cue completion rates to measure the change for each cue.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, NamedTuple

from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.scorer import Score

from eval_aware_bench.dataset import PromptRecord, load_task_inventory
from eval_aware_bench.generation import ReasoningStatus, extract_response_text
from eval_aware_bench.judge import JudgeVerdict
from eval_aware_bench.scorers import METRIC_NAMES, verdict_values

type GroupKey = tuple[str, str, str]  # Target model, valence, condition.
type LoggedReasoningStatus = ReasoningStatus | Literal["unknown"]


@dataclass
class GroupRates:
    """Repeat counts, reasoning availability, and five indicator rates for one group."""

    expected: int
    valid: int
    unscored: int
    errors: int
    missing: int
    reasoning_available: int
    reasoning_unavailable: int
    reasoning_unknown: int
    rates: dict[str, float | None]
    reasoning_status_counts: dict[LoggedReasoningStatus, int]
    usable_reasoning_rate: float | None


@dataclass
class PairedCompletion:
    """Mean cue-minus-baseline change in percentage points, per-task changes, and exclusions."""

    change_pp: float | None
    task_changes_pp: dict[str, float]
    excluded_tasks: dict[str, str]


@dataclass
class AnalysisReport:
    """Results by group, with each non-valid repeat indexed by log path, sample ID, and epoch."""

    groups: dict[GroupKey, GroupRates]
    paired: dict[GroupKey, PairedCompletion]
    failed_epochs: dict[tuple[str, str | int, int], str]


class _TaskCondition(NamedTuple):
    """Benchmark identity shared by all repeats of a sample."""

    task_id: str
    valence: str
    condition: str


class _Epoch(NamedTuple):
    """One repeat's outcome, legacy reasoning flag, evidence status, and valid indicators."""

    status: Literal["valid", "unscored", "error", "missing"]
    reasoning: bool | None
    reasoning_status: LoggedReasoningStatus
    values: dict[str, int] | None = None


def _score_values(score: Score, valence: str) -> dict[str, int]:
    """Return the five binary indicators from a scored judgment for this valence."""
    values = score.as_dict()
    if values.keys() == {"awareness_matched", "completion"}:
        # Older logs stored only these two indicators. Derive the full set from
        # the saved verdict and task valence, keeping the two recorded values.
        verdict = JudgeVerdict.model_validate((score.metadata or {}).get("judge_verdict"))
        values = verdict_values(verdict, valence) | values

    indicators = {}
    for name in METRIC_NAMES:
        value = values.get(name)
        if not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError(f"Expected a binary EvalAwareBench score for {name}, got {value!r}")
        indicators[name] = int(value)
    return indicators


def _reasoning_diagnostics(
    sample: EvalSample, score: Score | None
) -> tuple[bool | None, LoggedReasoningStatus]:
    """Recover the legacy presence flag and the evidence available to this repeat's judge."""
    metadata = (score.metadata or {}) if score else {}
    flag = metadata.get("reasoning_available")
    status = metadata.get("reasoning_status")
    if status not in ("unavailable", "empty", "usable"):
        if not sample.output.empty or sample.output.completion:
            status = extract_response_text(sample.output).reasoning_status
            if flag is True and status == "unavailable":
                # The flag records a string, but its text is absent from this copy of the output.
                status = "unknown"
        else:
            # A legacy True establishes presence but cannot distinguish empty from usable.
            status = "unavailable" if flag is False else "unknown"
    if not isinstance(flag, bool):
        flag = None if status == "unknown" else status != "unavailable"
    return flag, status


def _epoch(sample: EvalSample | None, valence: str) -> _Epoch:
    """Classify one scheduled repeat. An absent sample record is missing.

    A recorded error takes precedence even when a score was saved. Otherwise an
    invalidated sample, an absent scorer result, or Inspect's unscored NaN yields
    an unscored repeat. A valid repeat carries the five binary indicators.
    """
    if sample is None:
        return _Epoch("missing", None, "unknown")
    score = (sample.scores or {}).get("eval_awareness")
    presence, reasoning_status = _reasoning_diagnostics(sample, score)

    if sample.error is not None:
        return _Epoch("error", presence, reasoning_status)
    if sample.invalidation is not None or score is None:
        return _Epoch("unscored", presence, reasoning_status)
    if isinstance(score.value, float) and math.isnan(score.value):
        return _Epoch("unscored", presence, reasoning_status)
    return _Epoch("valid", presence, reasoning_status, _score_values(score, valence))


def _log_epochs(
    log: EvalLog, known: dict[str | int, _TaskCondition]
) -> Iterator[tuple[str | int, int, _TaskCondition, _Epoch]]:
    """Yield the sample ID, epoch number, task identity, and outcome for each scheduled repeat.

    Use both the dataset's selected IDs and the IDs of recorded samples. For a log
    without a selection list, only IDs with a surviving sample record are known
    to have been scheduled. Each ID has one repeat per configured epoch (one by
    default). Gaps are missing repeats. The inventory or supplied prompt records
    identify tasks whose sample metadata is absent.
    """
    epoch_count = log.eval.config.epochs or 1
    selected_ids = dict.fromkeys(log.eval.dataset.sample_ids or [])
    identities = known.copy()
    samples_by_epoch = {}
    for sample in log.samples or []:
        key = (sample.id, sample.epoch)
        if key in samples_by_epoch or not 1 <= sample.epoch <= epoch_count:
            raise ValueError(f"Duplicate or out-of-range epoch {key!r} in {log.location}")
        samples_by_epoch[key] = sample
        selected_ids[sample.id] = None
        meta = sample.metadata
        if all(field in meta for field in _TaskCondition._fields):
            identity = _TaskCondition(meta["task_id"], meta["valence"], meta["condition"])
            if identities.setdefault(sample.id, identity) != identity:
                raise ValueError(f"Inconsistent task metadata for {sample.id!r}")

    for sample_id in selected_ids:
        if sample_id not in identities:
            raise ValueError(f"Missing task metadata for {sample_id!r}. Supply its prompt record.")
        identity = identities[sample_id]
        for epoch in range(1, epoch_count + 1):
            yield (
                sample_id,
                epoch,
                identity,
                _epoch(samples_by_epoch.get((sample_id, epoch)), identity.valence),
            )


def _group_rates(rows: list[_Epoch]) -> GroupRates:
    """Summarize scheduled repeats for one target model, valence, and condition.

    Reasoning counts include all scheduled repeats. Usable reasoning is a fraction
    of repeats with known reasoning status. Each score rate uses only valid judgments.
    """
    valid_judgments = [row.values for row in rows if row.values is not None]
    status = Counter(row.status for row in rows)
    reasoning = Counter(row.reasoning for row in rows)
    evidence = Counter(row.reasoning_status for row in rows)
    known = len(rows) - evidence["unknown"]
    return GroupRates(
        expected=len(rows),
        valid=len(valid_judgments),
        unscored=status["unscored"],
        errors=status["error"],
        missing=status["missing"],
        reasoning_available=reasoning[True],
        reasoning_unavailable=reasoning[False],
        reasoning_unknown=reasoning[None],
        reasoning_status_counts={
            "usable": evidence["usable"],
            "empty": evidence["empty"],
            "unavailable": evidence["unavailable"],
            "unknown": evidence["unknown"],
        },
        usable_reasoning_rate=evidence["usable"] / known if known else None,
        rates={
            name: fmean(values[name] for values in valid_judgments) if valid_judgments else None
            for name in METRIC_NAMES
        },
    )


def _paired_completion(
    task_ids: list[str],
    baseline: dict[str, list[_Epoch]],
    cue: dict[str, list[_Epoch]],
    condition: str,
) -> PairedCompletion:
    """Compare one cue with baseline across tasks of the same model and valence.

    Both conditions must be scheduled and every scheduled repeat on both sides
    must be valid. Average completion across repeats within each task and condition,
    then average the cue-minus-baseline differences, giving each eligible task equal
    weight regardless of its repeat count or the model's awareness judgments.

    For example, 1/3 baseline completions and 2/3 cue completions contribute
    +33⅓ percentage points. An invalid judgment on either side excludes that
    task from this comparison. Only a valid judgment can establish a completion
    value of zero. If no task qualifies, change_pp is None.
    """
    changes = {}
    excluded = {}
    for task_id in task_ids:
        completion_rates = []
        problems = []
        for name, tasks in (("baseline", baseline), (condition, cue)):
            rows = tasks.get(task_id, [])
            completion_flags = [row.values["completion"] for row in rows if row.values is not None]
            if not rows:
                problems.append(f"{name} not scheduled")
            elif len(completion_flags) != len(rows):
                problems.append(f"{name} {len(completion_flags)}/{len(rows)} valid")
            else:
                completion_rates.append(fmean(completion_flags))
        if problems:
            excluded[task_id] = ", ".join(problems)
        else:
            baseline_rate, cue_rate = completion_rates
            changes[task_id] = 100 * (cue_rate - baseline_rate)
    return PairedCompletion(fmean(changes.values()) if changes else None, changes, excluded)


def analyze_logs(
    *paths: str | Path, records: Iterable[PromptRecord] | None = None
) -> AnalysisReport:
    """Pool repeats from saved logs and return grouped rates, paired changes, and failed epochs.

    Groups use the logged target model name, valence, and condition. Runs with the
    same model name share a group even if their grader or generation settings differ.
    Supply compatible runs, one copy of each, and only the chosen attempt for a retry.

    Optional prompt records identify samples with missing metadata and declare
    conditions to compare. Otherwise the pinned inventory supplies identities,
    and each model's scheduled samples determine which cues to compare with baseline.
    """
    known: dict[str | int, _TaskCondition]
    if records is None:
        known = {
            f"{row['task_id']}_{condition}": _TaskCondition(row["task_id"], valence, condition)
            for variants in load_task_inventory().values()
            for valence, row in variants.items()
            for condition in row["user_sha256"]
        }
        declared = {"baseline"}
    else:
        known = {
            r["id"]: _TaskCondition(r["task_id"], r["valence"], r["condition"]) for r in records
        }
        declared = {"baseline", *(r.condition for r in known.values())}
    conditions: dict[str, set[str]] = defaultdict(set)
    epochs_by_group: dict[GroupKey, dict[str, list[_Epoch]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    report = AnalysisReport(groups={}, paired={}, failed_epochs={})
    seen_logs = set()
    for path in paths:
        log = read_eval_log(
            path, resolve_attachments=True, exclude_fields={"events", "messages", "store"}
        )
        # Logs without an evaluation ID can still be checked for duplicate locations.
        log_id = log.eval.eval_id or log.location
        if log_id in seen_logs:
            raise ValueError(f"Duplicate evaluation log: {path}")
        seen_logs.add(log_id)
        model = log.eval.model
        if records is None and log.eval.task_version == "2-A":
            # Version 2-A defined the baseline/F8 pilot. Keep F8 in its report
            # even for a baseline-only log so the unscheduled comparison stays visible.
            conditions[model].add("F8")
        for sample_id, epoch, (task_id, valence, condition), row in _log_epochs(log, known):
            epochs_by_group[model, valence, condition][task_id].append(row)
            tasks[model, valence].add(task_id)
            conditions[model].add(condition)
            if row.status != "valid":
                report.failed_epochs[str(path), sample_id, epoch] = row.status

    for (model, valence), task_ids in sorted(tasks.items()):
        baseline = epochs_by_group[model, valence, "baseline"]
        for condition in sorted(declared | conditions[model]):
            key = (model, valence, condition)
            epochs_by_task = epochs_by_group[key]
            report.groups[key] = _group_rates(
                [row for rows in epochs_by_task.values() for row in rows]
            )
            if condition != "baseline":
                report.paired[key] = _paired_completion(
                    sorted(task_ids), baseline, epochs_by_task, condition
                )
    return report
