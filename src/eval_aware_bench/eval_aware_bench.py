"""The single-cue EvalAwareBench task for Inspect, with a four-sample pilot default."""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from eval_aware_bench.dataset import load_records, record_to_sample
from eval_aware_bench.scorers import eval_awareness

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)


@task
def eval_aware_bench(
    pair_ids: str | list[str] | None = None,
    valences: str | list[str] | None = None,
    conditions: str | list[str] | None = None,
) -> Task:
    """Measure evaluation awareness and completion. Defaults to the four-sample Purple Rain pilot.

    Args:
        pair_ids: Pair name, list of names, or "all". Defaults to lyrics_purple_rain.
        valences: Valence name, list of names, or "all". Defaults to capability and safety.
        conditions: Condition name, list of names, or "all" (baseline and F1 through F8). Defaults to baseline and F8.
    """
    samples = [record_to_sample(record) for record in load_records(pair_ids, valences, conditions)]
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=eval_awareness(),
        version=EVAL_VERSION,
    )
