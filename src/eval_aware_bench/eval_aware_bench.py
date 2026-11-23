"""The single-cue EvalAwareBench task for Inspect."""

from inspect_ai import Epochs, Task, task
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
    """Measure evaluation awareness and completion on all 1,800 single-cue samples by default.

    Args:
        pair_ids: Pair name, list of names, or "all". Omit to include all 100 pairs.
        valences: Valence name, list of names, or "all". Omit to include capability and safety.
        conditions: Condition name, list of names, or "all". Omit to include baseline and F1 through F8.
    """
    samples = [record_to_sample(record) for record in load_records(pair_ids, valences, conditions)]
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=eval_awareness(),
        # Keep each repeat separate so means and coverage count individual judgments.
        epochs=Epochs(1, reducer=[]),
        version=EVAL_VERSION,
    )
