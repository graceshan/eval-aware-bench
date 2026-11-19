"""The four-sample EvalAwareBench pilot for Inspect."""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from eval_aware_bench.dataset import load_records, record_to_sample
from eval_aware_bench.scorers import eval_awareness

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)


@task
def eval_aware_bench() -> Task:
    """Measure awareness and completion on the Purple Rain pair at baseline and with F8."""
    samples = [record_to_sample(record) for record in load_records()]
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=eval_awareness(),
        version=EVAL_VERSION,
    )
