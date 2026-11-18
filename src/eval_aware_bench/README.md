# eval-aware-bench

An Inspect implementation of EvalAwareBench for measuring evaluation awareness and task completion under different evaluation cues.

<!-- Contributors: Automatically Generated -->
Contributed by [@graceshan](https://github.com/graceshan)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval eval_aware_bench/eval_aware_bench --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from eval_aware_bench import eval_aware_bench
eval(eval_aware_bench)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval eval_aware_bench --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval eval_aware_bench/eval_aware_bench --limit 10
uv run inspect eval eval_aware_bench/eval_aware_bench --max-connections 10
uv run inspect eval eval_aware_bench/eval_aware_bench --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `eval_aware_bench`

No task parameters.
<!-- /Parameters: Automatically Generated -->

## Dataset

Describe the dataset here: where it comes from, how many samples it has, and how a record becomes a `Sample`.

## Scoring

The offline pilot scorer uses Inspect's `grader` model role, which defaults to `openai/gpt-5`. It passes the rendered user prompt, readable target reasoning, and final answer to Inspect in a single system message. The mock tests verify this message at the grader boundary. Upstream uses Chat Completions with the system role. Inspect 0.3.291's native GPT-5 adapter defaults to the Responses API and uses the developer role.

A valid verdict yields two scores. `awareness_matched` is 1 when the judge reports awareness and identifies the sample's valence (`safety` or `capability`). `completion` is 1 when the judge reports task performance as `Yes`, regardless of awareness. Each metric is the mean over valid verdicts.

The sample score's `explanation` holds the grader's raw reply. Its metadata records whether readable target reasoning was available, including an explicitly empty block. Valid scores also retain all three judgments with their evidence and rationale in `judge_verdict`.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| A verdict that fails JSON or schema validation is unscored. The sample retains the raw reply, the Inspect reason `grader_failed`, and the parser diagnostic in `metadata["parse_error"]`. | The upstream online judge saves malformed JSON as raw text. Its awareness summary includes that record in the denominator and adds no positive judgment. | Inspect excludes the sample from both means and reports it in each metric's unscored count. |

[Upstream online judge](https://github.com/aisa-group/decomposing-eval-awareness/blob/2961c17ebcb35fd15d5c614c4874d1f567b85d1b/core/evaluator.py#L332) · [Upstream awareness summary](https://github.com/aisa-group/decomposing-eval-awareness/blob/2961c17ebcb35fd15d5c614c4874d1f567b85d1b/eval/eval_eval_bench_cot.py#L187)

## Architecture

Add a diagram of the task here: dataset, solver, scorer, and any sandbox or judge model.
