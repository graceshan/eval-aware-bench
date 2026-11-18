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

Describe the scorer here: what counts as correct, and which metrics the task reports.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| Nothing yet. | | |

## Architecture

Add a diagram of the task here: dataset, solver, scorer, and any sandbox or judge model.
