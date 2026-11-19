# eval-aware-bench

The registered Inspect task runs the four-sample EvalAwareBench pilot. It presents the safety and capability versions of the Purple Rain task at baseline and with the F8 cue. Each epoch makes four target generations and four separate judge calls. [Dataset, scoring, and log analysis](src/eval_aware_bench/README.md)

## Getting started

```bash
uv sync
uv run pre-commit install
INSPECT_DISABLE_MODEL_API=1 uv run pytest tests/eval_aware_bench/test_eval_aware_bench.py
```

The test runs `eval_aware_bench/eval_aware_bench` with scripted target and grader replies, then reads the saved logs. For a provider-backed run, select the target with `--model` and the grader with `--model-role`. The grader defaults to `openai/gpt-5`. Provide credentials for both selected models.

```bash
uv run inspect eval eval_aware_bench/eval_aware_bench --model anthropic/claude-sonnet-5-5 --model-role grader=openai/gpt-5
uv run inspect view
```

Put `INSPECT_EVAL_MODEL` and your API keys in the git-ignored `.env` file to reuse them across runs.

The dev dependencies include the `anthropic` and `openai` SDKs. Inspect needs one package per model provider, so add the package for any other provider with `uv add --dev <package>`; Inspect's error message names the package when it is missing.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Slow tests run with `uv run pytest --runslow`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the other gates and markers.

## Layout

```text
src/eval_aware_bench/        the eval: task module, eval.yaml, README.md
tests/eval_aware_bench/      its tests
src/examples/            gpqa, agentic, llm_judge: worked examples, see Examples below
tests/examples/          their tests
agent_artefacts/trajectory_analysis/inspect_scout/   Scout scanners for the transcript audit
```

## Examples

`src/examples/` holds worked examples that are separate from your eval. They are not registered as a package, so run them by file path:

```bash
uv run inspect eval src/examples/gpqa/gpqa.py --limit 5
uv run inspect eval src/examples/agentic/agentic.py                        # ReAct agent, needs Docker
uv run inspect eval src/examples/agentic/agentic.py -T agent=claude_code   # Claude Code through inspect_swe
uv run inspect eval src/examples/llm_judge/llm_judge.py
```

When your eval works, commit your changes and remove the examples with `uvx copier update --defaults --data include_examples=false`, then run `uv sync`. The update deletes `src/examples/` and `tests/examples/`, drops the `inspect-swe` dev dependency and the examples line in `AGENTS.md`, and applies any newer template release. An update never changes `README.md`, so delete this section and the two examples lines under Layout by hand.

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Keeping up with the template

A weekly workflow runs `copier update` and opens a pull request when the template changed.
`README.md`, `src/eval_aware_bench/` and `tests/eval_aware_bench/` are never overwritten by an update.

## Eval documentation

See [src/eval_aware_bench/README.md](src/eval_aware_bench/README.md).
