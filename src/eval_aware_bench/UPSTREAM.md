# EvalAwareBench source and license

The renderer and prompt templates come from *Decomposing and Measuring Evaluation Awareness* by Changling Li, Terry Jingchen Zhang, Jie Zhang, Zhijing Jin, Florian Tramèr, Sahar Abdelnabi, and Maksym Andriushchenko. The reference revision is `2961c17ebcb35fd15d5c614c4874d1f567b85d1b`. [Source repository at the pinned revision](https://github.com/aisa-group/decomposing-eval-awareness/tree/2961c17ebcb35fd15d5c614c4874d1f567b85d1b) · [Paper](https://arxiv.org/abs/2605.23055)

The following files are byte-for-byte copies. Local paths are relative to this package.

- `eval_bench/render.py` → `render.py`
- `eval_bench/tasks/lyrics_purple_rain/capability.yaml` → `data/tasks/lyrics_purple_rain/capability.yaml`
- `eval_bench/tasks/lyrics_purple_rain/safety.yaml` → `data/tasks/lyrics_purple_rain/safety.yaml`
- `prompts/chat_prompt_realistic.txt` → `data/chat_prompt_realistic.txt`
- `LICENSE` → `LICENSE.upstream`

The copied material and its rendered prompts carry the upstream Creative Commons Attribution-NonCommercial 4.0 International license. [License text](LICENSE.upstream) · [Upstream data card](https://github.com/aisa-group/decomposing-eval-awareness/blob/2961c17ebcb35fd15d5c614c4874d1f567b85d1b/eval_bench/DATACARD.md)

The offline test fixture at `tests/eval_aware_bench/lyrics_purple_rain_reference.json` was generated from the pinned checkout. It uses the system prompt with the runner’s `.strip()` and the output of the upstream `render()` on the two templates returned by `run/run_eval_bench.py::load_all_tasks()`, using the `baseline` and `F8` entries from `build_configs()`. JSON escaping preserves blank lines and the final newline in each user message.
