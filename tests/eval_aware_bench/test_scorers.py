from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from eval_aware_bench import scorers
from eval_aware_bench.scorers import eval_awareness


@pytest.mark.parametrize("source", ["grader", "parser"])
async def test_eval_awareness_propagates_unexpected_errors(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = TaskState(
        model=ModelName("mockllm/target"),
        sample_id=1,
        epoch=1,
        input="Original prompt.",
        messages=[],
    )
    grader = get_model("mockllm/grader")
    resolve_grader = Mock(return_value=grader)
    monkeypatch.setattr(scorers, "get_model", resolve_grader)
    failure = TimeoutError("grader timeout") if source == "grader" else ValueError("parser bug")
    generate_reply = AsyncMock(return_value=ModelOutput(completion="{}"))
    if source == "grader":
        generate_reply.side_effect = failure
    else:
        monkeypatch.setattr(scorers, "parse_judge_verdict", Mock(side_effect=failure))
    monkeypatch.setattr(grader, "generate", generate_reply)
    with pytest.raises(type(failure)) as caught:
        await eval_awareness()(state, Target(""))
    assert caught.value is failure
    resolve_grader.assert_called_once_with(role="grader", default="openai/gpt-5")
