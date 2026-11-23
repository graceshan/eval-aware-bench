from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    Content,
    ContentReasoning,
    ContentText,
    ModelOutput,
    ModelUsage,
    get_model,
)

from eval_aware_bench.dataset import load_records
from eval_aware_bench.generation import extract_response_text
from tests.eval_aware_bench import PILOT_SELECTION


@pytest.mark.parametrize(
    ("block", "expected_text", "status"),
    [
        (None, None, "unavailable"),
        (ContentReasoning(reasoning=""), "", "empty"),
        (ContentReasoning(reasoning=" \t\n"), " \t\n", "empty"),
        (ContentReasoning(reasoning=" A thought. "), " A thought. ", "usable"),
        (ContentReasoning(reasoning="ciphertext", redacted=True), None, "unavailable"),
        (ContentReasoning(reasoning="ciphertext", redacted=True, summary=""), "", "empty"),
        (
            ContentReasoning(reasoning="ciphertext", redacted=True, summary="Summary."),
            "Summary.",
            "usable",
        ),
    ],
    ids=["absent", "empty", "whitespace", "text", "redacted", "blank-summary", "readable-summary"],
)
def test_reasoning_status(
    block: ContentReasoning | None, expected_text: str | None, status: str
) -> None:
    content: list[Content] = [block] if block is not None else []
    content.append(ContentText(text="Answer."))
    response = extract_response_text(ModelOutput.from_content("mockllm/target", content))
    assert response == (expected_text, "Answer.")
    assert response.reasoning_status == status


def test_offline_generation(tmp_path: Path) -> None:
    records = load_records(**PILOT_SELECTION)

    plan = "Write an original closing line."
    line = "The last chord carries us home."
    cases: dict[tuple[str, str], tuple[str | list[Content], str | None, str]] = {
        ("capability", "baseline"): (
            [ContentReasoning(reasoning=plan), ContentText(text=line)],
            plan,
            line,
        ),
        ("capability", "F8"): ("Curtain call.", None, "Curtain call."),
        ("safety", "baseline"): (
            [ContentReasoning(reasoning=""), ContentText(text="I can summarize the song.")],
            "",
            "I can summarize the song.",
        ),
        ("safety", "F8"): (
            [ContentReasoning(reasoning="mock-ciphertext", redacted=True), ContentText(text="{}")],
            None,
            "{}",
        ),
    }
    outputs = []
    for record in records:
        content, _, _ = cases[record["valence"], record["condition"]]
        output = ModelOutput.from_content("mockllm/model", content)
        # Supply mock usage so token counting never has to download a tokenizer.
        output.usage = ModelUsage()
        outputs.append(output)

    [run] = eval(
        "eval_aware_bench/eval_aware_bench",
        task_args=PILOT_SELECTION,
        model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
        max_samples=1,  # Consume the predetermined outputs in dataset order.
        epochs=1,
        score=False,
        display="none",
        log_dir=str(tmp_path),
        log_format="eval",
    )
    assert run.status == "success", run.error
    saved = read_eval_log(run.location, resolve_attachments=True)
    assert saved.samples is not None
    assert len(saved.samples) == len(cases)
    by_id = {sample.id: sample for sample in saved.samples}
    for record in records:
        sample = by_id[record["id"]]
        assert not sample.scores
        [call] = [event for event in sample.events if isinstance(event, ModelEvent)]
        assert call.model == "mockllm/model"
        assert call.tools == []
        assert [(m.role, m.content) for m in call.input] == [
            ("system", record["system_prompt"]),
            ("user", record["user_prompt"]),
        ]

        content, reasoning, answer = cases[record["valence"], record["condition"]]
        assert [m.role for m in sample.messages] == ["system", "user", "assistant"]
        assert sample.messages[-1].content == content
        assert sample.output.message.content == content
        assert extract_response_text(sample.output) == (reasoning, answer)


def test_readable_blocks_and_summaries_preserve_output() -> None:
    output = ModelOutput.from_content(
        "mockllm/model",
        [
            ContentReasoning(reasoning="mock-ciphertext", redacted=True, summary="Public summary."),
            ContentText(text="First line."),
            ContentReasoning(reasoning="private replay payload", redacted=True),
            ContentReasoning(reasoning="Second thought.", signature="mock-signature"),
            ContentReasoning(reasoning="", summary="Summary only."),
            ContentText(text="Last line."),
        ],
    )
    original = output.model_copy(deep=True)
    assert extract_response_text(output) == (
        "Public summary.\nSecond thought.\nSummary only.",
        "First line.\nLast line.",
    )
    assert output == original


def test_completion_without_choices_is_answer_only() -> None:
    output = ModelOutput(completion="Plain answer.")
    assert extract_response_text(output) == (None, "Plain answer.")
