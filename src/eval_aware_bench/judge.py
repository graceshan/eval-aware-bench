"""Format the EvalAwareBench judge request and parse its three judgments."""

import json
from importlib.resources import files
from typing import Annotated, Literal, Self

from inspect_ai.model import ChatMessage, ChatMessageSystem
from pydantic import BaseModel, Field, model_validator

from eval_aware_bench.generation import ResponseText


def format_judge_request(user_prompt: str, response: ResponseText) -> list[ChatMessage]:
    """Render one upstream system message, preserving the user prompt verbatim."""
    template = files("eval_aware_bench") / "data" / "judgePromptEvidenceBoth.txt"
    content = template.read_text(encoding="utf-8").format(
        task=user_prompt,
        reasoning=(response.reasoning or "").strip(),
        answer=response.answer.strip(),
    )
    return [ChatMessageSystem(content=content)]


# Require text while preserving the original whitespace in evidence and explanations.
NonBlankText = Annotated[str, Field(pattern=r"\S")]


class Judgment(BaseModel, strict=True, extra="forbid"):
    conclusion: Literal["Yes", "No"]
    evidence: Annotated[list[NonBlankText], Field(min_length=1)]
    reasoning: NonBlankText


class AwarenessJudgment(Judgment):
    awareness_type: Literal["safety", "capability", "N/A"]

    @model_validator(mode="after")
    def check_awareness_type(self) -> Self:
        if (self.conclusion == "No") != (self.awareness_type == "N/A"):
            raise ValueError("awareness_type must be N/A exactly when model_awareness is No")
        return self


class JudgeVerdict(BaseModel, strict=True, extra="forbid"):
    model_awareness: AwarenessJudgment
    hypothetical_recognition: Judgment
    task_performance: Judgment


class VerdictParseError(ValueError):
    """The judge output is malformed, ambiguous, or inconsistent with the rubric."""


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate judge verdict field {key!r}")
        fields[key] = value
    return fields


def parse_judge_verdict(raw: str) -> JudgeVerdict:
    """Parse one JSON verdict or raise VerdictParseError with the reason it failed."""
    try:
        fields = json.loads(raw.strip(), object_pairs_hook=_unique_fields)
        return JudgeVerdict.model_validate(fields)
    except (ValueError, RecursionError) as exc:
        raise VerdictParseError(str(exc)) from exc
