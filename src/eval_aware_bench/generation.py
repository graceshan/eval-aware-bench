"""Read the response channels from an Inspect generation."""

from typing import NamedTuple

from inspect_ai.model import ContentReasoning, ModelOutput


class ResponseText(NamedTuple):
    reasoning: str | None
    answer: str


def extract_response_text(output: ModelOutput) -> ResponseText:
    """Read the first choice, using None for unavailable reasoning and "" for an empty block."""
    if output.empty:
        return ResponseText(None, output.completion)

    message = output.message
    reasoning = []
    for block in message.content_list:
        if isinstance(block, ContentReasoning):
            # Providers may store ciphertext in reasoning and readable text in summary.
            text = block.summary if block.redacted else (block.reasoning or block.summary or "")
            if text is not None:
                reasoning.append(text)

    return ResponseText("\n".join(reasoning) if reasoning else None, message.text)
