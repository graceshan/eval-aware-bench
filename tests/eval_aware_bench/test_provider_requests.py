"""Exercise the packaged run configs through Inspect's CLI and the installed SDKs."""

import json
import socket
from pathlib import Path
from typing import Any

import httpx2
import pytest

from eval_aware_bench.generation import ResponseText
from eval_aware_bench.judge import format_judge_request
from tests.eval_aware_bench.test_eval_aware_bench import verdict
from tests.eval_aware_bench.test_run_config import run_config

ROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_CHAT = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"
CLAUDE = "claude-sonnet-4-5-20250929"
OSS = "openai/gpt-oss-120b"
RESPONSE = ResponseText("I can write an original song.", "The last chord carries us home.")


@pytest.fixture
def requests_sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    # Exercise API-key auth even on machines configured for Anthropic OAuth.
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    calls = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url.copy_with(query=None))
        body = json.loads(request.content)
        calls.append((url, body))
        assert request.method == "POST"
        model = body["model"]
        if request.url.host == "api.anthropic.com":
            assert request.headers["anthropic-version"] == "2023-06-01"
        if url == ANTHROPIC_MESSAGES + "/count_tokens":
            result = {"input_tokens": 8}
        elif url == ANTHROPIC_MESSAGES:
            assert "interleaved-thinking-2025-05-14" in request.headers["anthropic-beta"]
            result = {
                "id": "msg_test",
                "model": model,
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": RESPONSE.reasoning, "signature": "mock"},
                    {"type": "text", "text": RESPONSE.answer},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        elif url == OPENAI_RESPONSES:
            result = {
                "id": "resp_test",
                "model": model,
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "rs_test",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": RESPONSE.reasoning}],
                    },
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": RESPONSE.answer, "annotations": []}
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            }
        elif url in (ROUTER_CHAT, OPENAI_CHAT):
            if model == OSS:
                message = {
                    "role": "assistant",
                    "content": RESPONSE.answer,
                    "reasoning": RESPONSE.reasoning,
                }
            else:
                message = {"role": "assistant", "content": verdict("capability", "Yes")}
            result = {
                "id": "chatcmpl-test",
                "model": model,
                "object": "chat.completion",
                "created": 0,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        else:
            pytest.fail(f"Unexpected SDK endpoint {url}")
        return httpx2.Response(200, json=result)

    # Catch SDK-created clients too, including separate token counters and probes.
    transport = httpx2.MockTransport(respond)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", transport.handle_request)
    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", transport.handle_async_request
    )

    def block_connection(*args: object) -> None:
        pytest.fail("Network access outside the mock HTTP transport")

    monkeypatch.setattr(socket.socket, "connect", block_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", block_connection)
    return calls


@pytest.mark.parametrize(
    ("name", "target", "target_url"),
    [
        ("full-openrouter-gpt-oss-120b.yaml", "openrouter/" + OSS, ROUTER_CHAT),
        ("full-openai-gpt-5.yaml", "openai/gpt-5", OPENAI_RESPONSES),
        ("full-anthropic-sonnet-4-5.yaml", "anthropic/" + CLAUDE, ANTHROPIC_MESSAGES),
    ],
    ids=["openrouter", "openai", "anthropic"],
)
def test_full_provider_config(
    name: str,
    target: str,
    target_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requests_sent: list[tuple[str, dict[str, Any]]],
) -> None:
    plan = run_config(name, [], tmp_path / "plan", monkeypatch)
    assert plan.eval.task == "eval_aware_bench/eval_aware_bench"
    assert plan.eval.task_args == dict.fromkeys(("pair_ids", "valences", "conditions"), "all")
    assert plan.eval.model == target
    config = plan.eval.config
    assert config.sample_id is None
    assert config.limit is None
    assert config.epochs == 3
    assert config.epochs_reducer == []
    assert plan.eval.dataset.sample_ids is not None
    assert len(plan.eval.dataset.sample_ids) == len(set(plan.eval.dataset.sample_ids)) == 1800
    assert config.retry_on_error == 0
    assert config.fail_on_error is False
    assert config.score_on_error is False
    assert plan.plan.config.cache is False
    assert plan.plan.config.max_retries == 1
    assert plan.plan.config.max_connections == 10
    assert plan.eval.model_args["max_retries"] == 0
    assert plan.eval.model_roles is not None
    grader = plan.eval.model_roles["grader"]
    assert not isinstance(grader, list)
    assert grader.model == (
        "openrouter/openai/gpt-5" if target_url == ROUTER_CHAT else "openai/gpt-5"
    )
    assert grader.config.cache is False
    assert grader.config.max_retries == 1
    assert grader.config.max_connections == 20
    assert grader.args["max_retries"] == 0
    assert requests_sent == []

    # One real capability input for three mocked epochs exercises output-cache independence.
    monkeypatch.delenv("INSPECT_DISABLE_MODEL_API")
    run = run_config(
        name,
        ["--sample-id", "lyrics_purple_rain_capability_F8", "--max-samples", "1"],
        tmp_path / "wire",
        monkeypatch,
        run_samples=True,
    )
    assert run.status == "success", run.error
    assert run.results is not None
    assert run.results.completed_samples == 3
    assert all(r.scored_samples == 3 for r in run.results.scores)

    ref = json.loads(
        Path(__file__).with_name("lyrics_purple_rain_reference.json").read_text(encoding="utf-8")
    )
    system = ref["system_prompt"]
    user = ref["user_prompts"]["capability"]["F8"]
    [rubric] = format_judge_request(user, RESPONSE)
    user_message = {"role": "user", "content": user}
    target_bodies = {
        ROUTER_CHAT: {
            "model": OSS,
            "messages": [{"role": "system", "content": system}, user_message],
            "temperature": 0.3,
            "max_tokens": 4096,
            "reasoning": {"effort": "high"},
        },
        OPENAI_RESPONSES: {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}
                for role, text in [("developer", system), ("user", user)]
            ],
            "reasoning": {"effort": "high", "summary": "detailed"},
            "max_output_tokens": 4096,
            "include": [],
        },
        ANTHROPIC_MESSAGES: {
            "model": CLAUDE,
            "system": [{"type": "text", "text": system}],
            "messages": [user_message],
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 2457, "display": "summarized"},
            "stream": False,
            "tools": [],
        },
    }
    if target_url == ROUTER_CHAT:
        grader_url, grader_id, role = ROUTER_CHAT, "openai/gpt-5", "system"
    else:
        grader_url, grader_id, role = OPENAI_CHAT, "gpt-5", "developer"
    grader_body = {"model": grader_id, "messages": [{"role": role, "content": rubric.text}]}
    generations = [
        (url, body) for url, body in requests_sent if url != ANTHROPIC_MESSAGES + "/count_tokens"
    ]
    assert generations == [(target_url, target_bodies[target_url]), (grader_url, grader_body)] * 3
