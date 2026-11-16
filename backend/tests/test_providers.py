"""Provider adapters against a mocked HTTP transport (no network)."""

import json

import anthropic
import httpx
import httpx2
import pytest

from app.errors import UpstreamError
from app.llm import ClaudeWriter
from app.stt import OpenAICompatibleTranscriber


def claude_reply(payload: dict, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def writer_with(handler) -> ClaudeWriter:
    client = anthropic.AsyncAnthropic(
        api_key="test",
        base_url="https://claude.test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeWriter(client, "claude-opus-5-5")


async def test_gist_request_shape_and_parsing():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx2.Response(
            200,
            json=claude_reply(
                {
                    "language": "ar",
                    "gist": "g",
                    "action_items": [{"text": "t", "due": None}],
                    "replies": ["a", " b ", "", "c", "d"],
                }
            ),
        )

    result = await writer_with(handler).gist("hello")

    assert result.replies == ["a", "b", "c"]
    assert result.action_items[0].due is None
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "<transcript>\nhello\n</transcript>" in body["messages"][0]["content"]
    assert "server-side-fallback-2026-07-01" in seen["beta"]


async def test_refusal_is_reported_not_parsed():
    def handler(request):
        reply = claude_reply({}, stop_reason="refusal")
        reply["content"] = []
        return httpx2.Response(200, json=reply)

    with pytest.raises(UpstreamError) as exc:
        await writer_with(handler).compose("hello", "formal", "auto")
    assert exc.value.status_code == 422


async def test_rate_limit_maps_to_503():
    def handler(request):
        return httpx2.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}
        )

    with pytest.raises(UpstreamError) as exc:
        await writer_with(handler).gist("hello")
    assert exc.value.status_code == 503


async def test_transcriber_posts_multipart_and_returns_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  مرحبا  "})

    stt = OpenAICompatibleTranscriber(
        "https://stt.test/v1/",
        "k",
        "some-model",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    text = await stt.transcribe(b"abc", "note.ogg", "audio/ogg")

    assert text == "مرحبا"
    assert seen["url"] == "https://stt.test/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer k"
    assert b"some-model" in seen["body"] and b"abc" in seen["body"]


async def test_transcriber_failure_maps_to_502():
    stt = OpenAICompatibleTranscriber(
        "https://stt.test/v1",
        "k",
        "m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    with pytest.raises(UpstreamError) as exc:
        await stt.transcribe(b"abc", "note.ogg", "audio/ogg")
    assert exc.value.status_code == 502


def test_transcriber_requires_configuration():
    with pytest.raises(UpstreamError) as exc:
        OpenAICompatibleTranscriber("", "", "")
    assert exc.value.status_code == 503
