import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.errors import UpstreamError
from app.main import app, get_transcriber, get_writer
from app.schemas import ActionItem, ComposeResult, GistResult

AUTH = {"Authorization": "Bearer test-key"}
AUDIO = {"audio": ("note.ogg", b"fake-audio-bytes", "audio/ogg")}


class FakeTranscriber:
    def __init__(self, text="please send the deck by five"):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio, filename, content_type):
        self.calls += 1
        return self.text


class FakeWriter:
    def __init__(self):
        self.compose_args = None

    async def gist(self, transcript):
        return GistResult(
            language="en",
            gist="Hesham wants the deck by 5.",
            action_items=[ActionItem(text="Send the deck", due="by five")],
            replies=["On it.", "Which deck?", "Will get back to you."],
        )

    async def compose(self, transcript, tone, language):
        self.compose_args = (transcript, tone, language)
        return ComposeResult(message="Hi Hesham, the deck will be ready by 5.")


@pytest.fixture
def ctx():
    transcriber, writer = FakeTranscriber(), FakeWriter()
    app.dependency_overrides[get_settings] = lambda: Settings(
        app_api_keys="test-key", max_audio_mb=1, _env_file=None
    )
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_writer] = lambda: writer
    yield TestClient(app), transcriber, writer
    app.dependency_overrides.clear()


def test_health_needs_no_auth(ctx):
    client, _, _ = ctx
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_rejects_bad_auth(ctx, headers):
    client, transcriber, _ = ctx
    assert client.post("/v1/gist", files=AUDIO, headers=headers).status_code == 401
    assert transcriber.calls == 0


def test_gist(ctx):
    client, _, _ = ctx
    body = client.post("/v1/gist", files=AUDIO, headers=AUTH).json()
    assert body["transcript"] == "please send the deck by five"
    assert body["gist"] == "Hesham wants the deck by 5."
    assert body["action_items"] == [{"text": "Send the deck", "due": "by five"}]
    assert len(body["replies"]) == 3


def test_gist_rejects_oversized_audio(ctx):
    client, transcriber, _ = ctx
    big = {"audio": ("note.ogg", b"x" * (1024 * 1024 + 1), "audio/ogg")}
    assert client.post("/v1/gist", files=big, headers=AUTH).status_code == 413
    assert transcriber.calls == 0


def test_gist_rejects_empty_audio(ctx):
    client, _, _ = ctx
    empty = {"audio": ("note.ogg", b"", "audio/ogg")}
    assert client.post("/v1/gist", files=empty, headers=AUTH).status_code == 400


def test_gist_when_no_speech(ctx):
    client, transcriber, _ = ctx
    transcriber.text = ""
    assert client.post("/v1/gist", files=AUDIO, headers=AUTH).status_code == 422


def test_compose_from_audio(ctx):
    client, _, writer = ctx
    r = client.post("/v1/compose", files=AUDIO, data={"tone": "formal"}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["message"] == "Hi Hesham, the deck will be ready by 5."
    assert writer.compose_args == ("please send the deck by five", "formal", "auto")


def test_compose_from_transcript_skips_transcription(ctx):
    client, transcriber, writer = ctx
    r = client.post(
        "/v1/compose", data={"transcript": "deck by five", "language": "ar"}, headers=AUTH
    )
    assert r.status_code == 200
    assert transcriber.calls == 0
    assert writer.compose_args == ("deck by five", "casual", "ar")


def test_compose_needs_exactly_one_input(ctx):
    client, _, _ = ctx
    assert client.post("/v1/compose", data={"tone": "formal"}, headers=AUTH).status_code == 400
    both = client.post("/v1/compose", files=AUDIO, data={"transcript": "x"}, headers=AUTH)
    assert both.status_code == 400


def test_compose_rejects_unknown_tone(ctx):
    client, _, _ = ctx
    r = client.post("/v1/compose", data={"transcript": "x", "tone": "angry"}, headers=AUTH)
    assert r.status_code == 422


def test_upstream_error_becomes_http_error(ctx):
    client, _, writer = ctx

    async def boom(transcript):
        raise UpstreamError("The service is busy, try again shortly.", 503)

    writer.gist = boom
    r = client.post("/v1/gist", files=AUDIO, headers=AUTH)
    assert r.status_code == 503
    assert r.json() == {"detail": "The service is busy, try again shortly."}
