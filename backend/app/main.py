import secrets
from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .errors import UpstreamError
from .llm import ClaudeWriter, Writer
from .schemas import ComposeResponse, GistResponse, Tone
from .stt import OpenAICompatibleTranscriber, Transcriber

app = FastAPI(title="Zatoona backend", version="0.1.0")

MAX_TRANSCRIPT_CHARS = 50_000


@app.exception_handler(UpstreamError)
async def upstream_error_handler(_: Request, exc: UpstreamError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@lru_cache
def _transcriber() -> Transcriber:
    s = get_settings()
    return OpenAICompatibleTranscriber(s.stt_base_url, s.stt_api_key, s.stt_model)


@lru_cache
def _writer() -> Writer:
    return ClaudeWriter(anthropic.AsyncAnthropic(), get_settings().llm_model)


def get_transcriber() -> Transcriber:
    return _transcriber()


def get_writer() -> Writer:
    return _writer()


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or not any(secrets.compare_digest(token, k) for k in settings.api_keys):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


async def read_audio(upload: UploadFile, limit: int) -> bytes:
    """Read the upload into memory only; audio is never written to disk."""
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="Audio file is too large.")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    return b"".join(chunks)


async def transcribe_upload(
    upload: UploadFile, transcriber: Transcriber, settings: Settings
) -> str:
    audio = await read_audio(upload, settings.max_audio_bytes)
    transcript = await transcriber.transcribe(
        audio,
        upload.filename or "voice-note",
        upload.content_type or "application/octet-stream",
    )
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech was detected in the audio.")
    return transcript


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/gist", response_model=GistResponse, dependencies=[Depends(require_api_key)])
async def gist(
    audio: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    writer: Annotated[Writer, Depends(get_writer)],
) -> GistResponse:
    """A voice note someone sent the user -> gist, action items, suggested replies."""
    transcript = await transcribe_upload(audio, transcriber, settings)
    result = await writer.gist(transcript)
    return GistResponse(transcript=transcript, **result.model_dump())


@app.post("/v1/compose", response_model=ComposeResponse, dependencies=[Depends(require_api_key)])
async def compose(
    settings: Annotated[Settings, Depends(get_settings)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    writer: Annotated[Writer, Depends(get_writer)],
    audio: Annotated[UploadFile | None, File()] = None,
    transcript: Annotated[str | None, Form(max_length=MAX_TRANSCRIPT_CHARS)] = None,
    tone: Annotated[Tone, Form()] = Tone.casual,
    language: Annotated[str, Form(pattern=r"^(auto|[a-z]{2})$")] = "auto",
) -> ComposeResponse:
    """The user talking -> a written message.

    Send `audio` the first time. To switch tone or language afterwards, send the
    returned `transcript` back instead, which skips the upload and transcription.
    """
    if (audio is None) == (transcript is None):
        raise HTTPException(status_code=400, detail="Send exactly one of audio or transcript.")
    if audio is not None:
        text = await transcribe_upload(audio, transcriber, settings)
    else:
        text = (transcript or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Transcript is empty.")
    result = await writer.compose(text, tone.value, language)
    return ComposeResponse(transcript=text, message=result.message)
