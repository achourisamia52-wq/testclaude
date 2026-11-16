"""Speech-to-text behind a small interface so the provider can be swapped."""

from typing import Protocol

import httpx

from .errors import UpstreamError


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, filename: str, content_type: str) -> str: ...


class OpenAICompatibleTranscriber:
    """Any provider exposing POST {base_url}/audio/transcriptions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not (base_url and api_key and model):
            raise UpstreamError(
                "Speech-to-text is not configured (STT_BASE_URL, STT_API_KEY, STT_MODEL).",
                status_code=503,
            )
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=120)

    async def transcribe(self, audio: bytes, filename: str, content_type: str) -> str:
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                data={"model": self._model, "response_format": "json"},
                files={"file": (filename, audio, content_type)},
            )
        except httpx.HTTPError as exc:
            raise UpstreamError("Could not reach the speech-to-text provider.") from exc

        if response.status_code == 429:
            raise UpstreamError("Speech-to-text is rate limited, try again shortly.", 503)
        if response.status_code >= 400:
            raise UpstreamError(
                f"Speech-to-text failed with status {response.status_code}."
            )
        return str(response.json().get("text", "")).strip()
