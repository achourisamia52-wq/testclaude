"""Claude calls: transcript in, structured result out."""

from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from . import prompts
from .errors import UpstreamError
from .schemas import ComposeResult, GistResult

T = TypeVar("T", bound=BaseModel)


class Writer(Protocol):
    async def gist(self, transcript: str) -> GistResult: ...

    async def compose(self, transcript: str, tone: str, language: str) -> ComposeResult: ...


class ClaudeWriter:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def gist(self, transcript: str) -> GistResult:
        result = await self._parse(prompts.GIST_SYSTEM, prompts.gist_user(transcript), GistResult)
        result.replies = [r.strip() for r in result.replies if r.strip()][:3]
        return result

    async def compose(self, transcript: str, tone: str, language: str) -> ComposeResult:
        return await self._parse(
            prompts.COMPOSE_SYSTEM,
            prompts.compose_user(transcript, tone, language),
            ComposeResult,
        )

    async def _parse(self, system: str, user: str, schema: type[T]) -> T:
        try:
            response = await self._client.beta.messages.parse(
                model=self._model,
                max_tokens=16000,
                # Users wait on this inside a chat app; these are short rewrites.
                output_config={"effort": "low"},
                # If a safety classifier declines, retry once on a fallback model
                # inside the same call instead of failing the request.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.RateLimitError as exc:
            raise UpstreamError("The service is busy, try again shortly.", 503) from exc
        except anthropic.APIStatusError as exc:
            status = 503 if exc.status_code >= 500 else 502
            raise UpstreamError("The language model request failed.", status) from exc
        except anthropic.APIConnectionError as exc:
            raise UpstreamError("Could not reach the language model.") from exc

        if response.stop_reason == "refusal":
            raise UpstreamError("This voice note could not be processed.", 422)
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise UpstreamError("The language model returned an incomplete answer.")
        return response.parsed_output
