from enum import StrEnum

from pydantic import BaseModel, Field


class Tone(StrEnum):
    formal = "formal"
    casual = "casual"


class ActionItem(BaseModel):
    text: str = Field(description="One thing the listener is being asked to do.")
    due: str | None = Field(
        description="Deadline exactly as the speaker said it, or null if none was given."
    )


class GistResult(BaseModel):
    """What Claude returns for a voice note."""

    language: str = Field(description="Main language of the voice note, e.g. 'ar', 'en'.")
    gist: str = Field(description="What the speaker wants, in one or two sentences.")
    action_items: list[ActionItem]
    replies: list[str] = Field(description="Three short replies the listener could send.")


class GistResponse(GistResult):
    transcript: str


class ComposeResult(BaseModel):
    """What Claude returns for a dictated message."""

    message: str = Field(description="The written message, ready to send.")


class ComposeResponse(ComposeResult):
    transcript: str
