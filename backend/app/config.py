from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_api_keys: str = ""
    llm_model: str = "claude-opus-5-5"
    stt_base_url: str = ""
    stt_api_key: str = ""
    stt_model: str = ""
    max_audio_mb: int = 25

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.app_api_keys.split(",") if k.strip()]

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
