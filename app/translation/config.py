"""Runtime configuration for the translation feature.

Mounted inside the main face-ai-service app (same process, same port), so
this only configures translation-specific behavior — host/port/CORS are
owned by app/main.py.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class TranslationSettings(BaseSettings):
    # Optional shared-secret header check. Leave empty to disable.
    translate_api_key: str = ""

    # Comma-separated list of codes. Argos will download+install the packages
    # needed to translate between English and each of these languages so that
    # any pair among them can be reached (Argos pivots through English).
    supported_languages: str = "en,ar,fr,es,tr,de"
    default_target_language: str = "en"

    # Where Argos Translate stores downloaded/installed language packages.
    languages_dir: Path = BASE_DIR / "languages"

    # In-memory translated-text cache (see cache.py). 0 disables caching.
    translation_cache_max_size: int = 5000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def supported_language_list(self) -> list[str]:
        return [code.strip().lower() for code in self.supported_languages.split(",") if code.strip()]


settings = TranslationSettings()
