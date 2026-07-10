"""Runtime configuration for the translation service.

All values are overridable via environment variables (or a local .env file),
so the same code runs unmodified in dev, staging, and production containers.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002

    # Optional shared-secret header check, mirrors face-ai-service's pattern.
    # Leave empty to disable auth (e.g. when only reachable on a private network).
    translate_api_key: str = ""

    # Comma-separated list of codes. Argos will download+install the packages
    # needed to translate between English and each of these languages so that
    # any pair among them can be reached (Argos pivots through English).
    supported_languages: str = "en,ar,fr,es,tr,de"
    default_target_language: str = "en"

    # Where Argos Translate stores downloaded/installed language packages.
    # Keeping this inside the project makes "cache the language packages"
    # explicit and survives container rebuilds when the folder is a volume.
    languages_dir: Path = BASE_DIR / "languages"

    # In-memory translated-text cache (see cache.py). 0 disables caching.
    translation_cache_max_size: int = 5000

    # CORS - the Flutter app calls this service directly, so keep it open by
    # default; tighten via env var in production if the host is public.
    allowed_origins: str = "*"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def supported_language_list(self) -> list[str]:
        return [code.strip().lower() for code in self.supported_languages.split(",") if code.strip()]

    @property
    def allowed_origin_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
