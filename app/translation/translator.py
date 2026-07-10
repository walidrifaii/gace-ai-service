"""Translation engine abstraction + the Argos Translate implementation.

`BaseTranslator` is the seam that lets us swap Argos Translate for NLLB-200,
LibreTranslate, Google Translate, or DeepL later without touching routes.py
or models.py: any replacement just needs to implement `translate()` and be
handed to `TranslationService`.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .cache import CachedTranslation, TranslationCache, make_cache_key
from .config import TranslationSettings

logger = logging.getLogger("app.translation.translator")

# ISO 639-1 code -> display name, for the /languages endpoint and validation.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "tr": "Turkish",
    "de": "German",
}

PIVOT_LANGUAGE = "en"


class TranslationError(Exception):
    """Raised for any translation failure the API should surface to the client."""


@dataclass(frozen=True)
class TranslationResult:
    translated_text: str
    detected_source: str


class BaseTranslator(ABC):
    """Contract every translation backend (Argos, NLLB, LibreTranslate, ...) must satisfy."""

    @abstractmethod
    def warm_up(self) -> None:
        """Install/load whatever is needed so the first real request is fast."""

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        """Translate `text` from `source` (or 'auto') to `target`."""

    @abstractmethod
    def supported_languages(self) -> dict[str, str]:
        """Return {code: display_name} for languages this backend supports."""


class ArgosTranslator(BaseTranslator):
    """`BaseTranslator` backed by the free, offline Argos Translate engine."""

    def __init__(self, settings: TranslationSettings) -> None:
        self._settings = settings
        self._supported = {
            code: LANGUAGE_NAMES[code]
            for code in settings.supported_language_list
            if code in LANGUAGE_NAMES
        }
        # Guards Argos package-index/install calls, which mutate shared
        # on-disk state and are not safe to run concurrently. Translation
        # inference itself is not held under this lock so requests for
        # already-installed language pairs stay fast and concurrent.
        self._install_lock = threading.RLock()
        self._installed_pairs: set[tuple[str, str]] = set()
        self._warmed_up = False

        self._configure_package_dir()

    def _configure_package_dir(self) -> None:
        """Point Argos at our project-local `languages/` folder before it's imported.

        Argos Translate resolves its package storage directory the first time
        `argostranslate.settings` is imported, reading these env vars. Setting
        them here (before the first `import argostranslate`) keeps downloaded
        packages inside this folder so they survive as a simple cache instead
        of scattering into the OS user-data directory.
        """
        packages_dir = str(self._settings.languages_dir)
        self._settings.languages_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("ARGOS_TRANSLATE_PACKAGES_DIR", packages_dir)
        os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")

    def warm_up(self) -> None:
        """Install en<->X packages for every supported language.

        Argos translations pivot through English automatically when a direct
        package isn't installed, so installing English<->each language is
        enough to cover every pair among the supported languages.
        """
        if self._warmed_up:
            return
        with self._install_lock:
            if self._warmed_up:
                return
            for code in self._supported:
                if code == PIVOT_LANGUAGE:
                    continue
                self._ensure_package_installed(PIVOT_LANGUAGE, code)
                self._ensure_package_installed(code, PIVOT_LANGUAGE)
            self._warmed_up = True
        logger.info("Argos warm-up complete for languages: %s", sorted(self._supported))

    def supported_languages(self) -> dict[str, str]:
        return dict(self._supported)

    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        target = target.lower()
        if target not in self._supported:
            raise TranslationError(f"Unsupported target language: '{target}'")

        detected_source = source.lower()
        if detected_source == "auto":
            detected_source = self._detect_language(text)

        if detected_source not in self._supported:
            raise TranslationError(f"Unsupported or undetectable source language: '{detected_source}'")

        if detected_source == target:
            return TranslationResult(translated_text=text, detected_source=detected_source)

        self._ensure_pair_translatable(detected_source, target)

        try:
            translated_text = self._run_translation(text, detected_source, target)
        except TranslationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive, engine-specific failures
            logger.exception("Argos translation failed (%s -> %s)", detected_source, target)
            raise TranslationError("Translation failed") from exc

        return TranslationResult(translated_text=translated_text, detected_source=detected_source)

    # -- internals ---------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0  # deterministic detection
            code = detect(text).lower()
        except Exception as exc:
            raise TranslationError("Could not detect source language") from exc

        # langdetect occasionally returns region-qualified codes (e.g. "zh-cn").
        code = code.split("-")[0]
        return code

    def _ensure_pair_translatable(self, source: str, target: str) -> None:
        pair = (source, target)
        if pair in self._installed_pairs:
            return
        with self._install_lock:
            if pair in self._installed_pairs:
                return
            if source != PIVOT_LANGUAGE:
                self._ensure_package_installed(source, PIVOT_LANGUAGE)
            if target != PIVOT_LANGUAGE:
                self._ensure_package_installed(PIVOT_LANGUAGE, target)
            if source != PIVOT_LANGUAGE and target != PIVOT_LANGUAGE:
                # Direct package may also exist and is preferred by Argos when present.
                self._ensure_package_installed(source, target, required=False)
            self._installed_pairs.add(pair)

    def _ensure_package_installed(self, from_code: str, to_code: str, required: bool = True) -> None:
        import argostranslate.package

        installed = argostranslate.package.get_installed_packages()
        if any(p.from_code == from_code and p.to_code == to_code for p in installed):
            return

        logger.info("Downloading Argos package %s -> %s", from_code, to_code)
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        match = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if match is None:
            if required:
                raise TranslationError(f"No Argos package available for {from_code} -> {to_code}")
            return

        download_path = match.download()
        argostranslate.package.install_from_path(download_path)
        logger.info("Installed Argos package %s -> %s", from_code, to_code)

    def _run_translation(self, text: str, source: str, target: str) -> str:
        import argostranslate.translate

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == source), None)
        to_lang = next((lang for lang in installed_languages if lang.code == target), None)
        if from_lang is None or to_lang is None:
            raise TranslationError(f"Language pair not installed: {source} -> {target}")

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            raise TranslationError(f"No translation path from {source} to {target}")

        return translation.translate(text)


class TranslationService:
    """Orchestrates a `BaseTranslator` backend with the translated-text cache.

    This is the single entry point routes.py depends on. Because it only
    talks to `BaseTranslator`, swapping Argos for another engine is a
    one-line change (construct a different translator in main.py) with no
    changes to this class or to the API layer.
    """

    def __init__(self, translator: BaseTranslator, cache: TranslationCache) -> None:
        self._translator = translator
        self._cache = cache

    def warm_up(self) -> None:
        self._translator.warm_up()

    def supported_languages(self) -> dict[str, str]:
        return self._translator.supported_languages()

    def translate(self, text: str, source: str, target: str) -> tuple[TranslationResult, bool]:
        """Returns (result, was_cached)."""
        cache_key = make_cache_key(text, source, target)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return (
                TranslationResult(
                    translated_text=cached.translated_text,
                    detected_source=cached.detected_source,
                ),
                True,
            )

        result = self._translator.translate(text, source, target)
        self._cache.set(
            cache_key,
            CachedTranslation(
                translated_text=result.translated_text,
                detected_source=result.detected_source,
            ),
        )
        return result, False
