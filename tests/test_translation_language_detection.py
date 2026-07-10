"""Reproduces the "Unsupported or undetectable source language: 'xx'" bug
report and proves the fix in app/translation/translator.py resolves it.

Run directly (no pytest needed):
    .venv/Scripts/python tests/test_translation_language_detection.py

What it does:
  1. Runs the OLD detection strategy (single `langdetect.detect()` guess,
     hard-fail if it's not in our supported set) against real short chat
     messages, to show exactly how/why it raised errors like the one
     reported ("... source language: 'fi'").
  2. Runs the CURRENT `ArgosTranslator._detect_language` against the same
     inputs and asserts none of them raise.

Only needs `langdetect` installed (no argostranslate/ctranslate2 download
required — detection doesn't touch the translation engine itself).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.translation.config import TranslationSettings  # noqa: E402
from app.translation.translator import ArgosTranslator, TranslationError  # noqa: E402

# Real, short, casual chat messages -- exactly the kind of input that
# breaks single-guess language ID.
SAMPLE_MESSAGES = [
    "hi",
    "ok",
    "yes",
    "no",
    "good",
    "thanks",
    "lol",
    "nice",
    "wow",
    "how are you",
    "good morning",
    "good night",
    "salam",
    "shukran",
    "ok cool",
    "sure thing",
    "مرحبا كيف حالك",  # Arabic: "hello, how are you"
]

SUPPORTED = set(TranslationSettings().supported_language_list)


def old_buggy_detect(text: str) -> str:
    """The original (pre-fix) detection logic: single top guess, hard-fail
    if it isn't in our supported set. This is what production was running
    when the "unsupported or undetectable source language: 'fi'" error was
    reported.
    """
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    code = detect(text).lower().split("-")[0]
    if code not in SUPPORTED:
        raise TranslationError(f"Unsupported or undetectable source language: '{code}'")
    return code


def run() -> None:
    print(f"Supported languages: {sorted(SUPPORTED)}\n")

    print("=" * 70)
    print("STEP 1 - reproducing the bug with the OLD detection logic")
    print("=" * 70)
    old_failures = []
    for text in SAMPLE_MESSAGES:
        try:
            code = old_buggy_detect(text)
            print(f"  OK      {text!r:24} -> {code!r}")
        except TranslationError as exc:
            old_failures.append((text, str(exc)))
            print(f"  FAILED  {text!r:24} -> {exc}")

    print(
        f"\n{len(old_failures)}/{len(SAMPLE_MESSAGES)} short chat messages "
        "failed under the old logic -- this is the exact class of error "
        "that was reported.\n"
    )

    print("=" * 70)
    print("STEP 2 - same inputs against the CURRENT (fixed) detection logic")
    print("=" * 70)
    translator = ArgosTranslator(TranslationSettings())
    new_failures = []
    for text in SAMPLE_MESSAGES:
        try:
            code = translator._detect_language(text)  # noqa: SLF001 - testing internals on purpose
            print(f"  OK      {text!r:24} -> {code!r}")
        except TranslationError as exc:
            new_failures.append((text, str(exc)))
            print(f"  FAILED  {text!r:24} -> {exc}")

    print()
    assert not new_failures, f"Fixed detection still failed on: {new_failures}"
    print(f"All {len(SAMPLE_MESSAGES)} messages resolved to a supported language. Fix verified.")


if __name__ == "__main__":
    run()
