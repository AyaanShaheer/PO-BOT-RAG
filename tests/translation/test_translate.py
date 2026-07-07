from __future__ import annotations

from src.translation.translate import TranslationResult, should_translate


def test_should_translate_english_to_english():
    assert should_translate("en", "en") is False


def test_should_translate_tagalog_to_english():
    assert should_translate("tl", "en") is True


def test_translation_result_defaults():
    result = TranslationResult(
        source_language="tl",
        target_language="en",
        original_text="Kamusta",
        translated_text="Hello",
        used_translation=True,
    )
    assert result.used_translation is True
    assert result.source_language == "tl"