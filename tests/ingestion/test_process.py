from __future__ import annotations

from src.ingestion.preprocess import (
    clean_text,
    extract_text_from_html,
    normalize_whitespace,
    remove_common_boilerplate,
)


class TestWhitespaceNormalization:
    def test_normalize_whitespace(self):
        raw = "Hello   world\n\n\nThis\tis   text"
        assert normalize_whitespace(raw) == "Hello world\nThis is text"

    def test_normalize_whitespace_empty(self):
        assert normalize_whitespace("") == ""


class TestBoilerplateRemoval:
    def test_remove_common_boilerplate(self):
        raw = (
            "WELCOME MESSAGE\n"
            "Skip to content\n"
            "This portal provides information relating to employment.\n"
            "Contact us\n"
        )
        cleaned = remove_common_boilerplate(raw)
        assert "Skip to content" not in cleaned
        assert "This portal provides information" in cleaned

    def test_remove_common_boilerplate_keeps_substantive_text(self):
        raw = "The Employment Ordinance is the main piece of legislation."
        cleaned = remove_common_boilerplate(raw)
        assert cleaned == raw


class TestHtmlExtraction:
    def test_extract_text_from_html_basic(self):
        html = b"""
        <html>
          <body>
            <nav>Home | About</nav>
            <main>
              <h1>Employment Ordinance</h1>
              <p>The Employment Ordinance is the main piece of legislation.</p>
            </main>
          </body>
        </html>
        """
        text = extract_text_from_html(html)
        assert "Employment Ordinance" in text
        assert "main piece of legislation" in text
        assert "Home | About" not in text

    def test_clean_text_pipeline(self):
        raw = "  Hello   world \n\n Skip to content \n Main text here. "
        cleaned = clean_text(raw)
        assert "Skip to content" not in cleaned
        assert cleaned.startswith("Hello world")