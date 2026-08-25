"""Unicode NFC normalization and text cleaning module."""

import re
import unicodedata


class TextNormalizer:
    """Cleans raw text and standardizes Unicode to NFC."""

    # Regex patterns for cleaning
    HTML_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    EXTRA_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalizes Unicode encoding to NFC and removes artifacts."""
        if not text:
            return ""

        # Step 1: Standardize to Unicode NFC (canonical decomposition followed by canonical composition)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Strip HTML tags
        text = cls.HTML_PATTERN.sub(" ", text)

        # Step 3: Clean redundant whitespaces while preserving paragraph newlines
        text = cls.EXTRA_WHITESPACE.sub(" ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = cls.MULTIPLE_NEWLINES.sub("\n\n", text)

        return text.strip()
