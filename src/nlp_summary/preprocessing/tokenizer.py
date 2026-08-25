"""Word tokenization and POS tagging module with Underthesea integration."""

import re
from typing import List, Optional
from nlp_summary.models.preprocessing import Token
from nlp_summary.preprocessing.stopwords import StopwordFilter


class VietnameseTokenizer:
    """Tokenizes text into compound words and attaches POS tags using Underthesea."""

    def __init__(self, stopword_filter: Optional[StopwordFilter] = None):
        self.stopword_filter = stopword_filter or StopwordFilter(language="vi")
        self._underthesea_available = False
        try:
            from underthesea import pos_tag, word_tokenize
            self._pos_tag = pos_tag
            self._word_tokenize = word_tokenize
            self._underthesea_available = True
        except ImportError:
            self._pos_tag = None
            self._word_tokenize = None

    def tokenize_sentence(self, sentence_text: str) -> List[Token]:
        """Tokenizes a single sentence into a list of Token objects."""
        if not sentence_text or not sentence_text.strip():
            return []

        tokens: List[Token] = []

        if self._underthesea_available:
            try:
                # Use underthesea POS tagging which performs word segmentation automatically
                # Returns: [('Xử lý', 'V'), ('ngôn ngữ', 'N'), ('tự nhiên', 'A')]
                pos_tuples = self._pos_tag(sentence_text)
                for word, tag in pos_tuples:
                    clean_word = word.strip()
                    if not clean_word:
                        continue
                    # Format compound words with underscore for graph & stats extractors
                    normalized_word = clean_word.replace(" ", "_")
                    is_stop = self.stopword_filter.is_stopword(normalized_word)
                    is_compound = "_" in normalized_word

                    tokens.append(
                        Token(
                            text=normalized_word,
                            lemma=normalized_word.lower(),
                            pos_tag=tag,
                            is_stopword=is_stop,
                            is_compound=is_compound,
                            start_char=0,
                            end_char=0
                        )
                    )
                return tokens
            except Exception:
                pass  # Fallback to regex tokenizer if model error occurs

        # Fallback whitespace / regex tokenizer
        words = re.findall(r"\w+|[^\w\s]", sentence_text, re.UNICODE)
        for w in words:
            clean_w = w.strip()
            if not clean_w:
                continue
            is_stop = self.stopword_filter.is_stopword(clean_w)
            tokens.append(
                Token(
                    text=clean_w,
                    lemma=clean_w.lower(),
                    pos_tag="N" if clean_w.isalpha() else "PUNCT",
                    is_stopword=is_stop,
                    is_compound=False
                )
            )

        return tokens
