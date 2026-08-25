"""Factory registry for keyword extractors."""

from typing import Dict, Type
from nlp_summary.keywords.base import KeywordExtractorBase
from nlp_summary.keywords.tfidf import TFIDFExtractor
from nlp_summary.keywords.textrank import TextRankExtractor
from nlp_summary.keywords.yake_extractor import YAKEExtractor
from nlp_summary.keywords.keybert_extractor import KeyBERTExtractor


class ExtractorRegistry:
    """Registry providing instances of keyword extractors."""

    _EXTRACTORS: Dict[str, Type[KeywordExtractorBase]] = {
        "tfidf": TFIDFExtractor,
        "textrank": TextRankExtractor,
        "yake": YAKEExtractor,
        "keybert": KeyBERTExtractor,
    }

    @classmethod
    def get_extractor(cls, name: str = "textrank", **kwargs) -> KeywordExtractorBase:
        """Instantiates and returns the requested extractor."""
        clean_name = name.lower().strip()
        if clean_name not in cls._EXTRACTORS:
            clean_name = "textrank"

        extractor_cls = cls._EXTRACTORS[clean_name]
        return extractor_cls(**kwargs)
