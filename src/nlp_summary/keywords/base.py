"""Abstract base class for keyword extractors."""

from abc import ABC, abstractmethod
from typing import List, Optional
from nlp_summary.models.preprocessing import ProcessedDocument
from nlp_summary.models.keywords import ExtractedKeyword


class KeywordExtractorBase(ABC):
    """Base strategy class for all keyword extraction algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract_keywords(
        self,
        doc: ProcessedDocument,
        top_k: int = 10,
        ngram_range: tuple = (1, 3)
    ) -> List[ExtractedKeyword]:
        """Extracts top_k keywords/keyphrases from a ProcessedDocument."""
        pass
