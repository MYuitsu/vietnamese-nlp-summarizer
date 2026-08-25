"""Data models for text preprocessing."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    """Represents an individual token or compound word."""
    text: str
    lemma: Optional[str] = None
    pos_tag: Optional[str] = None
    is_stopword: bool = False
    is_compound: bool = False
    start_char: int = 0
    end_char: int = 0


class Sentence(BaseModel):
    """Represents a segmented sentence with metadata."""
    index: int
    raw_text: str
    cleaned_text: str
    tokens: List[Token] = Field(default_factory=list)
    start_char: int = 0
    end_char: int = 0

    @property
    def words(self) -> List[str]:
        """Returns non-stopword token texts or all token texts if all filtered."""
        content_tokens = [t.text for t in self.tokens if not t.is_stopword]
        return content_tokens if content_tokens else [t.text for t in self.tokens]


class ProcessedDocument(BaseModel):
    """Container for the preprocessed document."""
    original_text: str
    cleaned_text: str
    language: str = "vi"
    sentences: List[Sentence] = Field(default_factory=list)
    total_tokens: int = 0

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)
