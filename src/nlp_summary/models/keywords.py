"""Data models for keyword extraction."""

from typing import List
from pydantic import BaseModel, Field


class ExtractedKeyword(BaseModel):
    """Represents an extracted keyword or keyphrase."""
    keyword: str
    normalized_score: float = Field(ge=0.0, le=1.0, description="Confidence/importance score normalized to [0, 1]")
    raw_score: float = Field(description="Raw score from the specific algorithm")
    frequency: int = Field(default=1, description="Occurrences in document")
    ngram_length: int = Field(default=1, description="Number of constituent words/syllables")
    algorithm: str = Field(description="Name of extraction algorithm (tfidf, textrank, yake, keybert)")


class KeywordExtractionResult(BaseModel):
    """Result object of a keyword extraction run."""
    algorithm: str
    top_k: int
    execution_time_ms: float
    keywords: List[ExtractedKeyword]
