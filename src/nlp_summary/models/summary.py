"""Data models for summary generation."""

from typing import List, Literal
from pydantic import BaseModel, Field
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.scoring import ScoredSentence


class LengthConstraint(BaseModel):
    """Budget / length constraint for generated summary."""
    mode: Literal["compression_ratio", "sentence_count", "max_words"] = "compression_ratio"
    value: float = Field(default=0.30, description="Ratio in (0, 1] or integer count")


class HighlightSpan(BaseModel):
    """Token / keyword span location for frontend highlighting."""
    keyword: str
    sentence_index: int
    start_char: int
    end_char: int


class GeneratedSummary(BaseModel):
    """Container for the generated extractive summary."""
    summary_text: str
    selected_sentences: List[ScoredSentence] = Field(default_factory=list)
    selected_keywords: List[ExtractedKeyword] = Field(default_factory=list)
    original_sentence_count: int = 0
    summary_sentence_count: int = 0
    original_word_count: int = 0
    summary_word_count: int = 0
    compression_ratio: float = 0.0
    keyword_coverage_ratio: float = 0.0
    highlight_spans: List[HighlightSpan] = Field(default_factory=list)
    highlighted_summary_html: str = ""
    algorithm_used: str = "textrank"
