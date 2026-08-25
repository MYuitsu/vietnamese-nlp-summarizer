"""Data models for sentence scoring and ranking."""

from typing import List
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Explainable score components for a sentence."""
    keyword_score: float = Field(description="Contribution from extracted keywords density")
    position_score: float = Field(description="Contribution from sentence position/lead bias")
    title_match_score: float = Field(description="Contribution from title/heading overlap")
    length_penalty: float = Field(description="Penalty applied for overly short/long sentence")
    final_score: float = Field(description="Final linear combination score")


class ScoredSentence(BaseModel):
    """Sentence enriched with salience scores and keyword containment."""
    index: int
    raw_text: str
    cleaned_text: str
    score: float
    breakdown: ScoreBreakdown
    contained_keywords: List[str] = Field(default_factory=list)


class ScoringWeightsConfig(BaseModel):
    """Configuration for sentence scoring weights (sums to 1.0 ideally)."""
    weight_keyword: float = 0.50
    weight_position: float = 0.25
    weight_title: float = 0.15
    weight_length_penalty: float = 0.10
    position_decay_lambda: float = 0.15
    min_sentence_length: int = 5
    max_sentence_length: int = 50
