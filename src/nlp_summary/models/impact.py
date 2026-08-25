"""Business value & productivity analytics model for Keyword-Based Summarization."""

from pydantic import BaseModel, Field
from typing import List, Dict


class ReadingProductivityMetrics(BaseModel):
    """Calculates time saved, reading speedup, and information density."""

    original_word_count: int = Field(..., description="Total words in original text")
    summary_word_count: int = Field(..., description="Total words in summary")
    compression_ratio: float = Field(..., description="Percentage of text reduction")
    original_reading_time_sec: float = Field(..., description="Reading time for original text (at ~220 WPM)")
    summary_reading_time_sec: float = Field(..., description="Reading time for summary")
    time_saved_sec: float = Field(..., description="Seconds saved")
    time_saved_percent: float = Field(..., description="Percentage of reading time saved")
    speedup_factor: float = Field(..., description="Reading speedup multiplier, e.g. 4.5x faster")
    keyword_density_boost: float = Field(..., description="Concentration increase of core information")


class DomainROIConfig(BaseModel):
    """Configuration for domain ROI simulation."""

    domain_name: str
    documents_per_day: int
    avg_words_per_doc: int
    hourly_rate_vnd: float = 100_000.0  # e.g., 100k VND/hour researcher/analyst salary


class DomainROISimulationResult(BaseModel):
    """Result of ROI simulation over a monthly period."""

    domain_name: str
    monthly_docs_processed: int
    monthly_reading_hours_before: float
    monthly_reading_hours_after: float
    hours_saved_per_month: float
    monthly_cost_saved_vnd: float
    productivity_gain_percent: float
