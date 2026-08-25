"""Analytics service to evaluate reading productivity and economic ROI."""

from nlp_summary.models.impact import (
    ReadingProductivityMetrics,
    DomainROIConfig,
    DomainROISimulationResult,
)


class ProductivityAnalyticsService:
    """Computes real-world productivity savings and financial impact."""

    VIETNAMESE_WORDS_PER_MINUTE: float = 220.0  # Average Vietnamese reading speed

    @classmethod
    def calculate_reading_impact(
        cls, original_text: str, summary_text: str, keyword_coverage: float = 0.8
    ) -> ReadingProductivityMetrics:
        """Calculates exact reading time reduction and efficiency gain."""
        orig_words = len(original_text.split())
        sum_words = max(1, len(summary_text.split()))

        orig_time_sec = (orig_words / cls.VIETNAMESE_WORDS_PER_MINUTE) * 60.0
        sum_time_sec = (sum_words / cls.VIETNAMESE_WORDS_PER_MINUTE) * 60.0

        time_saved_sec = max(0.0, orig_time_sec - sum_time_sec)
        time_saved_pct = (time_saved_sec / orig_time_sec * 100.0) if orig_time_sec > 0 else 0.0
        speedup = (orig_words / sum_words) if sum_words > 0 else 1.0
        compression = round((1.0 - (sum_words / orig_words)) * 100.0, 1) if orig_words > 0 else 0.0

        # Information concentration: How much denser key info is per 100 words
        orig_density = (keyword_coverage / orig_words * 100) if orig_words > 0 else 0
        sum_density = (keyword_coverage / sum_words * 100) if sum_words > 0 else 0
        density_boost = round((sum_density / orig_density), 2) if orig_density > 0 else 1.0

        return ReadingProductivityMetrics(
            original_word_count=orig_words,
            summary_word_count=sum_words,
            compression_ratio=compression,
            original_reading_time_sec=round(orig_time_sec, 1),
            summary_reading_time_sec=round(sum_time_sec, 1),
            time_saved_sec=round(time_saved_sec, 1),
            time_saved_percent=round(time_saved_pct, 1),
            speedup_factor=round(speedup, 1),
            keyword_density_boost=density_boost,
        )

    @classmethod
    def simulate_monthly_roi(
        cls, config: DomainROIConfig, avg_compression_ratio: float = 0.75
    ) -> DomainROISimulationResult:
        """Simulates organizational time and financial savings per month."""
        working_days = 22
        monthly_docs = config.documents_per_day * working_days
        total_orig_words = monthly_docs * config.avg_words_per_doc

        monthly_hours_before = (total_orig_words / cls.VIETNAMESE_WORDS_PER_MINUTE) / 60.0
        monthly_hours_after = monthly_hours_before * (1.0 - avg_compression_ratio)
        hours_saved = monthly_hours_before - monthly_hours_after
        cost_saved = hours_saved * config.hourly_rate_vnd

        return DomainROISimulationResult(
            domain_name=config.domain_name,
            monthly_docs_processed=monthly_docs,
            monthly_reading_hours_before=round(monthly_hours_before, 1),
            monthly_reading_hours_after=round(monthly_hours_after, 1),
            hours_saved_per_month=round(hours_saved, 1),
            monthly_cost_saved_vnd=round(cost_saved, 0),
            productivity_gain_percent=round(avg_compression_ratio * 100, 1),
        )
