"""Comparable metrics for LLM summaries generated from the same experiment input."""

from __future__ import annotations

from pydantic import BaseModel

from nlp_summary.evaluation.rouge import RougeEvaluator
from nlp_summary.summarizer.llm_summarizer import LLMSummaryResult


class LLMBenchmarkRow(BaseModel):
    """One observed result row; no metric in this model is simulated."""

    model: str
    status: str
    latency_ms: float
    word_count: int
    compression_ratio: float
    anchor_coverage: float
    number_faithfulness: float
    rouge_1_f1: float | None = None
    rouge_2_f1: float | None = None
    rouge_l_f1: float | None = None
    error: str | None = None


def build_benchmark_row(
    result: LLMSummaryResult,
    reference_summary: str = "",
) -> LLMBenchmarkRow:
    """Convert an API result to a reproducible benchmark row."""
    rouge_1 = rouge_2 = rouge_l = None
    if reference_summary.strip() and result.summary.strip() and not result.is_fallback:
        rouge = RougeEvaluator().evaluate(result.summary, reference_summary)
        rouge_1 = rouge.rouge_1.f1_score
        rouge_2 = rouge.rouge_2.f1_score
        rouge_l = rouge.rouge_l.f1_score

    return LLMBenchmarkRow(
        model=result.model_used,
        status="Thành công" if result.summary and not result.is_fallback else "Lỗi API",
        latency_ms=result.latency_ms,
        word_count=result.summary_word_count,
        compression_ratio=result.compression_ratio,
        anchor_coverage=result.anchor_coverage,
        number_faithfulness=result.number_faithfulness,
        rouge_1_f1=rouge_1,
        rouge_2_f1=rouge_2,
        rouge_l_f1=rouge_l,
        error=result.fallback_reason,
    )
