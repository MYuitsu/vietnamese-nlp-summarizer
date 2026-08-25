"""Data models for evaluation and benchmarking."""

from typing import List
from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)


class RougeEvaluationResult(BaseModel):
    rouge_1: MetricScore
    rouge_2: MetricScore
    rouge_l: MetricScore
    keyword_retention: float = Field(ge=0.0, le=1.0, description="Proportion of key concepts retained")


class BenchmarkSummaryRow(BaseModel):
    algorithm: str
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    keyword_retention: float
    avg_latency_ms: float


class BenchmarkReport(BaseModel):
    dataset_name: str = "VietNews-Subset"
    sample_count: int
    results: List[BenchmarkSummaryRow]
    markdown_table: str
    latex_table: str
