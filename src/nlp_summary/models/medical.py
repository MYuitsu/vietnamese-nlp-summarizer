"""Data models for Medical & Clinical EHR Summarization."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.summary import GeneratedSummary


class CategorizedMedicalKeyword(BaseModel):
    """Keyword categorized into medical domains."""

    keyword: str
    category: str = Field(..., description="Triệu chứng | Chẩn đoán | Điều trị / Thuốc | Cảnh báo / Dị ứng | Chỉ số")
    raw_score: float
    badge_color: str


class ClinicalSoapSummary(BaseModel):
    """SOAP clinical structured handover summary."""

    patient_context: Optional[str] = Field(None, description="Patient age, gender, admission reason")
    subjective_symptoms: List[str] = Field(default_factory=list, description="Triệu chứng cơ năng / lý do khám")
    objective_findings: List[str] = Field(default_factory=list, description="Kết quả cận lâm sàng, xét nghiệm, sinh hiệu")
    assessment_diagnosis: List[str] = Field(default_factory=list, description="Chẩn đoán xác định / phân biệt")
    plan_treatment: List[str] = Field(default_factory=list, description="Phác đồ điều trị, đơn thuốc, theo dõi")
    critical_alerts: List[str] = Field(default_factory=list, description="Tiền sử dị ứng, cảnh báo nguy hiểm")
    categorized_keywords: List[CategorizedMedicalKeyword] = Field(default_factory=list)
    extractive_summary: GeneratedSummary
