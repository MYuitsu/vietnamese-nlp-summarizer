"""FastAPI Backend Application exposing NLP Summarization Services."""

import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nlp_summary.keywords.registry import ExtractorRegistry
from nlp_summary.models.evaluation import RougeEvaluationResult
from nlp_summary.models.keywords import ExtractedKeyword, KeywordExtractionResult
from nlp_summary.models.preprocessing import ProcessedDocument
from nlp_summary.models.scoring import ScoringWeightsConfig
from nlp_summary.models.summary import GeneratedSummary, LengthConstraint
from nlp_summary.evaluation.rouge import RougeEvaluator
from nlp_summary.preprocessing.pipeline import PreprocessingPipeline
from nlp_summary.summarizer.engine import ExtractiveSummarizerEngine

app = FastAPI(
    title="Keyword-Based Text Summarization API",
    description="REST API for Vietnamese & Multilingual Extractive Text Summarization",
    version="1.0.0",
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ExtractiveSummarizerEngine()
preprocessor = PreprocessingPipeline()
evaluator = RougeEvaluator()


class PreprocessRequest(BaseModel):
    text: str


class KeywordRequest(BaseModel):
    text: str
    algorithm: str = "textrank"
    top_k: int = 10


class SummarizeRequest(BaseModel):
    text: str
    title: Optional[str] = None
    algorithm: str = "textrank"
    top_k_keywords: int = 10
    length_constraint: LengthConstraint = Field(default_factory=LengthConstraint)
    scoring_weights: Optional[ScoringWeightsConfig] = None
    mmr_lambda: float = 0.7


class SummarizeResponse(BaseModel):
    summary: GeneratedSummary
    extracted_keywords: List[ExtractedKeyword]
    execution_time_ms: float


class EvaluateRequest(BaseModel):
    candidate_summary: str
    reference_summary: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "nlp-keyword-summary", "version": "1.0.0"}


@app.post("/api/v1/preprocess", response_model=ProcessedDocument)
def preprocess_text(req: PreprocessRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return preprocessor.process(req.text)


@app.post("/api/v1/extract-keywords", response_model=KeywordExtractionResult)
def extract_keywords(req: KeywordRequest):
    start = time.time()
    doc = preprocessor.process(req.text)
    extractor = ExtractorRegistry.get_extractor(req.algorithm)
    kws = extractor.extract_keywords(doc, top_k=req.top_k)
    elapsed = (time.time() - start) * 1000.0

    return KeywordExtractionResult(
        algorithm=req.algorithm,
        top_k=req.top_k,
        execution_time_ms=round(elapsed, 2),
        keywords=kws,
    )


@app.post("/api/v1/summarize", response_model=SummarizeResponse)
def summarize_text(req: SummarizeRequest):
    start = time.time()
    doc = preprocessor.process(req.text)
    extractor = ExtractorRegistry.get_extractor(req.algorithm)
    kws = extractor.extract_keywords(doc, top_k=req.top_k_keywords)

    summary = engine.summarize(
        text=req.text,
        title=req.title,
        algorithm=req.algorithm,
        top_k_keywords=req.top_k_keywords,
        length_constraint=req.length_constraint,
        scoring_weights=req.scoring_weights,
        mmr_lambda=req.mmr_lambda,
    )
    elapsed = (time.time() - start) * 1000.0

    return SummarizeResponse(
        summary=summary,
        extracted_keywords=kws,
        execution_time_ms=round(elapsed, 2),
    )


@app.post("/api/v1/evaluate", response_model=RougeEvaluationResult)
def evaluate_summary(req: EvaluateRequest):
    if not req.candidate_summary or not req.reference_summary:
        raise HTTPException(status_code=400, detail="Both candidate and reference summaries required")
    return evaluator.evaluate(req.candidate_summary, req.reference_summary)
