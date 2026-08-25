"""Unified Extractive Summarization Engine Facade."""

from typing import List, Optional
from nlp_summary.keywords.registry import ExtractorRegistry
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument
from nlp_summary.models.scoring import ScoringWeightsConfig
from nlp_summary.models.summary import GeneratedSummary, LengthConstraint
from nlp_summary.preprocessing.pipeline import PreprocessingPipeline
from nlp_summary.scoring.scorer import SentenceScorer
from nlp_summary.summarizer.mmr_summarizer import MMRSummarizer


class ExtractiveSummarizerEngine:
    """High-level facade orchestrating Preprocessing -> Extraction -> Scoring -> MMR Summary."""

    def __init__(
        self,
        language: str = "vi",
        default_algorithm: str = "textrank",
        custom_stopwords: Optional[List[str]] = None,
    ):
        self.language = language
        self.default_algorithm = default_algorithm
        self.preprocessor = PreprocessingPipeline(language=language, custom_stopwords=custom_stopwords)
        self.scorer = SentenceScorer()

    def summarize(
        self,
        text: str,
        title: Optional[str] = None,
        algorithm: Optional[str] = None,
        top_k_keywords: int = 10,
        length_constraint: Optional[LengthConstraint] = None,
        scoring_weights: Optional[ScoringWeightsConfig] = None,
        mmr_lambda: float = 0.7,
    ) -> GeneratedSummary:
        """Executes full extractive summarization pipeline."""
        if not text or not text.strip():
            return GeneratedSummary(summary_text="")

        algo_name = (algorithm or self.default_algorithm).lower().strip()
        constraint = length_constraint or LengthConstraint(mode="compression_ratio", value=0.30)

        # 1. Preprocess Document
        doc: ProcessedDocument = self.preprocessor.process(text)
        if not doc.sentences:
            return GeneratedSummary(summary_text="")

        # 2. Extract Keywords
        extractor = ExtractorRegistry.get_extractor(algo_name)
        keywords: List[ExtractedKeyword] = extractor.extract_keywords(doc, top_k=top_k_keywords)

        # 3. Score Sentences
        scorer = SentenceScorer(config=scoring_weights) if scoring_weights else self.scorer
        scored_sentences = scorer.score_sentences(doc, keywords=keywords, title=title)

        # 4. Generate MMR Summary
        summary = MMRSummarizer.generate_summary(
            doc=doc,
            scored_sentences=scored_sentences,
            keywords=keywords,
            constraint=constraint,
            mmr_lambda=mmr_lambda,
            algorithm_name=algo_name,
        )

        return summary
