"""Unified preprocessing pipeline coordinator."""

from typing import Optional, List
from nlp_summary.models.preprocessing import ProcessedDocument, Sentence
from nlp_summary.preprocessing.normalizer import TextNormalizer
from nlp_summary.preprocessing.sentence_splitter import SentenceSegmenter
from nlp_summary.preprocessing.tokenizer import VietnameseTokenizer
from nlp_summary.preprocessing.stopwords import StopwordFilter


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for documents."""

    def __init__(
        self,
        language: str = "vi",
        custom_stopwords: Optional[List[str]] = None
    ):
        self.language = language
        self.stopword_filter = StopwordFilter(language=language, custom_stopwords=custom_stopwords)
        self.tokenizer = VietnameseTokenizer(stopword_filter=self.stopword_filter)

    def process(self, raw_text: str) -> ProcessedDocument:
        """Processes raw text into a fully annotated ProcessedDocument."""
        # 1. Normalization
        cleaned_text = TextNormalizer.normalize(raw_text)
        if not cleaned_text:
            return ProcessedDocument(
                original_text=raw_text,
                cleaned_text="",
                language=self.language,
                sentences=[],
                total_tokens=0
            )

        # 2. Sentence segmentation
        sentences: List[Sentence] = SentenceSegmenter.split(cleaned_text)

        # 3. Tokenization & POS Tagging for each sentence
        total_tokens = 0
        for sent in sentences:
            sent.tokens = self.tokenizer.tokenize_sentence(sent.cleaned_text)
            total_tokens += len(sent.tokens)

        return ProcessedDocument(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            language=self.language,
            sentences=sentences,
            total_tokens=total_tokens
        )
