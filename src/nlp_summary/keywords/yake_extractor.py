"""YAKE (Yet Another Keyword Extractor) Wrapper implementation."""

from typing import List, Tuple
from nlp_summary.keywords.base import KeywordExtractorBase
from nlp_summary.keywords.normalizer import ScoreNormalizer
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument


class YAKEExtractor(KeywordExtractorBase):
    """Extracts keywords using statistical and contextual features via YAKE."""

    def __init__(self, language: str = "vi", max_ngram_size: int = 3):
        super().__init__(name="yake")
        self.language = language
        self.max_ngram_size = max_ngram_size
        self._yake_available = False
        try:
            import yake
            self._yake_module = yake
            self._yake_available = True
        except ImportError:
            self._yake_module = None

    def extract_keywords(
        self,
        doc: ProcessedDocument,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 3),
    ) -> List[ExtractedKeyword]:
        """Extracts keywords using YAKE heuristics."""
        if not doc.cleaned_text:
            return []

        if not self._yake_available:
            # Fallback to TextRank if yake not installed
            from nlp_summary.keywords.textrank import TextRankExtractor
            return TextRankExtractor().extract_keywords(doc, top_k=top_k, ngram_range=ngram_range)

        try:
            custom_kw_extractor = self._yake_module.KeywordExtractor(
                lan=self.language if self.language in ["vi", "en"] else "en",
                n=min(ngram_range[1], self.max_ngram_size),
                dedupLim=0.8,
                dedupFunc="seqm",
                windowsSize=2,
                top=top_k * 2,
                features=None,
            )

            # In YAKE, lower score = higher importance
            raw_keywords = custom_kw_extractor.extract_keywords(doc.cleaned_text)
            if not raw_keywords:
                return []

            raw_scores = {kw: float(score) for kw, score in raw_keywords}

            # Invert so higher normalized score = higher importance
            norm_scores = ScoreNormalizer.min_max_normalize(raw_scores, reverse=True)

            sorted_items = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results: List[ExtractedKeyword] = []
            for kw, norm_score in sorted_items:
                raw_score = raw_scores.get(kw, 0.0)
                ngram_len = len(kw.split())
                results.append(
                    ExtractedKeyword(
                        keyword=kw,
                        normalized_score=round(norm_score, 4),
                        raw_score=round(raw_score, 6),
                        frequency=1,
                        ngram_length=ngram_len,
                        algorithm=self.name,
                    )
                )

            return results
        except Exception:
            return []
