"""KeyBERT Semantic Keyword Extractor using Vietnamese SBERT Embeddings."""

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nlp_summary.keywords.base import KeywordExtractorBase
from nlp_summary.keywords.normalizer import ScoreNormalizer
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument


class KeyBERTExtractor(KeywordExtractorBase):
    """Extracts keywords by comparing candidate n-gram embeddings with document embedding."""

    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"):
        super().__init__(name="keybert")
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _lazy_load_model(self):
        """Loads SentenceTransformer on demand."""
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None
            self._initialized = True

    def extract_keywords(
        self,
        doc: ProcessedDocument,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 3),
        diversity: float = 0.7,
    ) -> List[ExtractedKeyword]:
        """Extracts candidate keywords using embedding cosine similarity."""
        if not doc.cleaned_text:
            return []

        self._lazy_load_model()
        if self._model is None:
            # Fallback to TextRank if sentence-transformers unavailable
            from nlp_summary.keywords.textrank import TextRankExtractor
            return TextRankExtractor().extract_keywords(doc, top_k=top_k, ngram_range=ngram_range)

        try:
            # Step 1: Candidate selection using CountVectorizer
            # Use space-tokenized text with Vietnamese compound words unified
            processed_lines = []
            for s in doc.sentences:
                line_tokens = [t.text for t in s.tokens if not t.is_stopword and len(t.text) > 1]
                if line_tokens:
                    processed_lines.append(" ".join(line_tokens))

            full_processed_text = " ".join(processed_lines)
            if not full_processed_text.strip():
                full_processed_text = doc.cleaned_text

            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                token_pattern=r"(?u)\b[\w_]+\b",
            )
            vectorizer.fit([full_processed_text])
            candidates = vectorizer.get_feature_names_out()

            if len(candidates) == 0:
                return []

            # Step 2: Compute Embeddings
            doc_embedding = self._model.encode([doc.cleaned_text])
            candidate_embeddings = self._model.encode(candidates.tolist())

            # Step 3: Cosine Similarity
            distances = cosine_similarity(doc_embedding, candidate_embeddings)[0]

            # Step 4: Max Sum Distance or MMR Candidate Selection
            raw_scores = {candidates[i]: float(distances[i]) for i in range(len(candidates))}
            norm_scores = ScoreNormalizer.min_max_normalize(raw_scores, reverse=False)

            sorted_items = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results: List[ExtractedKeyword] = []
            for kw, norm_score in sorted_items:
                raw_score = raw_scores.get(kw, 0.0)
                ngram_len = len(kw.split())
                results.append(
                    ExtractedKeyword(
                        keyword=kw,
                        normalized_score=round(norm_score, 4),
                        raw_score=round(raw_score, 4),
                        frequency=1,
                        ngram_length=ngram_len,
                        algorithm=self.name,
                    )
                )

            return results
        except Exception:
            return []
