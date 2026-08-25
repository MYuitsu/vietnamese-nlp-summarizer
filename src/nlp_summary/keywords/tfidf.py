"""TF-IDF Keyword Extractor implementation with POS-tag Filtering (Nouns, Verbs, Adjectives)."""

from collections import Counter
from typing import List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_summary.keywords.base import KeywordExtractorBase
from nlp_summary.keywords.normalizer import ScoreNormalizer
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument

# Standard Vietnamese POS tags for content words (Nouns, Verbs, Adjectives)
ALLOWED_POS_TAGS: Set[str] = {"N", "Np", "Nc", "Nu", "V", "Vb", "A"}


class TFIDFExtractor(KeywordExtractorBase):
    """Extracts keywords using intra-document TF-IDF with POS-guided filtering."""

    def __init__(self, allowed_pos_tags: Optional[Set[str]] = None):
        super().__init__(name="tfidf")
        self.allowed_pos_tags = allowed_pos_tags or ALLOWED_POS_TAGS

    def extract_keywords(
        self,
        doc: ProcessedDocument,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 3),
    ) -> List[ExtractedKeyword]:
        """Extracts keywords by computing TF-IDF across sentence documents with POS filtering."""
        if not doc.sentences:
            return []

        # Prepare sentence corpora using POS-filtered segmented compound tokens
        sentence_texts: List[str] = []
        token_counter = Counter()
        token_pos_map = {}

        for sent in doc.sentences:
            valid_tokens = []
            for t in sent.tokens:
                # 1. Stopword and format checks
                if t.is_stopword or not (t.text.isalnum() or "_" in t.text):
                    continue
                # 2. Spec 012 POS Filter: Keep Nouns, Verbs, and Adjectives
                if t.pos_tag and t.pos_tag not in self.allowed_pos_tags:
                    continue

                tok_lower = t.text.lower()
                valid_tokens.append(tok_lower)
                token_pos_map[tok_lower] = t.pos_tag

            token_counter.update(valid_tokens)
            if valid_tokens:
                sentence_texts.append(" ".join(valid_tokens))

        if not sentence_texts:
            return []

        # If document has only 1 sentence, replicate it to allow IDF computation
        if len(sentence_texts) == 1:
            sentence_texts = [sentence_texts[0], sentence_texts[0]]

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                token_pattern=r"(?u)\b[\w_]+\b",
                max_features=100,
            )
            tfidf_matrix = vectorizer.fit_transform(sentence_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Aggregate mean TF-IDF score across all sentences
            mean_scores = tfidf_matrix.mean(axis=0).A1
            raw_scores = {feature_names[i]: float(mean_scores[i]) for i in range(len(feature_names))}

            # Normalize to [0, 1]
            norm_scores = ScoreNormalizer.min_max_normalize(raw_scores, reverse=False)

            # Sort descending by normalized score
            sorted_items = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results: List[ExtractedKeyword] = []
            for kw, norm_score in sorted_items:
                raw_score = raw_scores.get(kw, 0.0)
                ngram_len = len(kw.split())
                freq = token_counter.get(kw, 1)
                results.append(
                    ExtractedKeyword(
                        keyword=kw,
                        normalized_score=round(norm_score, 4),
                        raw_score=round(raw_score, 4),
                        frequency=freq,
                        ngram_length=ngram_len,
                        algorithm=self.name,
                    )
                )

            return results
        except Exception:
            return []
