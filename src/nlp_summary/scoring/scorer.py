"""Composite sentence scorer combining keyword density, position, title, and length heuristics."""

import math
from typing import Dict, List, Optional
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument, Sentence
from nlp_summary.models.scoring import ScoreBreakdown, ScoredSentence, ScoringWeightsConfig


class SentenceScorer:
    """Calculates multi-feature salience scores for sentences."""

    def __init__(self, config: Optional[ScoringWeightsConfig] = None):
        self.config = config or ScoringWeightsConfig()

    def score_sentences(
        self,
        doc: ProcessedDocument,
        keywords: List[ExtractedKeyword],
        title: Optional[str] = None,
    ) -> List[ScoredSentence]:
        """Scores all sentences in the document."""
        if not doc.sentences:
            return []

        # Create keyword lookup map
        kw_map: Dict[str, float] = {
            kw.keyword.lower().replace(" ", "_"): kw.normalized_score
            for kw in keywords
        }

        # Tokenize title tokens if title exists
        title_tokens = set()
        if title:
            clean_title = title.lower().replace(" ", "_")
            title_tokens = set(clean_title.split())

        scored_list: List[ScoredSentence] = []
        total_sents = len(doc.sentences)

        for sent in doc.sentences:
            # 1. Keyword Density Score
            contained_kws = []
            kw_score_sum = 0.0
            content_tokens = [t.text.lower() for t in sent.tokens if not t.is_stopword]

            for tok in content_tokens:
                if tok in kw_map:
                    kw_score_sum += kw_map[tok]
                    contained_kws.append(tok)

            # Keyword score normalized by content length (with log dampening)
            kw_score = 0.0
            if content_tokens:
                kw_score = min(1.0, kw_score_sum / max(1, math.sqrt(len(content_tokens))))

            # 2. Position Bias Score (Exponential decay)
            idx = sent.index
            pos_score = math.exp(-self.config.position_decay_lambda * idx)

            # 3. Title Match Score
            title_score = 0.0
            if title_tokens and content_tokens:
                overlap = sum(1 for tok in content_tokens if tok in title_tokens)
                title_score = min(1.0, overlap / len(title_tokens))

            # 4. Length Penalty
            word_count = len(sent.tokens)
            length_penalty = 0.0
            if word_count < self.config.min_sentence_length:
                length_penalty = (self.config.min_sentence_length - word_count) / self.config.min_sentence_length
            elif word_count > self.config.max_sentence_length:
                length_penalty = min(1.0, (word_count - self.config.max_sentence_length) / 30.0)

            # 5. Composite Weighted Score
            final_score = (
                self.config.weight_keyword * kw_score
                + self.config.weight_position * pos_score
                + self.config.weight_title * title_score
                - self.config.weight_length_penalty * length_penalty
            )
            final_score = max(0.0, final_score)

            breakdown = ScoreBreakdown(
                keyword_score=round(kw_score, 4),
                position_score=round(pos_score, 4),
                title_match_score=round(title_score, 4),
                length_penalty=round(length_penalty, 4),
                final_score=round(final_score, 4),
            )

            scored_list.append(
                ScoredSentence(
                    index=sent.index,
                    raw_text=sent.raw_text,
                    cleaned_text=sent.cleaned_text,
                    score=round(final_score, 4),
                    breakdown=breakdown,
                    contained_keywords=list(set(contained_kws)),
                )
            )

        return scored_list
