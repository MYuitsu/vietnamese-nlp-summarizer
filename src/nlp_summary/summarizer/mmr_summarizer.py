"""MMR (Maximal Marginal Relevance) Selector and Chronological Reconstructor."""

import math
import re
from typing import List, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument
from nlp_summary.models.scoring import ScoredSentence
from nlp_summary.models.summary import GeneratedSummary, HighlightSpan, LengthConstraint


class MMRSummarizer:
    """Selects non-redundant sentences using Maximal Marginal Relevance and reconstructs order."""

    @staticmethod
    def _compute_similarity_matrix(sentences: List[ScoredSentence]) -> np.ndarray:
        """Computes pairwise cosine similarity between sentence TF-IDF vectors."""
        corpus = [s.cleaned_text for s in sentences]
        if not corpus or len(corpus) == 1:
            return np.zeros((len(corpus), len(corpus)))

        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w_]+\b")
            tfidf = vectorizer.fit_transform(corpus)
            return cosine_similarity(tfidf, tfidf)
        except Exception:
            return np.zeros((len(corpus), len(corpus)))

    @classmethod
    def generate_summary(
        cls,
        doc: ProcessedDocument,
        scored_sentences: List[ScoredSentence],
        keywords: List[ExtractedKeyword],
        constraint: LengthConstraint,
        mmr_lambda: float = 0.7,
        algorithm_name: str = "textrank",
    ) -> GeneratedSummary:
        """Runs MMR selection and formats the final summary."""
        if not scored_sentences:
            return GeneratedSummary(summary_text="", algorithm_used=algorithm_name)

        total_sentences = len(scored_sentences)

        # 1. Determine target sentence count based on mode
        if constraint.mode == "sentence_count":
            target_k = int(min(max(1, constraint.value), total_sentences))
        elif constraint.mode == "compression_ratio":
            ratio = max(0.05, min(1.0, constraint.value))
            target_k = int(max(1, math.ceil(total_sentences * ratio)))
        elif constraint.mode == "max_words":
            target_k = total_sentences  # Handled in greedy accumulation loop
        else:
            target_k = max(1, int(total_sentences * 0.3))

        # 2. Similarity Matrix
        sim_matrix = cls._compute_similarity_matrix(scored_sentences)

        # 3. Greedy MMR Selection
        selected_indices: List[int] = []
        unselected_indices: Set[int] = set(range(total_sentences))
        current_word_count = 0
        max_words_limit = int(constraint.value) if constraint.mode == "max_words" else 999999

        while unselected_indices and (\
            (constraint.mode != "max_words" and len(selected_indices) < target_k)\
            or (constraint.mode == "max_words" and current_word_count < max_words_limit)\
        ):
            best_idx = -1
            best_mmr_score = -float("inf")

            for i in unselected_indices:
                salience = scored_sentences[i].score

                # Maximum similarity to any previously selected sentence
                max_sim = 0.0
                if selected_indices:
                    max_sim = max(float(sim_matrix[i][j]) for j in selected_indices)

                mmr_val = mmr_lambda * salience - (1.0 - mmr_lambda) * max_sim

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = i

            if best_idx == -1:
                break

            sent_words = len(scored_sentences[best_idx].cleaned_text.split())
            if constraint.mode == "max_words" and selected_indices and (current_word_count + sent_words > max_words_limit):
                unselected_indices.remove(best_idx)
                continue

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
            current_word_count += sent_words

        # 4. Chronological Reordering
        selected_indices.sort()
        final_sentences = [scored_sentences[i] for i in selected_indices]
        summary_text = " ".join(s.raw_text for s in final_sentences)

        # 5. Extract Highlight Spans and HTML
        highlight_spans: List[HighlightSpan] = []
        kw_list = sorted([kw.keyword.lower().replace("_", " ") for kw in keywords], key=len, reverse=True)

        for sent in final_sentences:
            for clean_kw in kw_list:
                start = 0
                sent_lower = sent.raw_text.lower()
                while True:
                    idx = sent_lower.find(clean_kw, start)
                    if idx == -1:
                        break
                    highlight_spans.append(
                        HighlightSpan(
                            keyword=clean_kw,
                            sentence_index=sent.index,
                            start_char=idx,
                            end_char=idx + len(clean_kw),
                        )
                    )
                    start = idx + len(clean_kw)

        # Generate HTML with highlighted keywords
        html_summary = summary_text
        for kw in kw_list:
            if kw.strip():
                pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
                html_summary = pattern.sub(r'<mark style="background-color: #FEF08A; padding: 0.1rem 0.3rem; border-radius: 3px; font-weight: 600;">\1</mark>', html_summary)

        # 6. Metadata and Keyword Coverage
        orig_words = len(doc.cleaned_text.split())
        sum_words = len(summary_text.split())
        covered_kws = set()
        for s in final_sentences:
            covered_kws.update(s.contained_keywords)

        coverage = len(covered_kws) / max(1, len(keywords))

        return GeneratedSummary(
            summary_text=summary_text,
            selected_sentences=final_sentences,
            selected_keywords=keywords,
            original_sentence_count=total_sentences,
            summary_sentence_count=len(final_sentences),
            original_word_count=orig_words,
            summary_word_count=sum_words,
            compression_ratio=round((sum_words / max(1, orig_words)), 3),
            keyword_coverage_ratio=round(min(1.0, coverage), 3),
            highlight_spans=highlight_spans,
            highlighted_summary_html=html_summary,
            algorithm_used=algorithm_name,
        )
