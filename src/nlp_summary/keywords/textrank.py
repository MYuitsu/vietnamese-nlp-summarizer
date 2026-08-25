"""Graph-based TextRank Keyword Extractor implementation."""

from collections import Counter
from typing import Dict, List, Tuple
import networkx as nx

from nlp_summary.keywords.base import KeywordExtractorBase
from nlp_summary.keywords.normalizer import ScoreNormalizer
from nlp_summary.models.keywords import ExtractedKeyword
from nlp_summary.models.preprocessing import ProcessedDocument


class TextRankExtractor(KeywordExtractorBase):
    """Extracts keywords by constructing a word co-occurrence graph and running PageRank."""

    def __init__(self, window_size: int = 3, damping_factor: float = 0.85):
        super().__init__(name="textrank")
        self.window_size = window_size
        self.damping_factor = damping_factor

    def extract_keywords(
        self,
        doc: ProcessedDocument,
        top_k: int = 10,
        ngram_range: Tuple[int, int] = (1, 3),
    ) -> List[ExtractedKeyword]:
        """Builds co-occurrence graph of candidate tokens and computes PageRank."""
        if not doc.sentences:
            return []

        # Collect candidate tokens (Nouns, Verbs, Adjectives, Compounds)
        words_sequence: List[str] = []
        token_counter = Counter()

        for sent in doc.sentences:
            for t in sent.tokens:
                text_clean = t.text.lower().strip()
                if (
                    not t.is_stopword
                    and len(text_clean) > 1
                    and (t.pos_tag in ["N", "Np", "Nc", "V", "A"] or t.is_compound)
                ):
                    words_sequence.append(text_clean)
                    token_counter[text_clean] += 1

        if not words_sequence:
            return []

        # Build undirected graph with sliding window
        graph = nx.Graph()
        for i in range(len(words_sequence)):
            w1 = words_sequence[i]
            graph.add_node(w1)
            for j in range(i + 1, min(i + self.window_size, len(words_sequence))):
                w2 = words_sequence[j]
                if w1 != w2:
                    if graph.has_edge(w1, w2):
                        graph[w1][w2]["weight"] += 1.0
                    else:
                        graph.add_edge(w1, w2, weight=1.0)

        if graph.number_of_nodes() == 0:
            return []

        try:
            # Run PageRank
            pagerank_scores: Dict[str, float] = nx.pagerank(
                graph,
                alpha=self.damping_factor,
                max_iter=100,
                tol=1e-6,
                weight="weight",
            )

            # Normalize scores to [0, 1]
            norm_scores = ScoreNormalizer.min_max_normalize(pagerank_scores, reverse=False)

            # Sort descending
            sorted_items = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results: List[ExtractedKeyword] = []
            for kw, norm_score in sorted_items:
                raw_score = pagerank_scores.get(kw, 0.0)
                ngram_len = len(kw.split("_")) if "_" in kw else 1
                freq = token_counter.get(kw, 1)

                results.append(
                    ExtractedKeyword(
                        keyword=kw,
                        normalized_score=round(norm_score, 4),
                        raw_score=round(raw_score, 6),
                        frequency=freq,
                        ngram_length=ngram_len,
                        algorithm=self.name,
                    )
                )

            return results
        except Exception:
            return []
