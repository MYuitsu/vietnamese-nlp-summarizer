"""ROUGE metric evaluation for Vietnamese and multilingual text."""

from collections import Counter
from typing import List, Tuple
from nlp_summary.models.evaluation import MetricScore, RougeEvaluationResult
from nlp_summary.preprocessing.tokenizer import VietnameseTokenizer


class RougeEvaluator:
    """Computes word-segmented ROUGE-1, ROUGE-2, and ROUGE-L."""

    def __init__(self):
        self.tokenizer = VietnameseTokenizer()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into compound word tokens."""
        tokens = self.tokenizer.tokenize_sentence(text)
        return [t.text.lower() for t in tokens if t.text.isalnum() or "_" in t.text]

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extracts n-gram frequency counter."""
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def _compute_ngram_overlap(self, cand_tokens: List[str], ref_tokens: List[str], n: int) -> MetricScore:
        """Computes ROUGE-N Precision, Recall, and F1."""
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not cand_ngrams or not ref_ngrams:
            return MetricScore(precision=0.0, recall=0.0, f1_score=0.0)

        overlap = 0
        for ngram, count in cand_ngrams.items():
            if ngram in ref_ngrams:
                overlap += min(count, ref_ngrams[ngram])

        total_cand = sum(cand_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        p = overlap / max(1, total_cand)
        r = overlap / max(1, total_ref)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        return MetricScore(precision=round(p, 4), recall=round(r, 4), f1_score=round(f1, 4))

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Calculates Longest Common Subsequence length between two token lists."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def _compute_lcs(self, cand_tokens: List[str], ref_tokens: List[str]) -> MetricScore:
        """Computes ROUGE-L Precision, Recall, and F1."""
        if not cand_tokens or not ref_tokens:
            return MetricScore(precision=0.0, recall=0.0, f1_score=0.0)

        lcs_len = self._lcs_length(cand_tokens, ref_tokens)
        p = lcs_len / len(cand_tokens)
        r = lcs_len / len(ref_tokens)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        return MetricScore(precision=round(p, 4), recall=round(r, 4), f1_score=round(f1, 4))

    def evaluate(self, candidate_summary: str, reference_summary: str) -> RougeEvaluationResult:
        """Calculates full ROUGE-1, ROUGE-2, and ROUGE-L metrics."""
        cand_tokens = self._tokenize(candidate_summary)
        ref_tokens = self._tokenize(reference_summary)

        rouge_1 = self._compute_ngram_overlap(cand_tokens, ref_tokens, n=1)
        rouge_2 = self._compute_ngram_overlap(cand_tokens, ref_tokens, n=2)
        rouge_l = self._compute_lcs(cand_tokens, ref_tokens)

        # Approximate keyword retention based on content token recall
        cand_set = set(cand_tokens)
        ref_set = set(ref_tokens)
        retention = len(cand_set.intersection(ref_set)) / max(1, len(ref_set))

        return RougeEvaluationResult(
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            keyword_retention=round(min(1.0, retention), 4),
        )
