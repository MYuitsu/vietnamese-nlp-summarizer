"""Score normalization utilities for keyword extractors."""

from typing import List, Dict
import math


class ScoreNormalizer:
    """Normalizes raw extractor scores to [0.0, 1.0] interval."""

    @staticmethod
    def min_max_normalize(score_dict: Dict[str, float], reverse: bool = False) -> Dict[str, float]:
        """Performs min-max scaling to map scores to [0, 1].

        If reverse=True (e.g. for YAKE where lower raw score means higher importance),
        the scaling is inverted so higher normalized score means higher importance.
        """
        if not score_dict:
            return {}

        raw_scores = list(score_dict.values())
        min_val = min(raw_scores)
        max_val = max(raw_scores)

        # Handle edge case where all scores are identical
        if math.isclose(min_val, max_val, rel_tol=1e-7, abs_tol=1e-7):
            return {k: 1.0 for k in score_dict}

        normalized: Dict[str, float] = {}
        for k, v in score_dict.items():
            if reverse:
                # Invert: smaller raw -> larger normalized
                norm_score = (max_val - v) / (max_val - min_val)
            else:
                norm_score = (v - min_val) / (max_val - min_val)
            # Clip between 0.0 and 1.0
            normalized[k] = max(0.0, min(1.0, float(norm_score)))

        return normalized
