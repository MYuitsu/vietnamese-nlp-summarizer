"""Stopwords dictionary and dynamic filtering logic for Vietnamese, English, and Domain-specific corpora."""

from typing import Set, Optional, List, Dict
import json
import os

# Core curated Vietnamese stopwords (General & Academic/News)
VIETNAMESE_GENERAL_STOPWORDS: Set[str] = {
    "và", "của", "là", "có", "được", "trong", "đã", "cho", "với", "không", "các",
    "những", "một", "này", "khi", "để", "tại", "từ", "về", "như", "theo", "đến",
    "người", "ra", "lại", "nhiều", "vào", "trên", "cũng", "đó", "thì", "sẽ",
    "bị", "hơn", "sau", "đang", "phải", "nhưng", "rất", "cùng", "qua", "làm",
    "khác", "nếu", "bởi", "vì", "nay", "lên", "trước", "cả", "ở", "tới", "chỉ",
    "còn", "hay", "gì", "nào", "đều", "nên", "vẫn", "đây", "tự", "việc", "sự",
    "cái", "thế", "nữa", "nhất", "hết", "chứ", "biết", "thấy", "cho_biết", "cho_rằng",
    "theo_đó", "cụ_thể", "hiện_nay", "tuy_nhiên", "do_đó", "vì_vậy", "ngoài_ra",
    "mặt_khác", "tóm_lại", "đồng_thời", "như_vậy", "ngay_cả", "thực_tế", "vừa_qua",
    "ngày_nay", "trước_đây", "sau_đó", "ngay_sau", "lúc_này", "vừa", "mới", "luôn",
    "từng", "vừa_mới", "chưa", "chẳng", "đâu", "nào_đó", "ai_đó", "bao_giờ", "bao_lâu",
    "vô_cùng", "quá", "lắm", "hãy", "đừng", "chớ", "nhé", "nha", "ạ", "ơi", "à", "ừ",
    "rằng", "thì_là", "thay_vì", "nhờ_đó", "do", "bởi_vì", "mặc_dù", "dù_cho",
    "thế_nhưng", "song", "bởi_thế", "cho_nên", "kể_cả", "bao_gồm", "thuộc", "liên_quan",
    "về_việc", "nhằm", "mục_đích", "trường_hợp", "tổng_số", "khoảng", "chừng",
    "gần_như", "hầu_như", "hầu_hết", "toàn_bộ", "đa_số", "phần_lớn", "số_lượng",
    "mỗi", "mọi", "từng_cái", "các_loại", "v.v.", "vân_vân", "v.v", "v_v"
}

# Curated Medical & Administrative Stopwords (Noise in Clinical EHR)
CLINICAL_ADMIN_STOPWORDS: Set[str] = {
    "kính_gửi", "hội_chẩn", "hội_chẩn_lúc", "vào_viện_lúc", "ngày_vào_viện",
    "ngày_ra_viện", "bệnh_nhân", "bệnh_viện", "khoa_khám_bệnh", "bác_sĩ_điều_trị",
    "theo_chỉ_định", "ghi_nhận", "kết_quả_cho_thấy", "tiến_hành", "thực_hiện",
    "phút", "giờ", "ngày", "tháng", "năm", "tuổi", "nam", "nữ", "kính_chuyển",
    "theo_quy_định", "tại_chỗ", "cụ_thể_như", "được_biết", "cho_hay", "nhận_định"
}

# Full Default Vietnamese Stopwords Union
DEFAULT_VIETNAMESE_STOPWORDS: Set[str] = VIETNAMESE_GENERAL_STOPWORDS.union(CLINICAL_ADMIN_STOPWORDS)

ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}


class StopwordFilter:
    """Filters stopwords dynamically with realtime add, remove, presets, and import/export capabilities."""

    def __init__(self, language: str = "vi", custom_stopwords: Optional[List[str]] = None, include_clinical: bool = True):
        self.language = language
        self.include_clinical = include_clinical
        
        if language == "vi":
            self.stopwords: Set[str] = set(DEFAULT_VIETNAMESE_STOPWORDS if include_clinical else VIETNAMESE_GENERAL_STOPWORDS)
        else:
            self.stopwords: Set[str] = set(ENGLISH_STOPWORDS)

        self.custom_stopwords: Set[str] = set()
        if custom_stopwords:
            self.add_stopwords(custom_stopwords)

    def add_stopwords(self, words: List[str]) -> None:
        """Dynamically adds new stopwords in real-time."""
        for word in words:
            cleaned = word.lower().strip().replace(" ", "_")
            if cleaned:
                self.stopwords.add(cleaned)
                self.custom_stopwords.add(cleaned)

    def remove_stopwords(self, words: List[str]) -> None:
        """Dynamically removes stopwords from filter."""
        for word in words:
            cleaned = word.lower().strip().replace(" ", "_")
            self.stopwords.discard(cleaned)
            self.custom_stopwords.discard(cleaned)

    def is_stopword(self, token_text: str) -> bool:
        """Checks if a token text (lowercase or underscore normalized) is in stopwords."""
        clean = token_text.lower().strip()
        return clean in self.stopwords or clean.replace(" ", "_") in self.stopwords

    def export_custom_stopwords(self) -> List[str]:
        """Returns sorted list of active custom stopwords."""
        return sorted(list(self.custom_stopwords))

    def get_all_stopwords(self) -> List[str]:
        """Returns all currently active stopwords."""
        return sorted(list(self.stopwords))

    def reset_defaults(self, include_clinical: bool = True) -> None:
        """Resets to standard language stopwords."""
        self.include_clinical = include_clinical
        if self.language == "vi":
            self.stopwords = set(DEFAULT_VIETNAMESE_STOPWORDS if include_clinical else VIETNAMESE_GENERAL_STOPWORDS)
        else:
            self.stopwords = set(ENGLISH_STOPWORDS)
        self.custom_stopwords.clear()
