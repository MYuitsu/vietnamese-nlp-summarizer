"""Clinical EHR and Medical Text Summarization Engine with Dynamic Lexicon Management."""

import re
from typing import List, Dict, Optional
from nlp_summary.keywords.registry import ExtractorRegistry
from nlp_summary.models.medical import CategorizedMedicalKeyword, ClinicalSoapSummary
from nlp_summary.models.summary import LengthConstraint
from nlp_summary.summarizer.engine import ExtractiveSummarizerEngine


class MedicalSummarizerEngine:
    """Specialized Clinical Summarizer for Vietnamese & Multilingual Medical Records."""

    DEFAULT_CATEGORIES_MAP: Dict[str, Dict] = {
        "Triệu chứng & Lâm sàng": {
            "keywords": [
                "sốt", "ho", "khó_thở", "khó thở", "đau_ngực", "đau thắt ngực", "chóng_mặt", "mệt_mỏi", "buồn_nôn",
                "đau_đầu", "co_giật", "phù", "tiêu_chảy", "xuất_huyết", "mất_ngủ", "đau_bụng",
                "chest pain", "dyspnea", "fever", "cough", "fatigue", "headache", "dizziness"
            ],
            "color": "#3B82F6",  # Blue
        },
        "Chẩn đoán & Bệnh lý": {
            "keywords": [
                "viêm_phổi", "đái_tháo_đường", "đái tháo đường", "tăng_huyết_áp", "tăng huyết áp", "nhồi_máu_cơ_tim", "nhồi máu cơ tim", "suy_tim",
                "tai_biến", "đột_quỵ", "suy_thận", "viêm_gan", "hen_phế_quản", "nhiễm_trùng",
                "ung_thư", "viêm_ruột_thừa", "loét_dạ_dày", "rối_loạn_mỡ_máu",
                "pneumonia", "diabetes", "hypertension", "myocardial infarction", "stroke", "sepsis"
            ],
            "color": "#8B5CF6",  # Purple
        },
        "Thuốc & Phác đồ Điều trị": {
            "keywords": [
                "kháng_sinh", "insulin", "paracetamol", "aspirin", "amoxicillin", "kháng_viêm",
                "truyền_dịch", "phẫu_thuật", "nội_soi", "thở_oxy", "hạ_áp", "thuốc_chẹn_beta",
                "kháng_đông", "chăm_sóc_cấp_1", "theo_dõi_sinh_hiệu", "can thiệp", "stent", "clopidogrel",
                "antibiotics", "analgesic", "oxygen therapy", "angioplasty"
            ],
            "color": "#10B981",  # Green
        },
        "Cảnh báo & Dị ứng": {
            "keywords": [
                "dị_ứng", "dị ứng", "tiền_sử", "tiền sử", "chống_chỉ_định", "nguy_kịch", "nguy_hiểm", "sốc_phản_vệ",
                "suy_hô_hấp_cấp", "tụt_huyết_áp", "hôn_mê", "penicillin",
                "allergy", "anaphylaxis", "contraindication", "adverse effect"
            ],
            "color": "#EF4444",  # Red
        },
        "Chỉ số Xét nghiệm & Sinh hiệu": {
            "keywords": [
                "huyết_áp", "huyết áp", "sp_o2", "spo2", "nhịp_tim", "đường_huyết", "bạch_cầu", "hồng_cầu",
                "crp", "creatinine", "men_gan", "x_quang", "ct_scan", "mri", "điện_tim", "troponin", "ecg", "eeg",
                "blood pressure", "heart rate", "white blood cells", "ct scan"
            ],
            "color": "#F59E0B",  # Orange
        }
    }

    def __init__(self, engine: Optional[ExtractiveSummarizerEngine] = None, custom_lexicon: Optional[Dict] = None):
        self.engine = engine or ExtractiveSummarizerEngine(language="vi")
        self.categories_map = custom_lexicon or {k: dict(v) for k, v in self.DEFAULT_CATEGORIES_MAP.items()}

    def add_custom_term(self, term: str, category: str, color: Optional[str] = None) -> None:
        """Dynamically registers a new medical term into a category."""
        clean_term = term.lower().strip()
        if category not in self.categories_map:
            self.categories_map[category] = {
                "keywords": [],
                "color": color or "#6B7280"
            }
        if clean_term not in self.categories_map[category]["keywords"]:
            self.categories_map[category]["keywords"].append(clean_term)

    def remove_term(self, term: str, category: Optional[str] = None) -> None:
        """Removes a term from custom lexicon."""
        clean_term = term.lower().strip()
        for cat_name, cfg in self.categories_map.items():
            if category is None or category == cat_name:
                if clean_term in cfg["keywords"]:
                    cfg["keywords"].remove(clean_term)

    def categorize_keyword(self, kw_text: str, score: float) -> CategorizedMedicalKeyword:
        """Classifies extracted keywords into clinical categories using dynamic lexicon."""
        clean_kw = kw_text.lower().strip()

        for category, config in self.categories_map.items():
            for target in config["keywords"]:
                if target in clean_kw or clean_kw in target:
                    return CategorizedMedicalKeyword(
                        keyword=kw_text,
                        category=category,
                        raw_score=score,
                        badge_color=config["color"],
                    )

        # Default clinical concept
        return CategorizedMedicalKeyword(
            keyword=kw_text,
            category="Thông tin Y khoa khác",
            badge_color="#6B7280",
            raw_score=score,
        )

    def summarize_clinical_record(
        self,
        medical_record_text: str,
        patient_title: str = "Bệnh án Hội chẩn",
        algorithm: str = "textrank",
        target_sentences: int = 3,
    ) -> ClinicalSoapSummary:
        """Produces a structured clinical summary with SOAP categorization."""
        algo_name = (algorithm or "textrank").lower().strip()
        doc = self.engine.preprocessor.process(medical_record_text)
        extractor = ExtractorRegistry.get_extractor(algo_name)
        extracted_kws = extractor.extract_keywords(doc, top_k=12)

        # 1. Run extractive summarizer
        gen_sum = self.engine.summarize(
            text=medical_record_text,
            title=patient_title,
            algorithm=algo_name,
            top_k_keywords=12,
            length_constraint=LengthConstraint(mode="sentence_count", value=target_sentences),
            mmr_lambda=0.75,
        )

        # 2. Categorize all extracted keywords
        categorized_kws = [
            self.categorize_keyword(kw.keyword, kw.normalized_score)
            for kw in extracted_kws
        ]

        # 3. Always inspect ALL original sentences for critical alerts (Safety Guaranteed)
        alert_triggers = self.categories_map.get("Cảnh báo & Dị ứng", {}).get("keywords", [])
        assess_triggers = self.categories_map.get("Chẩn đoán & Bệnh lý", {}).get("keywords", [])
        plan_triggers = self.categories_map.get("Thuốc & Phác đồ Điều trị", {}).get("keywords", [])
        obj_triggers = self.categories_map.get("Chỉ số Xét nghiệm & Sinh hiệu", {}).get("keywords", [])

        alerts = []
        for sent in doc.sentences:
            s_text = sent.raw_text
            lower_s = s_text.lower()
            if any(k in lower_s for k in alert_triggers):
                if s_text not in alerts:
                    alerts.append(s_text)

        # 4. Classify extracted summary sentences into SOAP clinical dimensions
        subjective = []
        objective = []
        assessment = []
        plan = []

        for sent in gen_sum.selected_sentences:
            s_text = sent.raw_text
            lower_s = s_text.lower()

            if any(k in lower_s for k in assess_triggers):
                assessment.append(s_text)
            elif any(k in lower_s for k in plan_triggers):
                plan.append(s_text)
            elif any(k in lower_s for k in obj_triggers):
                objective.append(s_text)
            else:
                subjective.append(s_text)

        return ClinicalSoapSummary(
            patient_context=patient_title,
            subjective_symptoms=subjective,
            objective_findings=objective,
            assessment_diagnosis=assessment,
            plan_treatment=plan,
            critical_alerts=alerts,
            categorized_keywords=categorized_kws,
            extractive_summary=gen_sum,
        )
