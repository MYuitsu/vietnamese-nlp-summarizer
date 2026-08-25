"""Vietnamese & Multilingual abbreviation-aware sentence segmentation."""

import re
from typing import List
from nlp_summary.models.preprocessing import Sentence


class SentenceSegmenter:
    """Splits normalized text into sentences while protecting medical & general abbreviations."""

    # Common Vietnamese, Medical & Multilingual abbreviations to protect
    ABBREVIATIONS = [
        # Medical & Clinical Abbreviations
        "BS.", "ThS. BS.", "TS. BS.", "PGS. TS.", "GS. TS.", "BSCKI.", "BSCKII.",
        "BN.", "ĐTĐ.", "THA.", "NMCT.", "K.", "Rx.", "Dr.", "Prof.",
        # Medical Measurements / Formulas
        "mg.", "ml.", "g.", "kg.", "mcg.", "IU.", "mEq.", "mmol/l.", "mg/dl.",
        # General Geography & Administrative
        "TP.", "GS.", "TS.", "ThS.", "PGS.", "KS.", "NXB.",
        "Q.", "H.", "P.", "T.", "Thg.", "Tr.",
        "v.v.", "v.v", "e.g.", "i.e.", "Mr.", "Mrs.", "Ms."
    ]

    # Placeholder format
    PLACEHOLDER_PREFIX = "___ABBR_"

    @classmethod
    def split(cls, text: str) -> List[Sentence]:
        """Segments text into a list of Sentence objects."""
        if not text or not text.strip():
            return []

        # Step 1: Protect known abbreviations by replacing dots with placeholders
        working_text = text
        saved_abbrs = {}
        for idx, abbr in enumerate(cls.ABBREVIATIONS):
            if abbr in working_text:
                placeholder = f"{cls.PLACEHOLDER_PREFIX}{idx}___"
                saved_abbrs[placeholder] = abbr
                # Case-insensitive replacement of abbreviation
                pattern = re.compile(re.escape(abbr), re.IGNORECASE)
                working_text = pattern.sub(placeholder, working_text)

        # Step 2: Protect decimal numbers & lab ratios (e.g., 3.14, 0.45 ng/ml, SpO2 98.5%)
        def num_replacer(match):
            return match.group(0).replace(".", "___DOT___")
        working_text = re.sub(r"\b\d+\.\d+\b", num_replacer, working_text)

        # Step 3: Split by sentence terminal punctuation or paragraph breaks
        raw_parts = re.split(r"(?<=[.!?…\n])\s+", working_text)

        sentences: List[Sentence] = []
        current_offset = 0

        for raw_part in raw_parts:
            part = raw_part.strip()
            if not part:
                continue

            # Restore abbreviations and decimal dots
            for placeholder, original in saved_abbrs.items():
                part = part.replace(placeholder, original)
            part = part.replace("___DOT___", ".")

            # Find actual start and end offsets in original text if possible
            start_pos = text.find(part, current_offset)
            if start_pos == -1:
                start_pos = current_offset
            end_pos = start_pos + len(part)
            current_offset = end_pos

            sentences.append(
                Sentence(
                    index=len(sentences),
                    raw_text=part,
                    cleaned_text=part,
                    tokens=[],
                    start_char=start_pos,
                    end_char=end_pos
                )
            )

        return sentences
