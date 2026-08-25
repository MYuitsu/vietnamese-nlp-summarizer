"""Document and Image OCR parsing utilities for TXT, PDF, DOCX, PNG, JPG, and JPEG."""

import io
from typing import Optional


class DocumentParser:
    """Extracts raw plain text from various file formats including scanned images."""

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        """Parses raw utf-8 / utf-16 text."""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        """Extracts text from PDF bytes."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
            return "\n\n".join(text)
        except Exception:
            return ""

    @staticmethod
    def parse_docx(file_bytes: bytes) -> str:
        """Extracts text from Word docx bytes."""
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(text)
        except Exception:
            return ""

    @staticmethod
    def parse_image(file_bytes: bytes, lang: str = "vie+eng") -> str:
        """Extracts text from Image bytes (PNG, JPG, JPEG) using OCR."""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(io.BytesIO(file_bytes))
            try:
                text = pytesseract.image_to_string(image, lang=lang)
            except Exception:
                # Fallback to default if vie language data not found
                text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"Không thể trích xuất văn bản từ ảnh: {str(e)}"

    @classmethod
    def extract_text(cls, file_name: str, file_bytes: bytes) -> str:
        """Route to appropriate parser based on file extension."""
        name = file_name.lower()
        if name.endswith(".pdf"):
            return cls.parse_pdf(file_bytes)
        elif name.endswith(".docx"):
            return cls.parse_docx(file_bytes)
        elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return cls.parse_image(file_bytes)
        else:
            return cls.parse_txt(file_bytes)
