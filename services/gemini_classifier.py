import base64
import logging
import os
import re

import google.generativeai as genai


logger = logging.getLogger(__name__)


class GeminiClassifier:
    def __init__(self, api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=resolved_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.debug("GeminiClassifier initialised with model=gemini-2.5-flash")

    @staticmethod
    def _build_prompt() -> str:
        return (
            "You are a document classification assistant.\n"
            "Analyse the document and respond with EXACTLY two lines and nothing else:\n"
            "Line 1: A short, descriptive category name for this document "
            "(e.g. 'Invoice', 'Medical Report', 'Insurance Policy'). "
            "Be concise — 1 to 4 words.\n"
            "Line 2: Your confidence score as a decimal between 0.0 and 1.0 (e.g. 0.92)."
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[str, float | None]:
        """Return (category, confidence) from Gemini's two-line response."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        category = lines[0] if lines else "Uncategorized"
        confidence: float | None = None
        if len(lines) >= 2:
            match = re.search(r"(\d+(?:\.\d+)?)", lines[1])
            if match:
                raw = float(match.group(1))
                confidence = raw / 100.0 if raw > 1.0 else raw
        return category, confidence

    def classify_document(
        self, document_bytes: bytes, mime_type: str = "application/pdf"
    ) -> tuple[str, float | None]:
        """
        Classify a document and return (category, confidence).
        Gemini freely determines the category — no predefined list.
        """
        prompt = self._build_prompt()
        doc_size_kb = len(document_bytes) / 1024
        logger.info(
            "Classifying document | mime_type=%s size=%.1f KB", mime_type, doc_size_kb
        )

        try:
            if mime_type == "text/plain":
                text_content = document_bytes.decode("utf-8", errors="replace")
                logger.debug("Sending plain-text document to Gemini")
                response = self.model.generate_content([prompt, text_content])
            else:
                encoded = base64.standard_b64encode(document_bytes).decode("utf-8")
                logger.debug("Sending binary document to Gemini as inline base64")
                response = self.model.generate_content([
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                    prompt,
                ])
        except Exception as exc:
            logger.exception("Gemini API call failed | mime_type=%s", mime_type)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = (getattr(response, "text", "") or "").strip()
        logger.debug("Raw Gemini response: %r", raw_text)

        category, confidence = self._parse_response(raw_text)
        logger.info(
            "Classification result | category=%r confidence=%s",
            category,
            f"{confidence:.2f}" if confidence is not None else "n/a",
        )
        return category, confidence
