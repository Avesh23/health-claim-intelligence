import base64
import json
import logging
import os
import re

import google.generativeai as genai
from models.classification import ExtractedData, LineItem

logger = logging.getLogger(__name__)


class GeminiClassifier:
    def __init__(self, api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=resolved_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.debug("GeminiClassifier initialised with model=gemini-2.5-flash")

    # ── Classification ──────────────────────────────────────────────

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
        prompt = self._build_prompt()
        doc_size_kb = len(document_bytes) / 1024
        logger.info(
            "Classifying document | mime_type=%s size=%.1f KB", mime_type, doc_size_kb
        )

        try:
            if mime_type == "text/plain":
                text_content = document_bytes.decode("utf-8", errors="replace")
                response = self.model.generate_content([prompt, text_content])
            else:
                encoded = base64.standard_b64encode(document_bytes).decode("utf-8")
                response = self.model.generate_content([
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                    prompt,
                ])
        except Exception as exc:
            logger.exception("Gemini classify call failed | mime_type=%s", mime_type)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = (getattr(response, "text", "") or "").strip()
        category, confidence = self._parse_response(raw_text)
        logger.info("Classification result | category=%r confidence=%s", category,
                     f"{confidence:.2f}" if confidence is not None else "n/a")
        return category, confidence

    # ── Data Extraction ─────────────────────────────────────────────

    @staticmethod
    def _build_extraction_prompt() -> str:
        return (
            "You are a document data-extraction assistant.\n"
            "Extract the following fields from the document. If a field is not present, "
            "set its value to null.\n\n"
            "Respond with ONLY valid JSON (no markdown fences, no extra text):\n"
            "{\n"
            '  "member_id": "...",\n'
            '  "policy_number": "...",\n'
            '  "claim_date": "...",\n'
            '  "treatment_date": "...",\n'
            '  "claimed_amount": "...",\n'
            '  "line_items": [\n'
            '    {"description": "...", "amount": "...", "quantity": "..."}\n'
            "  ],\n"
            '  "signature": "present / absent / unclear",\n'
            '  "location": "...",\n'
            '  "bank_amount": "..."\n'
            "}\n\n"
            "Rules:\n"
            "- Keep amounts as strings exactly as printed (e.g. \"₹12,345.00\").\n"
            "- Dates should be in DD-MM-YYYY format if possible.\n"
            "- line_items is a list; include every billable item you can find.\n"
            "- For signature, state whether a signature is present, absent, or unclear."
        )

    def extract_document(
        self, document_bytes: bytes, mime_type: str = "application/pdf"
    ) -> ExtractedData:
        """Extract structured fields from a document using Gemini."""
        prompt = self._build_extraction_prompt()
        doc_size_kb = len(document_bytes) / 1024
        logger.info(
            "Extracting data | mime_type=%s size=%.1f KB", mime_type, doc_size_kb
        )

        try:
            if mime_type == "text/plain":
                text_content = document_bytes.decode("utf-8", errors="replace")
                response = self.model.generate_content([prompt, text_content])
            else:
                encoded = base64.standard_b64encode(document_bytes).decode("utf-8")
                response = self.model.generate_content([
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                    prompt,
                ])
        except Exception as exc:
            logger.exception("Gemini extract call failed | mime_type=%s", mime_type)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = (getattr(response, "text", "") or "").strip()
        logger.debug("Raw extraction response: %r", raw_text)

        # Strip markdown code fences if Gemini adds them
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON: %s | raw=%r", exc, raw_text)
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

        # Build LineItem objects
        raw_items = parsed.get("line_items") or []
        line_items = [
            LineItem(
                description=item.get("description"),
                amount=item.get("amount"),
                quantity=item.get("quantity"),
            )
            for item in raw_items
            if isinstance(item, dict)
        ]

        result = ExtractedData(
            member_id=parsed.get("member_id"),
            policy_number=parsed.get("policy_number"),
            claim_date=parsed.get("claim_date"),
            treatment_date=parsed.get("treatment_date"),
            claimed_amount=parsed.get("claimed_amount"),
            line_items=line_items or None,
            signature=parsed.get("signature"),
            location=parsed.get("location"),
            bank_amount=parsed.get("bank_amount"),
        )

        logger.info("Extraction complete | fields_found=%d/9",
                     sum(1 for v in result.model_dump().values() if v is not None))
        return result

