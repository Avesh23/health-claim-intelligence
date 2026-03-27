import base64
import json
import logging
import os
import re

import google.generativeai as genai
from models.classification import ExtractedData, LineItem

logger = logging.getLogger(__name__)


class GeminiClassifier:
    CLASSIFICATION_LABELS = (
        "claim form",
        "discharge summary",
        "empty",
        "id proof",
        "invoice/bill",
        "proposal form",
        "policy form",
        "cashless authorisation",
        "other",
    )

    def __init__(self, api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=resolved_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.debug("GeminiClassifier initialised with model=gemini-2.5-flash")

    @staticmethod
    def _build_prompt() -> str:
        labels = "\n".join(f'- "{label}"' for label in GeminiClassifier.CLASSIFICATION_LABELS)
        label_options = " | ".join(GeminiClassifier.CLASSIFICATION_LABELS)
        return (
            "You are a health-insurance document page classification assistant.\n"
            "Analyse exactly one document page and classify it into exactly one of these labels:\n"
            f"{labels}\n\n"
            "If the page is blank or visually empty, return \"empty\".\n"
            "If the page contains mixed content, choose the dominant document type.\n"
            "Respond with ONLY valid JSON and no markdown fences:\n"
            "{\n"
            f'  "category": "{label_options}",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "Confidence must be a decimal between 0.0 and 1.0."
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[str, float | None]:
        """Return (category, confidence) from Gemini's classification response."""
        raw_text = text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        category = "other"
        confidence: float | None = None

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Falling back to regex classification parse | raw=%r", raw_text)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if lines:
                category = lines[0].strip('"').lower()
            if len(lines) >= 2:
                match = re.search(r"(\d+(?:\.\d+)?)", lines[1])
                if match:
                    raw = float(match.group(1))
                    confidence = raw / 100.0 if raw > 1.0 else raw
            return GeminiClassifier._normalise_category(category), confidence

        parsed_category = parsed.get("category")
        if isinstance(parsed_category, str) and parsed_category.strip():
            category = parsed_category.strip().lower()

        raw_confidence = parsed.get("confidence")
        if isinstance(raw_confidence, (int, float)):
            confidence = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            match = re.search(r"(\d+(?:\.\d+)?)", raw_confidence)
            if match:
                raw = float(match.group(1))
                confidence = raw / 100.0 if raw > 1.0 else raw

        if confidence is not None:
            confidence = max(0.0, min(confidence, 1.0))

        return GeminiClassifier._normalise_category(category), confidence

    @staticmethod
    def _normalise_category(category: str) -> str:
        raw = (category or "").strip().lower()
        if raw in GeminiClassifier.CLASSIFICATION_LABELS:
            return raw

        aliases = {
            "claim": "claim form",
            "claimform": "claim form",
            "claim_form": "claim form",
            "discharge": "discharge summary",
            "discharge_summary": "discharge summary",
            "blank": "empty",
            "id": "id proof",
            "id_proof": "id proof",
            "invoice": "invoice/bill",
            "bill": "invoice/bill",
            "invoice bill": "invoice/bill",
            "invoice_bill": "invoice/bill",
            "proposal": "proposal form",
            "proposal_form": "proposal form",
            "policy": "policy form",
            "policy_form": "policy form",
        }
        return aliases.get(raw, "other")

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
                response = self.model.generate_content(
                    [
                        {"inline_data": {"mime_type": mime_type, "data": encoded}},
                        prompt,
                    ]
                )
        except Exception as exc:
            logger.exception("Gemini classify call failed | mime_type=%s", mime_type)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = (getattr(response, "text", "") or "").strip()
        category, confidence = self._parse_response(raw_text)
        logger.info(
            "Classification result | category=%r confidence=%s",
            category,
            f"{confidence:.2f}" if confidence is not None else "n/a",
        )
        return category, confidence

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
            '- Keep amounts as strings exactly as printed (e.g. "Rs 12,345.00").\n'
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
                response = self.model.generate_content(
                    [
                        {"inline_data": {"mime_type": mime_type, "data": encoded}},
                        prompt,
                    ]
                )
        except Exception as exc:
            logger.exception("Gemini extract call failed | mime_type=%s", mime_type)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = (getattr(response, "text", "") or "").strip()
        logger.debug("Raw extraction response: %r", raw_text)

        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON: %s | raw=%r", exc, raw_text)
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

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

        logger.info(
            "Extraction complete | fields_found=%d/9",
            sum(1 for value in result.model_dump().values() if value is not None),
        )
        return result
