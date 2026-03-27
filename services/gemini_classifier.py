import base64
import json
import logging
import os
import re
from typing import Any

import google.generativeai as genai

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
    EXTRACTION_FIELDS_BY_CATEGORY = {
        "id proof": (
            "name",
            "id_type",
            "id_number",
            "date_of_birth",
            "phone_number",
            "address",
        ),
        "invoice/bill": (
            "bill_number",
            "bill_date",
            "customer_name",
            "address",
            "policy_number",
            "billing_items",
            "grand_total",
        ),
        "policy form": (
            "policy_number",
            "policy_issued_on",
            "period_of_insurance",
            "policy_status",
            "expiry_date",
            "customer_id",
        ),
        "discharge summary": (
            "patient_name",
            "date_of_admission",
            "date_of_discharge",
            "diagnosis",
        ),
    }

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

    @classmethod
    def normalise_extraction_category(cls, category: str) -> str:
        return cls._normalise_category(category)

    @classmethod
    def get_supported_extraction_categories(cls) -> tuple[str, ...]:
        return tuple(cls.EXTRACTION_FIELDS_BY_CATEGORY.keys())

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

    @classmethod
    def _build_extraction_prompt(cls, document_category: str) -> str:
        field_names = cls.EXTRACTION_FIELDS_BY_CATEGORY[document_category]
        json_lines = cls._build_extraction_json_lines(document_category, field_names)
        field_list = ", ".join(f'"{field_name}"' for field_name in field_names)

        return (
            "You are a document data-extraction assistant.\n"
            f'The document category is "{document_category}".\n'
            f"Extract ONLY these fields: {field_list}.\n"
            "If a field is not present, set its value to null.\n"
            "Respond with ONLY valid JSON (no markdown fences, no extra text):\n"
            "{\n"
            f"{json_lines}\n"
            "}\n\n"
            "Rules:\n"
            f'- Do not return any fields other than: {field_list}.\n'
            "- Dates should be in DD-MM-YYYY format if possible.\n"
            "- Keep names, IDs, addresses, and totals as written in the document when possible.\n"
            "- If a field is missing or unreadable, return null for that field.\n"
            "- For invoice/bill documents, include every line item you can find in billing_items.\n"
            "- If an invoice/bill line item contains multiple nested sub-items, return only the main parent billing item and ignore the sub-items.\n"
            "- Preserve status and diagnosis wording as written in the document when possible."
        )

    @staticmethod
    def _build_extraction_json_lines(
        document_category: str, field_names: tuple[str, ...]
    ) -> str:
        json_lines: list[str] = []
        for field_name in field_names:
            if document_category == "invoice/bill" and field_name == "billing_items":
                json_lines.extend(
                    [
                        '  "billing_items": [',
                        '    {"billing_item": "...", "quantity": "...", "rate": "...", "subtotal": "..."}',
                        "  ]",
                    ]
                )
                continue

            json_lines.append(f'  "{field_name}": "..."')

        return ",\n".join(json_lines)

    def extract_document(
        self,
        document_bytes: bytes,
        mime_type: str = "application/pdf",
        document_category: str = "policy form",
    ) -> dict[str, Any]:
        """Extract category-specific fields from a document using Gemini."""
        prompt = self._build_extraction_prompt(document_category)
        allowed_fields = self.EXTRACTION_FIELDS_BY_CATEGORY[document_category]
        doc_size_kb = len(document_bytes) / 1024
        logger.info(
            "Extracting data | category=%s mime_type=%s size=%.1f KB",
            document_category,
            mime_type,
            doc_size_kb,
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

        result = {field_name: parsed.get(field_name) for field_name in allowed_fields}
        if document_category == "invoice/bill":
            raw_items = result.get("billing_items")
            if isinstance(raw_items, list):
                result["billing_items"] = [
                    {
                        "billing_item": item.get("billing_item"),
                        "quantity": item.get("quantity"),
                        "rate": item.get("rate"),
                        "subtotal": item.get("subtotal"),
                    }
                    for item in raw_items
                    if isinstance(item, dict)
                ]
            else:
                result["billing_items"] = None

        logger.info(
            "Extraction complete | category=%s fields_found=%d/%d",
            document_category,
            sum(1 for value in result.values() if value is not None),
            len(allowed_fields),
        )
        return result
