import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types

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
    
    LABEL_DESCRIPTIONS = {
        "claim form": "A formal request to an insurance company for payment, usually contains sections for member details, treatment details, and a signature area.",
        "discharge summary": "A clinical report prepared by a physician or other health professional at the conclusion of a hospital stay, detailing the admission, diagnosis, and treatment.",
        "empty": "A blank page or a page with no meaningful text/graphics.",
        "id proof": "Identification documents like Passport, Aadhaar, PAN card, or Driver's license.",
        "invoice/bill": "A detailed list of goods or services provided, with individual costs and a total amount due.",
        "proposal form": "An application for insurance coverage, filled out by the prospective policyholder.",
        "policy form": "A document detailing the terms and conditions of the insurance policy, often includes the policy schedule.",
        "cashless authorisation": "A letter or document from the insurance company or TPA authorizing the hospital to provide treatment without upfront payment by the patient.",
        "other": "Any other document that does not fit the above categories."
    }

    EXTRACTION_FIELDS_BY_CATEGORY = {
        "cashless authorisation": (
            "claim_reference_number",
            "policy_name",
            "patient_name",
            "policy_number",
            "policy_period",
            "date_and_time",
            "final_approved_amount",
        ),
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
        "proposal form": (
            "name",
            "gender",
            "address",
            "phone_number",
            "period_of_insurance",
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
        "claim form": (
            "member_id",
            "policy_number",
            "claim_date",
            "treatment_date",
            "claimed_amount",
            "location",
            "bank_amount",
            "signature_status",
            "line_items",
        ),
    }

    def __init__(self, api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=resolved_api_key)
        self.model_id = "gemini-2.0-flash"
        logger.debug("GeminiClassifier initialised with model=%s", self.model_id)

    @staticmethod
    def _build_prompt() -> str:
        descriptions = "\n".join([f'- {label.upper()}: {desc}' for label, desc in GeminiClassifier.LABEL_DESCRIPTIONS.items()])
        label_options = " | ".join(GeminiClassifier.CLASSIFICATION_LABELS)
        return (
            "SYSTEM INSTRUCTION:\n"
            "You are an expert document classifier for a health insurance company.\n"
            "You will be provided with one or more images representing pages of a SINGLE document.\n"
            "Analyze all pages and provide a single classification for the entire document.\n\n"
            "CATEGORIES AND DESCRIPTIONS:\n"
            f"{descriptions}\n\n"
            "CRITICAL RULES:\n"
            "1. Choose the single most accurate category for the WHOLE document.\n"
            "2. If the document has mixed pages (e.g., a bill and an ID proof), choose the most 'important' one (usually the form or bill).\n"
            "3. Respond ONLY with a valid JSON object.\n\n"
            "RESPONSE FORMAT:\n"
            "{\n"
            f'  "category": "{label_options}",\n'
            '  "confidence": 0.0\n'
            "}\n"
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

    @classmethod
    def supports_extraction_category(cls, category: str) -> bool:
        return category in cls.EXTRACTION_FIELDS_BY_CATEGORY

    def classify_document(
        self, pages: list[tuple[bytes, str]]
    ) -> tuple[str, float | None]:
        """Classify a multi-page document as a single unit."""
        prompt = self._build_prompt()
        logger.info("Classifying document with %d pages", len(pages))

        try:
            parts = [prompt]
            for page_bytes, page_mime in pages:
                if page_mime == "text/plain":
                    parts.append(page_bytes.decode("utf-8", errors="replace"))
                else:
                    parts.append(types.Part.from_bytes(data=page_bytes, mime_type=page_mime))

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "category": {"type": "STRING"},
                            "confidence": {"type": "NUMBER"}
                        },
                        "required": ["category", "confidence"]
                    }
                )
            )
        except Exception as exc:
            logger.exception("Gemini multi-page classify call failed")
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = response.text or ""
        category, confidence = self._parse_response(raw_text)
        logger.info("Document classification result | category=%r", category)
        return category, confidence

    @classmethod
    def _build_extraction_prompt(cls, document_category: str) -> str:
        field_names = cls.EXTRACTION_FIELDS_BY_CATEGORY[document_category]
        field_list = ", ".join(f'"{field_name}"' for field_name in field_names)

        return (
            "SYSTEM INSTRUCTION:\n"
            "You are a highly accurate data extraction specialist for medical and insurance documents.\n"
            "You will be provided with multiple pages of a SINGLE document.\n"
            "Your goal is to extract information across ALL pages and return ONE unified JSON object.\n\n"
            f"DOCUMENT CATEGORY: {document_category.upper()}\n"
            f"FIELDS TO EXTRACT: {field_list}\n\n"
            "EXTRACTION RULES:\n"
            "1. Consolidate data from all pages. If a field appears on multiple pages (like Policy Number), return it once.\n"
            "2. If information for a field is spread across pages (e.g., a table), merge them into one complete list.\n"
            "3. For each field, provide a 'value' and a 'confidence' score.\n"
            "4. Format dates as DD-MM-YYYY.\n"
            "5. If a field is missing, use null/0.0.\n"
            "6. Respond ONLY with valid JSON.\n"
        )

    def _get_extraction_schema(self, document_category: str) -> dict[str, Any]:
        field_names = self.EXTRACTION_FIELDS_BY_CATEGORY.get(document_category, [])
        
        # Helper for a single field with confidence
        def field_with_confidence():
            return {
                "type": "OBJECT",
                "properties": {
                    "value": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "NUMBER"}
                },
                "required": ["value", "confidence"]
            }

        properties = {}
        for field in field_names:
            if field in ("billing_items", "line_items"):
                item_properties = {
                    "quantity": field_with_confidence(),
                    "rate": field_with_confidence(),
                    "subtotal": field_with_confidence(),
                }
                if field == "billing_items":
                    item_properties["billing_item"] = field_with_confidence()
                else:
                    item_properties["description"] = field_with_confidence()
                
                properties[field] = {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": item_properties,
                        "required": list(item_properties.keys())
                    }
                }
            else:
                properties[field] = field_with_confidence()
        
        return {
            "type": "OBJECT",
            "properties": properties,
            "required": list(field_names)
        }

    def extract_document(
        self,
        pages: list[tuple[bytes, str]],
        document_category: str = "policy form",
    ) -> dict[str, Any]:
        """Extract data from a multi-page document into a single consolidated result."""
        prompt = self._build_extraction_prompt(document_category)
        allowed_fields = self.EXTRACTION_FIELDS_BY_CATEGORY[document_category]
        logger.info("Extracting data from %d pages | category=%s", len(pages), document_category)

        try:
            parts = [prompt]
            for page_bytes, page_mime in pages:
                if page_mime == "text/plain":
                    parts.append(page_bytes.decode("utf-8", errors="replace"))
                else:
                    parts.append(types.Part.from_bytes(data=page_bytes, mime_type=page_mime))

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._get_extraction_schema(document_category)
                )
            )
        except Exception as exc:
            logger.exception("Gemini multi-page extract call failed")
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raw_text = response.text or "{}"
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON: %s | raw=%r", exc, raw_text)
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

        # Filtering logic
        result = {}
        for field_name in allowed_fields:
            field_data = parsed.get(field_name)
            if field_name in ("billing_items", "line_items"):
                if isinstance(field_data, list):
                    item_fields = ["quantity", "rate", "subtotal"]
                    if field_name == "billing_items":
                        item_fields.append("billing_item")
                    else:
                        item_fields.append("description")
                    result[field_name] = [
                        {k: item.get(k) for k in item_fields}
                        for item in field_data if isinstance(item, dict)
                    ]
                else:
                    result[field_name] = None
            else:
                result[field_name] = field_data

        return result
