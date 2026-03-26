import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from models.classification import BatchClassificationResponse, FileClassificationResult
from services.gemini_classifier import GeminiClassifier

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf": "application/pdf",
    "text/plain": "text/plain",
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/webp": "image/webp",
    "image/heic": "image/heic",
}


def get_classifier() -> GeminiClassifier:
    try:
        return GeminiClassifier()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/classify",
    response_model=BatchClassificationResponse,
    summary="Classify one or more documents",
)
async def classify_bill_document(
    files: list[UploadFile] = File(...),
    classifier: GeminiClassifier = Depends(get_classifier),
):
    """
    Upload one or more documents (PDF, image, or plain text).
    Returns a classification result (filename, category, confidence) for each file.
    """
    async def _classify_one(file: UploadFile) -> FileClassificationResult:
        content_type = file.content_type or "application/octet-stream"
        mime_type = ALLOWED_MIME_TYPES.get(content_type)
        if mime_type is None:
            logger.warning(
                "Unsupported file type | filename=%s mime=%s", file.filename, content_type
            )
            return FileClassificationResult(
                filename=file.filename or "unknown",
                category=f"Unsupported file type: {content_type}",
                confidence=None,
            )

        try:
            content = await file.read()
        except Exception as exc:
            logger.exception("Failed to read file | filename=%s", file.filename)
            return FileClassificationResult(
                filename=file.filename or "unknown",
                category=f"Read error: {exc}",
                confidence=None,
            )

        logger.info(
            "File received | filename=%s size=%d bytes mime_type=%s",
            file.filename, len(content), mime_type,
        )

        try:
            category, confidence = classifier.classify_document(content, mime_type=mime_type)
        except RuntimeError as exc:
            logger.error("Gemini error | filename=%s error=%s", file.filename, exc)
            return FileClassificationResult(
                filename=file.filename or "unknown",
                category=f"Gemini error: {exc}",
                confidence=None,
            )

        logger.info(
            "Classified | filename=%s category=%r confidence=%s",
            file.filename, category,
            f"{confidence:.2f}" if confidence is not None else "n/a",
        )
        return FileClassificationResult(
            filename=file.filename or "unknown",
            category=category,
            confidence=confidence,
        )

    results = await asyncio.gather(*[_classify_one(f) for f in files])
    return BatchClassificationResponse(results=list(results))
