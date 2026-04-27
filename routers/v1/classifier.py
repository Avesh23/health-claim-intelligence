import asyncio
import logging

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Request

from models.classification import (
    BatchClassificationResponse,
    BatchExtractionResponse,
    FileClassificationResult,
    FileExtractionResult,
)
from services.gemini_classifier import GeminiClassifier
from core.security import get_api_key
from core.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

ALLOWED_MIME_TYPES = {
    "application/pdf": "application/pdf",
    "text/plain": "text/plain",
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/webp": "image/webp",
    "image/heic": "image/heic",
}


def _is_visually_empty_pixmap(pixmap: fitz.Pixmap, threshold: int = 250) -> bool:
    samples = pixmap.samples
    channel_count = pixmap.n
    if channel_count < 1:
        return True

    for index in range(0, len(samples), channel_count):
        if channel_count >= 3:
            if (
                samples[index] < threshold
                or samples[index + 1] < threshold
                or samples[index + 2] < threshold
            ):
                return False
            continue

        if samples[index] < threshold:
            return False

    return True


def _is_empty_text_page(document_bytes: bytes) -> bool:
    return not document_bytes.decode("utf-8", errors="replace").strip()


def _is_empty_image_page(document_bytes: bytes, mime_type: str) -> bool:
    filetype = mime_type.split("/")[-1]

    try:
        document = fitz.open(stream=document_bytes, filetype=filetype)
    except Exception:
        logger.warning("Failed to inspect image for blank-page detection | mime=%s", mime_type)
        return False

    try:
        if document.page_count < 1:
            return True

        pixmap = document.load_page(0).get_pixmap(alpha=False)
        return _is_visually_empty_pixmap(pixmap)
    finally:
        document.close()


def get_classifier() -> GeminiClassifier:
    try:
        return GeminiClassifier()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def _render_pdf_pages(document_bytes: bytes) -> list[tuple[int, bytes, str, bool]]:
    try:
        document = fitz.open(stream=document_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"PDF parse error: {exc}") from exc

    page_images: list[tuple[int, bytes, str, bool]] = []
    for page_number in range(document.page_count):
        page = document.load_page(page_number)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        page_has_content = bool(page.get_text("text").strip()) or bool(page.get_drawings())
        is_empty = not page_has_content and _is_visually_empty_pixmap(pixmap)
        page_images.append((page_number + 1, pixmap.tobytes("png"), "image/png", is_empty))

    if not page_images:
        raise RuntimeError("PDF parse error: no pages found")

    document.close()
    return page_images


@router.post(
    "/classify",
    response_model=BatchClassificationResponse,
    summary="Classify one or more documents",
)
@limiter.limit("5/minute")
async def classify_bill_document(
    request: Request,
    files: list[UploadFile] = File(...),
    classifier: GeminiClassifier = Depends(get_classifier),
):
    """
    Upload one or more documents (PDF, image, or plain text).
    Returns a single classification for each file (multi-page analysis).
    """

    async def _classify_one(file: UploadFile) -> FileClassificationResult:
        filename = file.filename or "unknown"
        content_type = file.content_type or "application/octet-stream"
        mime_type = ALLOWED_MIME_TYPES.get(content_type)
        if mime_type is None:
            return FileClassificationResult(filename=filename, category="other", error=f"Unsupported file type: {content_type}")

        try:
            content = await file.read()
            if mime_type == "application/pdf":
                page_data = _render_pdf_pages(content)
                pages = [(pb, pm) for _, pb, pm, empty in page_data if not empty]
            else:
                pages = [(content, mime_type)]
            
            if not pages:
                return FileClassificationResult(filename=filename, category="empty", confidence=1.0)

            category, confidence = await asyncio.to_thread(classifier.classify_document, pages)
            return FileClassificationResult(filename=filename, category=category, confidence=confidence)
        except Exception as exc:
            logger.exception("Classification failed for %s", filename)
            return FileClassificationResult(filename=filename, category="other", error=str(exc))

    results = await asyncio.gather(*[_classify_one(file) for file in files])
    return BatchClassificationResponse(results=list(results))


@router.post(
    "/extract",
    response_model=BatchExtractionResponse,
    summary="Extract structured data from one or more documents",
)
@limiter.limit("5/minute")
async def extract_document_data(
    request: Request,
    files: list[UploadFile] = File(...),
    classifier: GeminiClassifier = Depends(get_classifier),
):
    """
    Upload one or more documents. Performs multi-page classification 
    followed by consolidated extraction.
    """
    async def _extract_one(file: UploadFile) -> FileExtractionResult:
        filename = file.filename or "unknown"
        content_type = file.content_type or "application/octet-stream"
        mime_type = ALLOWED_MIME_TYPES.get(content_type)
        if mime_type is None:
            return FileExtractionResult(filename=filename, document_category="other", error=f"Unsupported file type")

        try:
            content = await file.read()
            if mime_type == "application/pdf":
                page_data = _render_pdf_pages(content)
                pages = [(pb, pm) for _, pb, pm, empty in page_data if not empty]
            else:
                pages = [(content, mime_type)]

            if not pages:
                return FileExtractionResult(filename=filename, document_category="empty", error="Empty file")

            # 1. Classify the whole document
            category, confidence = await asyncio.to_thread(classifier.classify_document, pages)
            
            if not classifier.supports_extraction_category(category):
                return FileExtractionResult(
                    filename=filename, 
                    document_category=category, 
                    confidence=confidence,
                    error=f"No extraction schema for category: {category}"
                )

            # 2. Extract from the whole document
            data = await asyncio.to_thread(classifier.extract_document, pages, category)
            return FileExtractionResult(filename=filename, document_category=category, confidence=confidence, data=data)
            
        except Exception as exc:
            logger.exception("Extraction failed for %s", filename)
            return FileExtractionResult(filename=filename, document_category="other", error=str(exc))

    results = await asyncio.gather(*[_extract_one(file) for file in files])
    return BatchExtractionResponse(results=list(results))
