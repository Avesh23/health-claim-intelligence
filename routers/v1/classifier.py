import asyncio
import logging

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from models.classification import (
    BatchClassificationResponse,
    BatchExtractionResponse,
    ExtractedData,
    FileClassificationResult,
    FileExtractionResult,
    PageClassificationResult,
)
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
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
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
async def classify_bill_document(
    files: list[UploadFile] = File(...),
    classifier: GeminiClassifier = Depends(get_classifier),
):
    """
    Upload one or more documents (PDF, image, or plain text).
    Returns page-wise classification results for each file.
    """

    async def _classify_one(file: UploadFile) -> FileClassificationResult:
        filename = file.filename or "unknown"
        content_type = file.content_type or "application/octet-stream"
        mime_type = ALLOWED_MIME_TYPES.get(content_type)
        if mime_type is None:
            logger.warning(
                "Unsupported file type | filename=%s mime=%s", filename, content_type
            )
            return FileClassificationResult(
                filename=filename,
                error=f"Unsupported file type: {content_type}",
            )

        try:
            content = await file.read()
        except Exception as exc:
            logger.exception("Failed to read file | filename=%s", filename)
            return FileClassificationResult(
                filename=filename,
                error=f"Read error: {exc}",
            )

        logger.info(
            "File received | filename=%s size=%d bytes mime_type=%s",
            filename,
            len(content),
            mime_type,
        )

        try:
            if mime_type == "application/pdf":
                page_inputs = _render_pdf_pages(content)
            else:
                if mime_type == "text/plain":
                    is_empty = _is_empty_text_page(content)
                else:
                    is_empty = _is_empty_image_page(content, mime_type)
                page_inputs = [(1, content, mime_type, is_empty)]
        except RuntimeError as exc:
            logger.error("Failed to prepare file for classification | filename=%s", filename)
            return FileClassificationResult(filename=filename, error=str(exc))

        tasks = [
            asyncio.to_thread(classifier.classify_document, page_bytes, page_mime)
            for _, page_bytes, page_mime, is_empty in page_inputs
            if not is_empty
        ]
        raw_results = iter(await asyncio.gather(*tasks, return_exceptions=True))

        pages: list[PageClassificationResult] = []
        for page_number, _, _, is_empty in page_inputs:
            if is_empty:
                pages.append(
                    PageClassificationResult(
                        page_number=page_number,
                        category="empty",
                        confidence=1.0,
                    )
                )
                continue

            raw_result = next(raw_results)
            if isinstance(raw_result, Exception):
                logger.error(
                    "Gemini error | filename=%s page=%d error=%s",
                    filename,
                    page_number,
                    raw_result,
                )
                pages.append(
                    PageClassificationResult(
                        page_number=page_number,
                        category="other",
                        confidence=None,
                        error=f"Gemini error: {raw_result}",
                    )
                )
                continue

            category, confidence = raw_result
            logger.info(
                "Classified | filename=%s page=%d category=%r confidence=%s",
                filename,
                page_number,
                category,
                f"{confidence:.2f}" if confidence is not None else "n/a",
            )
            pages.append(
                PageClassificationResult(
                    page_number=page_number,
                    category=category,
                    confidence=confidence,
                )
            )

        return FileClassificationResult(filename=filename, pages=pages)

    results = await asyncio.gather(*[_classify_one(file) for file in files])
    return BatchClassificationResponse(results=list(results))


@router.post(
    "/extract",
    response_model=BatchExtractionResponse,
    summary="Extract structured data from one or more documents",
)
async def extract_document_data(
    files: list[UploadFile] = File(...),
    classifier: GeminiClassifier = Depends(get_classifier),
):
    """
    Upload one or more documents (PDF, image, or plain text).
    Returns extracted fields (member_id, policy_number, line_items, etc.)
    for each file.
    """

    async def _extract_one(file: UploadFile) -> FileExtractionResult:
        filename = file.filename or "unknown"
        content_type = file.content_type or "application/octet-stream"
        mime_type = ALLOWED_MIME_TYPES.get(content_type)
        if mime_type is None:
            logger.warning(
                "Unsupported file type | filename=%s mime=%s", filename, content_type
            )
            return FileExtractionResult(
                filename=filename,
                data=ExtractedData(),
            )

        try:
            content = await file.read()
        except Exception:
            logger.exception("Failed to read file | filename=%s", filename)
            return FileExtractionResult(
                filename=filename,
                data=ExtractedData(),
            )

        logger.info(
            "File received for extraction | filename=%s size=%d bytes mime_type=%s",
            filename,
            len(content),
            mime_type,
        )

        try:
            extracted = await asyncio.to_thread(
                classifier.extract_document, content, mime_type
            )
        except RuntimeError as exc:
            logger.error("Gemini extraction error | filename=%s error=%s", filename, exc)
            return FileExtractionResult(
                filename=filename,
                data=ExtractedData(),
            )

        logger.info("Extraction done | filename=%s", filename)
        return FileExtractionResult(
            filename=filename,
            data=extracted,
        )

    results = await asyncio.gather(*[_extract_one(file) for file in files])
    return BatchExtractionResponse(results=list(results))
