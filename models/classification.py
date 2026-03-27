from typing import Any, Optional
from pydantic import BaseModel, Field


class PageClassificationResult(BaseModel):
    page_number: int
    category: str
    confidence: Optional[float] = None
    error: Optional[str] = None


class FileClassificationResult(BaseModel):
    filename: str
    pages: list[PageClassificationResult] = Field(default_factory=list)
    error: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    results: list[FileClassificationResult]


class FileExtractionResult(BaseModel):
    filename: str
    document_category: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class BatchExtractionResponse(BaseModel):
    results: list[FileExtractionResult]
