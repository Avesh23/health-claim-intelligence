from typing import Any, Optional
from pydantic import BaseModel, Field


class FileClassificationResult(BaseModel):
    filename: str
    category: str
    confidence: Optional[float] = None
    error: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    results: list[FileClassificationResult]


class FileExtractionResult(BaseModel):
    filename: str
    document_category: str
    confidence: Optional[float] = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class BatchExtractionResponse(BaseModel):
    results: list[FileExtractionResult]
